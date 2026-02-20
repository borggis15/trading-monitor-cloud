from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from datetime import datetime, timezone, timedelta

from core.config import load_config
from core.db import get_engine
from core.features import compute_features
from core.ml import predict
from core.risk import size_from_atr, ev_bps
from core.model_store import load_latest_models

# --- Robustness gates (professional defaults)
ROBUST_MIN_NTEST = 80
ROBUST_MIN_SHARPE = 1.0
ROBUST_MIN_HITRATE = 0.52

# Instead of a single hard floor, we use two tiers:
# - HARD: blocks BUY entirely
# - SOFT: allows BUY but will reduce sizing / add warning
ROBUST_MAX_DD_HARD = -0.80
ROBUST_MAX_DD_SOFT = -0.55


def read_bars(engine, exchange: str, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            """
            select ts, open, high, low, close, volume
            from public.bars_15m
            where exchange=:exchange and symbol=:symbol
            order by ts
            """
        ),
        engine,
        params={"exchange": exchange, "symbol": symbol},
    )
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def read_latest_metrics(engine, exchange: str, symbol: str):
    row = pd.read_sql(
        text(
            """
            select sharpe, max_dd, hit_rate, profit_factor, n_test, trained_at, model_id
            from public.model_metrics
            where exchange=:exchange and symbol=:symbol
            order by trained_at desc
            limit 1
            """
        ),
        engine,
        params={"exchange": exchange, "symbol": symbol},
    )
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def upsert_features(engine, exchange: str, symbol: str, feat: pd.DataFrame):
    if feat is None or feat.empty:
        return 0

    f = feat.copy().reset_index()
    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f["exchange"] = exchange
    f["symbol"] = symbol

    cols = ["exchange", "symbol", "ts", "rsi", "ema_fast", "ema_slow", "atr", "zscore", "ret_fwd"]
    for c in cols:
        if c not in f.columns:
            f[c] = pd.NA

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.features_15m(exchange,symbol,ts,rsi,ema_fast,ema_slow,atr,zscore,ret_fwd)
                values (:exchange,:symbol,:ts,:rsi,:ema_fast,:ema_slow,:atr,:zscore,:ret_fwd)
                on conflict (exchange,symbol,ts) do update set
                  rsi=excluded.rsi,
                  ema_fast=excluded.ema_fast,
                  ema_slow=excluded.ema_slow,
                  atr=excluded.atr,
                  zscore=excluded.zscore,
                  ret_fwd=excluded.ret_fwd
                """
            ),
            f[cols].to_dict(orient="records"),
        )
    return len(f)


def upsert_signal(engine, row: dict):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.signals(
                  exchange,symbol,ts,
                  action,proba_up,ret_exp,risk_est,ev_bps,
                  size_eur,sl_price,tp_price,horizon,explanation,model_id,
                  asset_id, asset_name
                )
                values (
                  :exchange,:symbol,:ts,
                  :action,:proba_up,:ret_exp,:risk_est,:ev_bps,
                  :size_eur,:sl_price,:tp_price,:horizon,:explanation,:model_id,
                  :asset_id, :asset_name
                )
                on conflict (exchange,symbol,ts) do update set
                  action=excluded.action,
                  proba_up=excluded.proba_up,
                  ret_exp=excluded.ret_exp,
                  risk_est=excluded.risk_est,
                  ev_bps=excluded.ev_bps,
                  size_eur=excluded.size_eur,
                  sl_price=excluded.sl_price,
                  tp_price=excluded.tp_price,
                  horizon=excluded.horizon,
                  explanation=excluded.explanation,
                  model_id=excluded.model_id,
                  asset_id=excluded.asset_id,
                  asset_name=excluded.asset_name
                """
            ),
            row,
        )


def resolve_series(engine, inst: dict) -> tuple[str | None, str | None]:
    candidates = []
    candidates.append(("XETR", inst["xetra_symbol"]))
    for c in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", c))
    for y in inst.get("yahoo_candidates", []) or []:
        candidates.append(("YAHOO", y))
    candidates.append(("PRIMARY", inst["primary_symbol"]))

    for ex, sym in candidates:
        bars = read_bars(engine, ex, sym)
        if bars is not None and not bars.empty:
            print(f"[SERIES] {inst['name']} using {ex}:{sym} bars={len(bars)}")
            return ex, sym
    return None, None


def robust_gate(m: dict | None) -> tuple[str, str]:
    """
    Returns (tier, reason)
    tier: "none" (ok), "soft" (allow but cautious), "hard" (block BUY)
    """
    if not m:
        return "hard", "no_metrics"

    n_test = m.get("n_test")
    sharpe = m.get("sharpe")
    max_dd = m.get("max_dd")
    hit = m.get("hit_rate")

    reasons = []

    # Core minimums
    if n_test is None or int(n_test) < ROBUST_MIN_NTEST:
        reasons.append(f"n_test<{ROBUST_MIN_NTEST}")
    if sharpe is None or float(sharpe) < ROBUST_MIN_SHARPE:
        reasons.append(f"sharpe<{ROBUST_MIN_SHARPE}")
    if hit is None or float(hit) < ROBUST_MIN_HITRATE:
        reasons.append(f"hit_rate<{ROBUST_MIN_HITRATE}")

    # Drawdown tiering
    dd = None if max_dd is None else float(max_dd)
    if dd is None:
        reasons.append("max_dd_missing")
        return "soft", ",".join(reasons) if reasons else "soft"

    if dd < ROBUST_MAX_DD_HARD:
        reasons.append(f"max_dd<{ROBUST_MAX_DD_HARD}")
        return "hard", ",".join(reasons)
    if dd < ROBUST_MAX_DD_SOFT:
        reasons.append(f"max_dd<{ROBUST_MAX_DD_SOFT}")
        return "soft", ",".join(reasons) if reasons else "soft"

    # If only minor issues, keep soft; if none, ok
    if reasons:
        return "soft", ",".join(reasons)
    return "none", "ok"


def main():
    cfg = load_config()
    engine = get_engine()

    processed = 0
    horizon_days = int(cfg["ml"]["horizon_bars"])

    fee_bps = float(cfg["backtest"]["fee_bps"])
    slippage_bps = float(cfg["backtest"]["slippage_bps"])

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name
        asset_name = name

        ex, sym = resolve_series(engine, inst)
        if not ex or not sym:
            print(f"[SCORE SKIP] {name}: no bars in DB")
            continue

        bars = read_bars(engine, ex, sym).tail(1200)  # keep more context than 800
        if bars.empty:
            print(f"[SCORE SKIP] {name}: empty bars")
            continue

        feat = compute_features(bars, horizon_bars=horizon_days)
        if feat is None or feat.empty:
            print(f"[SCORE SKIP] {name}: empty features")
            continue

        upsert_features(engine, ex, sym, feat)

        clf, reg, meta = load_latest_models(engine, ex, sym)
        proba, ret_exp = predict(clf, reg, feat)

        last = feat.dropna().tail(1)
        risk_est = None
        if not last.empty and "atr" in last.columns:
            atr = float(last["atr"].iloc[0])
            price = float(bars["close"].iloc[-1])
            if price > 0 and atr > 0:
                risk_est = atr / price

        ev = ev_bps(ret_exp, risk_est, fee_bps, slippage_bps)

        # --- Signal logic (adds a neutral zone)
        action = "HOLD"
        if proba is not None and ret_exp is not None and ev is not None:
            # BUY zone: clearly positive
            if proba >= 0.62 and ev >= 5.0:
                action = "BUY"
            # SELL zone: clearly negative
            elif proba <= 0.40 and ev <= -5.0:
                action = "SELL"
            else:
                action = "HOLD"

        # Robustness gating for BUY
        m = read_latest_metrics(engine, ex, sym)
        tier, reason = robust_gate(m)

        if action == "BUY" and tier == "hard":
            action = "HOLD"

        # sizing
        size_eur = sl = tp = None
        if action == "BUY" and not last.empty and "atr" in last.columns:
            atr = float(last["atr"].iloc[0])
            price = float(bars["close"].iloc[-1])

            size_eur, sl = size_from_atr(
                capital_eur=float(cfg["signals"]["capital_eur"]),
                risk_pct=float(cfg["signals"]["risk_per_trade_pct"]),
                max_pos_pct=float(cfg["signals"]["max_position_pct"]),
                price=price,
                atr=atr,
                sl_atr_mult=float(cfg["signals"]["sl_atr_mult"]),
            )
            tp = float(price + float(cfg["signals"]["tp_atr_mult"]) * atr)

            # Soft tier: reduce size to be conservative
            if tier == "soft" and size_eur is not None:
                size_eur = float(size_eur) * 0.5

        ts = bars.index[-1] if len(bars) else datetime.now(timezone.utc)
        target_ts = ts + timedelta(days=horizon_days)
        model_id = meta["model_id"] if meta else "no_model"

        # Professional explanation, parseable
        expl_parts = [
            f"name={name}",
            f"series={ex}:{sym}",
            f"model_id={model_id}",
            f"horizon_days={horizon_days}",
            f"signal_time_utc={ts.strftime('%Y-%m-%d %H:%M:%S')}",
            f"target_time_utc={target_ts.strftime('%Y-%m-%d %H:%M:%S')}",
            f"proba_up={None if proba is None else round(float(proba),3)}",
            f"ret_exp={None if ret_exp is None else round(float(ret_exp),4)}",
            f"ev_bps={None if ev is None else round(float(ev),2)}",
            f"robust_tier={tier}",
            f"robust_reason={reason}",
            "exit_rule=SELL_signal_or_target_time",
        ]
        if m:
            try:
                expl_parts.append(
                    "metrics="
                    f"n_test:{m.get('n_test')},"
                    f"sharpe:{round(float(m.get('sharpe')),2)},"
                    f"max_dd:{round(float(m.get('max_dd')),2)},"
                    f"hit_rate:{round(float(m.get('hit_rate')),2)}"
                )
            except Exception:
                pass

        upsert_signal(engine, {
            "exchange": ex,
            "symbol": sym,
            "ts": ts,
            "action": action,
            "proba_up": proba,
            "ret_exp": ret_exp,
            "risk_est": risk_est,
            "ev_bps": ev,
            "size_eur": size_eur,
            "sl_price": sl,
            "tp_price": tp,
            "horizon": f"{horizon_days}d",
            "explanation": " | ".join(expl_parts),
            "model_id": model_id,
            "asset_id": asset_id,
            "asset_name": asset_name,
        })

        processed += 1
        print(f"[SCORE OK] {name} -> {action} ({ex}:{sym}) tier={tier} reason={reason}")

    print(f"SCORE OK: {processed} inst")


if __name__ == "__main__":
    main()
