from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from datetime import datetime, timezone

from core.config import load_config
from core.db import get_engine
from core.features import compute_features
from core.ml import predict
from core.risk import size_from_atr, ev_bps
from core.model_store import load_latest_models

# --- Performance / robustness knobs ---
SCORE_MAX_BARS = 700
FEATURES_UPSERT_ROWS = 5

# Riesgo fallback (si no podemos estimar ATR/price)
RISK_FALLBACK = 0.02  # 2%

ROBUST_MIN_NTEST = 60
ROBUST_MIN_SHARPE = 1.0
ROBUST_MIN_HITRATE = 0.52
ROBUST_MAX_DD_FLOOR = -0.35


def read_bars(engine, exchange: str, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            """
            select ts, open, high, low, close, volume
            from public.bars_1d
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
            from public.model_metrics_1d
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


def upsert_features(engine, exchange: str, symbol: str, feat: pd.DataFrame) -> int:
    if feat is None or feat.empty:
        return 0

    f = feat.copy().reset_index()
    if "ts" not in f.columns:
        f = f.rename(columns={"index": "ts"})

    f["ts"] = pd.to_datetime(f["ts"], utc=True)
    f["exchange"] = exchange
    f["symbol"] = symbol

    f = f.sort_values("ts").tail(FEATURES_UPSERT_ROWS)

    cols = ["exchange", "symbol", "ts", "rsi", "ema_fast", "ema_slow", "atr", "zscore", "ret_fwd"]
    for c in cols:
        if c not in f.columns:
            f[c] = pd.NA

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.features_1d(exchange,symbol,ts,rsi,ema_fast,ema_slow,atr,zscore,ret_fwd)
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
                insert into public.signals_1d(
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
    candidates.append(("XETR", inst.get("xetra_symbol")))
    for c in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", c))
    for y in inst.get("yahoo_candidates", []) or []:
        candidates.append(("YAHOO", y))
    candidates.append(("PRIMARY", inst.get("primary_symbol")))

    for ex, sym in candidates:
        if not ex or not sym:
            continue
        bars = read_bars(engine, ex, sym)
        if bars is not None and not bars.empty:
            print(f"[SERIES] {inst['name']} using {ex}:{sym} bars={len(bars)}")
            return ex, sym

    return None, None


def robust_ok_for_buy(m: dict | None) -> tuple[bool, str]:
    if not m:
        return False, "no metrics"

    n_test = m.get("n_test")
    sharpe = m.get("sharpe")
    max_dd = m.get("max_dd")
    hit = m.get("hit_rate")

    reasons = []
    ok = True

    if n_test is None or int(n_test) < ROBUST_MIN_NTEST:
        ok = False
        reasons.append(f"n_test<{ROBUST_MIN_NTEST}")
    if sharpe is None or float(sharpe) < ROBUST_MIN_SHARPE:
        ok = False
        reasons.append(f"sharpe<{ROBUST_MIN_SHARPE}")
    if hit is None or float(hit) < ROBUST_MIN_HITRATE:
        ok = False
        reasons.append(f"hit_rate<{ROBUST_MIN_HITRATE}")
    if max_dd is None or float(max_dd) < ROBUST_MAX_DD_FLOOR:
        ok = False
        reasons.append(f"max_dd<{ROBUST_MAX_DD_FLOOR}")

    return ok, ("ok" if ok else ", ".join(reasons))


def main():
    cfg = load_config()
    engine = get_engine()

    processed = 0
    horizon_days = int(cfg["ml"]["horizon_bars"])

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name
        asset_name = name

        ex, sym = resolve_series(engine, inst)
        if not ex or not sym:
            print(f"[SCORE SKIP] {name}: no daily bars in DB")
            continue

        bars = read_bars(engine, ex, sym)
        if bars is None or bars.empty:
            print(f"[SCORE SKIP] {name}: no daily bars in DB")
            continue

        bars = bars.tail(SCORE_MAX_BARS)

        feat = compute_features(bars, horizon_bars=horizon_days)
        if feat is None or feat.empty:
            print(f"[SCORE SKIP] {name}: features empty")
            continue

        upsert_features(engine, ex, sym, feat)

        clf, reg, meta = load_latest_models(engine, ex, sym)
        proba, ret_exp = predict(clf, reg, feat)

        last = feat.dropna().tail(1)
        risk_est = None
        if not last.empty and "atr" in last.columns:
            try:
                atr = float(last["atr"].iloc[0])
                price = float(bars["close"].iloc[-1])
                if price > 0 and pd.notna(atr):
                    risk_est = atr / price
            except Exception:
                risk_est = None

        # ✅ fallback para no dejar EV en null
        if risk_est is None:
            risk_est = RISK_FALLBACK

        ev = ev_bps(
            ret_exp,
            risk_est,
            float(cfg["backtest"]["fee_bps"]),
            float(cfg["backtest"]["slippage_bps"]),
        )

        action = "HOLD"
        if proba is not None and ret_exp is not None and ev is not None:
            if proba > 0.60 and ev > 2.0:
                action = "BUY"
            elif proba < 0.45 and ev < -2.0:
                action = "SELL"

        m = read_latest_metrics(engine, ex, sym)
        ok_buy, ok_reason = robust_ok_for_buy(m)
        if action == "BUY" and not ok_buy:
            action = "HOLD"

        size_eur = sl = tp = None
        if action == "BUY" and not last.empty and "atr" in last.columns:
            try:
                atr = float(last["atr"].iloc[0])
                price = float(bars["close"].iloc[-1])
                if pd.notna(atr) and price > 0:
                    size_eur, sl = size_from_atr(
                        capital_eur=float(cfg["signals"]["capital_eur"]),
                        risk_pct=float(cfg["signals"]["risk_per_trade_pct"]),
                        max_pos_pct=float(cfg["signals"]["max_position_pct"]),
                        price=price,
                        atr=atr,
                        sl_atr_mult=float(cfg["signals"]["sl_atr_mult"]),
                    )
                    tp = float(price + float(cfg["signals"]["tp_atr_mult"]) * atr)
            except Exception:
                size_eur = sl = tp = None

        ts = bars.index[-1] if len(bars) else datetime.now(timezone.utc)
        model_id = meta["model_id"] if meta else "no_model"

        expl = [
            f"{name}",
            f"series={ex}:{sym}",
            f"model={model_id}",
            f"proba={None if proba is None else round(float(proba),3)}",
            f"ret_exp={None if ret_exp is None else round(float(ret_exp),4)}",
            f"risk={round(float(risk_est),4)}",
            f"ev_bps={None if ev is None else round(float(ev),2)}",
            f"gate={ok_reason}",
        ]

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
            "explanation": " | ".join(expl),
            "model_id": model_id,
            "asset_id": asset_id,
            "asset_name": asset_name,
        })

        processed += 1
        print(f"[SCORE OK] {name} -> {action} ({ex}:{sym}) gate={ok_reason}")

    print(f"SCORE OK: {processed} inst")


if __name__ == "__main__":
    main()
