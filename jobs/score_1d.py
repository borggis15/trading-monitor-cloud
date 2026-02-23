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


ROBUST_MIN_NTEST = 80
ROBUST_MIN_SHARPE = 1.0
ROBUST_MIN_HITRATE = 0.52
ROBUST_MAX_DD_FLOOR = -0.35


def read_bars_1d(engine, exchange: str, symbol: str) -> pd.DataFrame:
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
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.dropna(subset=["ts"]).set_index("ts").sort_index()


def read_latest_metrics_1d(engine, exchange: str, symbol: str):
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


def upsert_signal_1d(engine, row: dict):
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


def resolve_series_1d(engine, inst: dict) -> tuple[str | None, str | None]:
    candidates: list[tuple[str, str]] = []
    for s in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", s))
    for y in inst.get("yahoo_candidates", []) or []:
        candidates.append(("YAHOO", y))

    for ex, sym in candidates:
        if not sym:
            continue
        bars = read_bars_1d(engine, ex, sym)
        if bars is not None and not bars.empty:
            print(f"[SERIES] {inst['name']} using {ex}:{sym} bars={len(bars)}")
            return ex, sym

    return None, None


def robust_ok_for_buy(m: dict | None) -> tuple[bool, str]:
    if not m:
        return False, "no_metrics"

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


def action_from_thresholds(proba_up, ev, meta: dict | None) -> tuple[str, dict]:
    thr = (meta or {}).get("thresholds") or {}
    buy_p = float(thr.get("buy_proba", 0.62))
    sell_p = float(thr.get("sell_proba", 0.40))
    buy_ev = float(thr.get("buy_ev_bps", 25.0))
    sell_ev = float(thr.get("sell_ev_bps", -25.0))

    decision = "HOLD"
    if proba_up is not None and ev is not None:
        if float(proba_up) >= buy_p and float(ev) >= buy_ev:
            decision = "BUY"
        elif float(proba_up) <= sell_p and float(ev) <= sell_ev:
            decision = "SELL"

    used = {"buy_p": buy_p, "sell_p": sell_p, "buy_ev": buy_ev, "sell_ev": sell_ev}
    return decision, used


def main():
    cfg = load_config()
    engine = get_engine()

    processed = 0
    horizon_days = int(cfg["ml"].get("horizon_bars", 10))  # fallback razonable

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name
        asset_name = name

        ex, sym = resolve_series_1d(engine, inst)
        if not ex or not sym:
            print(f"[SCORE SKIP] {name}: no daily bars in DB")
            continue

        bars = read_bars_1d(engine, ex, sym)

        feat = compute_features(bars, horizon_bars=horizon_days)
        if feat is None or feat.empty:
            print(f"[SCORE SKIP] {name}: empty features")
            continue

        clf, reg, meta = load_latest_models(engine, ex, sym, tf="1d")
        if clf is None or reg is None:
            print(f"[SCORE SKIP] {name}: no daily model yet")
            continue

        proba, ret_exp = predict(clf, reg, feat, features=(meta or {}).get("features"))
        last = feat.dropna().tail(1)

        risk_est = None
        if not last.empty and "atr" in last.columns:
            atr = float(last["atr"].iloc[0])
            price = float(bars["close"].iloc[-1])
            if price > 0:
                risk_est = atr / price

        ev = ev_bps(
            ret_exp,
            risk_est,
            float(cfg["backtest"]["fee_bps"]),
            float(cfg["backtest"]["slippage_bps"]),
        )

        pre_action, thr_used = action_from_thresholds(proba, ev, meta)

        m = read_latest_metrics_1d(engine, ex, sym)
        ok_buy, ok_reason = robust_ok_for_buy(m)

        action = pre_action
        if action == "BUY" and not ok_buy:
            action = "HOLD"

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

        ts = bars.index[-1] if len(bars) else datetime.now(timezone.utc)
        model_id = (meta or {}).get("model_id") or "no_model"

        expl = [
            f"name={name}",
            f"series={ex}:{sym}",
            f"model={model_id}",
            f"proba={None if proba is None else round(float(proba), 6)}",
            f"ret_exp={None if ret_exp is None else round(float(ret_exp), 6)}",
            f"risk_est={None if risk_est is None else round(float(risk_est), 6)}",
            f"ev_bps={None if ev is None else round(float(ev), 6)}",
            f"pre_action={pre_action}",
            f"thr(buy_p={thr_used['buy_p']},sell_p={thr_used['sell_p']},buy_ev={thr_used['buy_ev']},sell_ev={thr_used['sell_ev']})",
        ]
        if m:
            expl.append(
                f"robust(n={m.get('n_test')},sh={round(float(m.get('sharpe')),2)},dd={round(float(m.get('max_dd')),2)},hit={round(float(m.get('hit_rate')),2)})"
            )
        expl.append(f"buy_gate={ok_reason}")

        upsert_signal_1d(
            engine,
            {
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
            },
        )

        processed += 1
        print(f"[SCORE OK] {name} -> {action} ({ex}:{sym}) gate={ok_reason}")

    print(f"SCORE OK: {processed} inst")


if __name__ == "__main__":
    main()
