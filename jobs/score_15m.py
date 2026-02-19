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
    candidates = [("XETR", inst["xetra_symbol"])]
    for c in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", c))
    candidates.append(("PRIMARY", inst["primary_symbol"]))

    for ex, sym in candidates:
        bars = read_bars(engine, ex, sym)
        if bars is not None and not bars.empty:
            return ex, sym
    return None, None


def main():
    cfg = load_config()
    engine = get_engine()

    processed = 0

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name
        asset_name = name

        ex, sym = resolve_series(engine, inst)
        if not ex or not sym:
            print(f"[SCORE SKIP] {name}: sin barras en BD")
            continue

        bars = read_bars(engine, ex, sym)
        feat = compute_features(bars, horizon_bars=int(cfg["ml"]["horizon_bars"]))
        if feat is None or feat.empty:
            print(f"[SCORE SKIP] {name}: features vacío")
            continue

        upsert_features(engine, ex, sym, feat)

        clf, reg, meta = load_latest_models(engine, ex, sym)
        proba, ret_exp = predict(clf, reg, feat)

        risk_est = None
        last = feat.dropna().tail(1)
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

        action = "HOLD"
        if proba is not None and ret_exp is not None and ev is not None:
            if proba > 0.60 and ev > 2.0:
                action = "BUY"
            elif proba < 0.45 and ev < -2.0:
                action = "SELL"

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
        model_id = meta["model_id"] if meta else "no_model"

        explanation = f"{name} | serie={ex}:{sym} | model={model_id}"
        if meta and meta.get("trained_at"):
            explanation += f" | trained_at={meta['trained_at']}"

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
            "horizon": f"{int(cfg['ml']['horizon_bars'])}d",
            "explanation": explanation,
            "model_id": model_id,
            "asset_id": asset_id,
            "asset_name": asset_name,
        })

        processed += 1
        print(f"[SCORE OK] {name} -> {action} ({ex}:{sym})")

    print(f"SCORE OK: {processed} inst")


if __name__ == "__main__":
    main()
