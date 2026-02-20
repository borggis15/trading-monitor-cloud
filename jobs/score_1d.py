# jobs/score_1d.py
from __future__ import annotations

from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import text

from core.config import load_config
from core.db import get_engine
from core.features import compute_features
from core.ml import predict
from core.risk import size_from_atr, ev_bps
from core.model_store import load_latest_models


# Robust gates (puedes ajustar luego, pero esto es “conservador-profesional”)
ROBUST_MIN_NTEST = 120
ROBUST_MIN_SHARPE = 1.0
ROBUST_MIN_HITRATE = 0.52
ROBUST_MAX_DD_FLOOR = -0.35


def _cfg_block(cfg: dict, key: str, fallback_key: str = "ml") -> dict:
    """
    Permite tener secciones específicas para 1d:
      - ml_1d, signals_1d, backtest_1d
    y si no existen, usa las generales:
      - ml, signals, backtest
    """
    return cfg.get(key) or cfg.get(fallback_key) or {}


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
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def read_latest_metrics_1d(engine, exchange: str, symbol: str):
    df = pd.read_sql(
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
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def upsert_signal_1d(engine, row: dict):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.signals_1d(
                  exchange, symbol, ts,
                  action, proba_up, ret_exp, risk_est, ev_bps,
                  size_eur, sl_price, tp_price, horizon, explanation, model_id,
                  asset_id, asset_name
                )
                values (
                  :exchange, :symbol, :ts,
                  :action, :proba_up, :ret_exp, :risk_est, :ev_bps,
                  :size_eur, :sl_price, :tp_price, :horizon, :explanation, :model_id,
                  :asset_id, :asset_name
                )
                on conflict (exchange, symbol, ts) do update set
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
    candidates = []

    # STOOQ primero en 1d (porque te está funcionando bien)
    for c in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", c))

    # YAHOO como alternativa
    for y in inst.get("yahoo_candidates", []) or []:
        candidates.append(("YAHOO", y))

    # XETR al final (en 1d puede que no esté en plan gratis)
    if inst.get("xetra_symbol"):
        candidates.append(("XETR", inst["xetra_symbol"]))

    # PRIMARY último
    if inst.get("primary_symbol"):
        candidates.append(("PRIMARY", inst["primary_symbol"]))

    for ex, sym in candidates:
        bars = read_bars_1d(engine, ex, sym)
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

    ok = True
    reasons = []

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

    ml = _cfg_block(cfg, "ml_1d", "ml")
    backtest = _cfg_block(cfg, "backtest_1d", "backtest")
    sig = _cfg_block(cfg, "signals_1d", "signals")

    horizon_days = int(ml.get("horizon_bars", 10))  # en 1d, “bars” == días
    fee_bps = float(backtest.get("fee_bps", 0.0))
    slippage_bps = float(backtest.get("slippage_bps", 0.0))

    processed = 0

    for inst in cfg["universe"]:
        name = inst["name"]
        asset_id = inst.get("isin") or inst.get("asset_id") or name
        asset_name = name

        ex, sym = resolve_series_1d(engine, inst)
        if not ex or not sym:
            print(f"[SCORE SKIP] {name}: no daily bars in DB")
            continue

        bars = read_bars_1d(engine, ex, sym)
        if bars is None or bars.empty:
            print(f"[SCORE SKIP] {name}: empty daily bars")
            continue

        # Para daily sí queremos histórico (pero controlado)
        # si tienes 2500, lo dejamos tal cual; si mañana crece, limitamos a 4000
        bars = bars.tail(4000)

        feat = compute_features(bars, horizon_bars=horizon_days)
        if feat is None or feat.empty:
            print(f"[SCORE SKIP] {name}: features empty")
            continue

        # ✅ Cargar modelos 1d desde model_store_1d
        clf, reg, meta = load_latest_models(engine, ex, sym, tf="1d")
        if clf is None or reg is None:
            print(f"[SCORE SKIP] {name}: no 1d model in model_store_1d for {ex}:{sym}")
            continue

        proba, ret_exp = predict(clf, reg, feat)

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
            fee_bps,
            slippage_bps,
        )

        # Reglas base (igual que 15m)
        action = "HOLD"
        if proba is not None and ret_exp is not None and ev is not None:
            if proba > 0.60 and ev > 2.0:
                action = "BUY"
            elif proba < 0.45 and ev < -2.0:
                action = "SELL"

        # Robust gate con métricas 1d
        m = read_latest_metrics_1d(engine, ex, sym)
        ok_buy, ok_reason = robust_ok_for_buy(m)
        if action == "BUY" and not ok_buy:
            action = "HOLD"

        size_eur = sl = tp = None
        if action == "BUY" and not last.empty and "atr" in last.columns:
            atr = float(last["atr"].iloc[0])
            price = float(bars["close"].iloc[-1])
            size_eur, sl = size_from_atr(
                capital_eur=float(sig.get("capital_eur", 10000)),
                risk_pct=float(sig.get("risk_per_trade_pct", 1.0)),
                max_pos_pct=float(sig.get("max_position_pct", 25.0)),
                price=price,
                atr=atr,
                sl_atr_mult=float(sig.get("sl_atr_mult", 2.0)),
            )
            tp = float(price + float(sig.get("tp_atr_mult", 3.0)) * atr)

        ts = bars.index[-1] if len(bars) else datetime.now(timezone.utc)
        model_id = (meta or {}).get("model_id") or "no_model"

        expl = [
            f"{name}",
            f"serie={ex}:{sym}",
            f"model={model_id}",
            f"proba={None if proba is None else round(float(proba),3)}",
            f"ret_exp={None if ret_exp is None else round(float(ret_exp),4)}",
            f"ev_bps={None if ev is None else round(float(ev),2)}",
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
