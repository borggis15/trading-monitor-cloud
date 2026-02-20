from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timezone

from core.config import load_config
from core.db import get_engine
from core.features import compute_features
from core.ml import train_models, predict
from core.model_store import save_models
from core.risk import ev_bps

FEATURES = ["rsi", "ema_fast", "ema_slow", "atr", "zscore"]


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


def resolve_series_1d(engine, inst: dict) -> tuple[str | None, str | None]:
    candidates: list[tuple[str, str | None]] = []

    # Para daily: normalmente STOOQ/YAHOO funcionan mejor que XETR (según tu setup actual)
    # Aun así, mantenemos XETR primero si existe.
    candidates.append(("XETR", inst.get("xetra_symbol")))

    for c in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", c))

    for y in inst.get("yahoo_candidates", []) or []:
        candidates.append(("YAHOO", y))

    candidates.append(("PRIMARY", inst.get("primary_symbol")))

    for ex, sym in candidates:
        if not sym:
            continue
        bars = read_bars_1d(engine, ex, sym)
        if bars is not None and not bars.empty:
            print(f"[SERIES] {inst['name']} using {ex}:{sym} bars={len(bars)}")
            return ex, sym

    return None, None


def compute_walk_forward_metrics(
    feat: pd.DataFrame,
    min_train: int,
    fee_bps: float,
    slippage_bps: float,
    max_oos_points: int = 252,  # daily: ~1 año de OOS
):
    df = feat.copy()
    need = FEATURES + ["ret_fwd"]
    if df is None or df.empty or any(c not in df.columns for c in need):
        return {
            "n_test": 0,
            "hit_rate": None,
            "avg_ret": None,
            "sharpe": None,
            "max_dd": None,
            "profit_factor": None,
            "notes": "missing columns or empty feat",
        }

    df = df.dropna(subset=need).copy()
    if len(df) < (min_train + 30):
        return {
            "n_test": 0,
            "hit_rate": None,
            "avg_ret": None,
            "sharpe": None,
            "max_dd": None,
            "profit_factor": None,
            "notes": f"too few rows after dropna: {len(df)}",
        }

    start = min_train
    end = len(df) - 1
    if (end - start) > max_oos_points:
        start = end - max_oos_points

    realized = []
    hits = 0
    pos_count = 0
    in_pos = False

    for i in range(start, end):
        train = df.iloc[:i].copy()
        test = df.iloc[i : i + 1].copy()
        if len(train) < min_train or test.empty:
            continue

        clf, reg, _ = train_models(train, model_type="hgb", min_rows=min_train)
        proba, ret_exp = predict(clf, reg, test)

        atr = float(test["atr"].iloc[0]) if np.isfinite(test["atr"].iloc[0]) else None
        risk_est = 0.02 if atr else None

        ev = ev_bps(ret_exp, risk_est, fee_bps, slippage_bps)

        signal = "HOLD"
        if proba is not None and ret_exp is not None and ev is not None:
            if proba > 0.60 and ev > 2.0:
                signal = "BUY"
            elif proba < 0.45 and ev < -2.0:
                signal = "SELL"

        if signal == "BUY":
            in_pos = True
        elif signal == "SELL":
            in_pos = False

        r = float(test["ret_fwd"].iloc[0])
        strat_r = r if in_pos else 0.0

        realized.append(strat_r)
        if strat_r != 0.0:
            pos_count += 1
            if strat_r > 0:
                hits += 1

    if len(realized) < 30:
        return {
            "n_test": int(len(realized)),
            "hit_rate": None,
            "avg_ret": None,
            "sharpe": None,
            "max_dd": None,
            "profit_factor": None,
            "notes": f"too few OOS samples: {len(realized)}",
        }

    rets = np.array(realized, dtype=float)
    avg = float(np.mean(rets))
    std = float(np.std(rets) + 1e-12)
    sharpe = float((avg / std) * np.sqrt(252)) if std > 0 else None

    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    max_dd = float(np.min(dd))

    gains = float(rets[rets > 0].sum())
    losses = float(abs(rets[rets < 0].sum()) + 1e-12)
    pf = float(gains / losses) if losses > 0 else None

    hit_rate = float(hits / pos_count) if pos_count > 0 else None

    return {
        "n_test": int(len(rets)),
        "hit_rate": hit_rate,
        "avg_ret": avg,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": pf,
        "notes": "",
    }


def insert_metrics_public(
    engine,
    exchange: str,
    symbol: str,
    model_id: str,
    trained_at: datetime,
    horizon: str,
    m: dict,
):
    # Guardamos en la MISMA tabla que 15m, pero con model_id que termina en _1d
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.model_metrics(
                  exchange, symbol, model_id, trained_at, horizon,
                  n_test, hit_rate, avg_ret, sharpe, max_dd, profit_factor, notes
                )
                values (
                  :exchange, :symbol, :model_id, :trained_at, :horizon,
                  :n_test, :hit_rate, :avg_ret, :sharpe, :max_dd, :profit_factor, :notes
                )
                """
            ),
            {
                "exchange": exchange,
                "symbol": symbol,
                "model_id": model_id,
                "trained_at": trained_at,
                "horizon": horizon,
                "n_test": m.get("n_test"),
                "hit_rate": m.get("hit_rate"),
                "avg_ret": m.get("avg_ret"),
                "sharpe": m.get("sharpe"),
                "max_dd": m.get("max_dd"),
                "profit_factor": m.get("profit_factor"),
                "notes": m.get("notes", ""),
            },
        )


def main():
    cfg = load_config()
    engine = get_engine()

    # Para 1d puedes tener en config "horizon_days_1d". Si no existe, usamos horizon_bars.
    ml = cfg.get("ml", {})
    horizon = int(ml.get("horizon_days_1d") or ml.get("horizon_bars") or 10)
    min_rows = int(ml.get("min_train_rows_1d") or ml.get("min_train_rows") or 200)
    model_type = str(ml.get("model_type", "hgb"))

    fee_bps = float(cfg["backtest"]["fee_bps"])
    slippage_bps = float(cfg["backtest"]["slippage_bps"])

    inserted = 0

    for inst in cfg["universe"]:
        name = inst["name"]

        ex, sym = resolve_series_1d(engine, inst)
        if not ex or not sym:
            print(f"[TRAIN SKIP] {name}: no daily bars in DB")
            continue

        bars = read_bars_1d(engine, ex, sym)
        if bars is None or bars.empty:
            print(f"[TRAIN SKIP] {name}: no daily bars")
            continue

        feat = compute_features(bars, horizon_bars=horizon)
        if feat is None or feat.empty:
            print(f"[TRAIN SKIP] {name}: empty features")
            continue

        clf, reg, train_rows = train_models(feat, model_type=model_type, min_rows=min_rows)
        model_id = f"{model_type}_{horizon}d_1d"
        meta = {"name": name, "train_rows": int(train_rows), "horizon_days": int(horizon), "timeframe": "1d"}

        # Guarda modelo en model_store_1d
        save_models(engine, ex, sym, model_id, clf, reg, meta, tf="1d")

        trained_at = datetime.now(timezone.utc)

        m = compute_walk_forward_metrics(
            feat,
            min_train=min_rows,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_oos_points=252,
        )

        # ✅ Aquí está la clave: métricas 1d a public.model_metrics
        insert_metrics_public(engine, ex, sym, model_id, trained_at, f"{horizon}d_1d", m)
        inserted += 1

        print(
            f"[METRICS OK] {name} {ex}:{sym} n_test={m.get('n_test')} "
            f"sharpe={m.get('sharpe')} max_dd={m.get('max_dd')} hit_rate={m.get('hit_rate')} notes={m.get('notes')}"
        )

    with engine.begin() as conn:
        c = conn.execute(text("select count(*) from public.model_metrics where model_id like '%\\_1d' escape '\\'")).scalar()

    print(f"[TRAIN DONE] metrics_inserted={inserted} total_1d_metrics_rows={c}")


if __name__ == "__main__":
    main()
