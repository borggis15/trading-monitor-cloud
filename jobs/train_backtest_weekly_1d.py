from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import text
from datetime import datetime, timezone

from core.config import load_config
from core.db import get_engine
from core.features import compute_features
from core.ml import train_models, predict
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


def load_assets_from_bars(engine) -> list[dict]:
    # Train on whatever exists in bars_1d; this avoids mismatches.
    df = pd.read_sql(
        text(
            """
            select distinct exchange, symbol, asset_id, asset_name
            from public.bars_1d
            """
        ),
        engine,
    )
    return df.to_dict(orient="records")


def save_models_1d(engine, exchange: str, symbol: str, model_id: str, clf, reg, meta: dict):
    import pickle
    clf_blob = pickle.dumps(clf)
    reg_blob = pickle.dumps(reg)

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.model_store_1d(exchange,symbol,model_id,trained_at,clf_pickle,reg_pickle,meta)
                values (:exchange,:symbol,:model_id,now(),:clf_pickle,:reg_pickle,:meta::jsonb)
                """
            ),
            {
                "exchange": exchange,
                "symbol": symbol,
                "model_id": model_id,
                "clf_pickle": clf_blob,
                "reg_pickle": reg_blob,
                "meta": meta,
            },
        )


def compute_walk_forward_metrics(
    feat: pd.DataFrame,
    min_train: int,
    fee_bps: float,
    slippage_bps: float,
    max_oos_points: int = 260,   # ~1 year daily
    retrain_every: int = 10,
    rolling_train_window: int = 3000,
):
    df = feat.copy()
    need = FEATURES + ["ret_fwd"]
    if df is None or df.empty or any(c not in df.columns for c in need):
        return {"n_test": 0, "hit_rate": None, "avg_ret": None, "sharpe": None, "max_dd": None, "profit_factor": None, "notes": "missing_columns"}

    df = df.dropna(subset=need).copy()
    if len(df) < (min_train + 100):
        return {"n_test": 0, "hit_rate": None, "avg_ret": None, "sharpe": None, "max_dd": None, "profit_factor": None, "notes": f"too_few_rows={len(df)}"}

    start = min_train
    end = len(df) - 1
    if (end - start) > max_oos_points:
        start = end - max_oos_points

    realized = []
    hits = 0
    pos_count = 0
    in_pos = False
    trades = 0

    clf = reg = None

    for step_i, i in enumerate(range(start, end)):
        train_end = i
        train_start = max(0, train_end - rolling_train_window)
        train = df.iloc[train_start:train_end].copy()
        test = df.iloc[i:i+1].copy()

        if len(train) < min_train or test.empty:
            continue

        if clf is None or reg is None or (step_i % retrain_every == 0):
            clf, reg, _ = train_models(train, model_type="hgb", min_rows=min_train)

        proba, ret_exp = predict(clf, reg, test)

        atr = float(test["atr"].iloc[0]) if np.isfinite(test["atr"].iloc[0]) else None
        risk_est = 0.02 if atr else None
        ev = ev_bps(ret_exp, risk_est, fee_bps, slippage_bps)

        signal = "HOLD"
        if proba is not None and ret_exp is not None and ev is not None:
            if proba >= 0.62 and ev >= 5.0:
                signal = "BUY"
            elif proba <= 0.40 and ev <= -5.0:
                signal = "SELL"

        prev = in_pos
        if signal == "BUY":
            in_pos = True
        elif signal == "SELL":
            in_pos = False
        if (not prev) and in_pos:
            trades += 1

        r = float(test["ret_fwd"].iloc[0])
        strat_r = r if in_pos else 0.0
        realized.append(strat_r)

        if strat_r != 0.0:
            pos_count += 1
            if strat_r > 0:
                hits += 1

    if len(realized) < 60:
        return {"n_test": int(len(realized)), "hit_rate": None, "avg_ret": None, "sharpe": None, "max_dd": None, "profit_factor": None, "notes": "too_few_oos"}

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

    notes = "" if trades > 0 else "no_trades"
    return {"n_test": int(len(rets)), "hit_rate": hit_rate, "avg_ret": avg, "sharpe": sharpe, "max_dd": max_dd, "profit_factor": pf, "notes": notes}


def insert_metrics_1d(engine, exchange: str, symbol: str, model_id: str, trained_at: datetime, horizon: str, m: dict):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.model_metrics_1d(
                  exchange,symbol,model_id,trained_at,horizon,
                  n_test,hit_rate,avg_ret,sharpe,max_dd,profit_factor,notes
                )
                values (
                  :exchange,:symbol,:model_id,:trained_at,:horizon,
                  :n_test,:hit_rate,:avg_ret,:sharpe,:max_dd,:profit_factor,:notes
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

    horizon_days = int(cfg["ml"]["horizon_bars"])     # en daily, esto son días
    min_rows = max(200, int(cfg["ml"]["min_train_rows"]))  # mínimo sensato en daily
    model_type = cfg["ml"]["model_type"]

    fee_bps = float(cfg["backtest"]["fee_bps"])
    slippage_bps = float(cfg["backtest"]["slippage_bps"])

    inserted = 0
    assets = load_assets_from_bars(engine)

    for a in assets:
        ex = a["exchange"]
        sym = a["symbol"]
        name = a.get("asset_name") or f"{ex}:{sym}"

        bars = read_bars_1d(engine, ex, sym)
        if bars is None or bars.empty or len(bars) < min_rows + horizon_days + 50:
            print(f"[TRAIN_1D SKIP] {name}: not enough bars ({0 if bars is None else len(bars)})")
            continue

        feat = compute_features(bars, horizon_bars=horizon_days)
        if feat is None or feat.empty:
            print(f"[TRAIN_1D SKIP] {name}: empty features")
            continue

        clf, reg, train_rows = train_models(feat, model_type=model_type, min_rows=min_rows)
        model_id = f"{model_type}_{horizon_days}d_1d"
        meta = {"name": name, "train_rows": int(train_rows), "horizon_days": int(horizon_days), "timeframe": "1d"}

        save_models_1d(engine, ex, sym, model_id, clf, reg, meta)

        trained_at = datetime.now(timezone.utc)
        m = compute_walk_forward_metrics(feat, min_train=min_rows, fee_bps=fee_bps, slippage_bps=slippage_bps)

        insert_metrics_1d(engine, ex, sym, model_id, trained_at, f"{horizon_days}d", m)
        inserted += 1

        print(f"[METRICS_1D OK] {name} {ex}:{sym} n_test={m.get('n_test')} sharpe={m.get('sharpe')} max_dd={m.get('max_dd')} hit_rate={m.get('hit_rate')} notes={m.get('notes')}")

    print(f"[TRAIN_1D DONE] metrics_inserted={inserted}")


if __name__ == "__main__":
    main()
