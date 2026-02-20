# jobs/train_backtest_weekly_1d.py
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


# ----------------------------
# Table detection (daily bars)
# ----------------------------
def detect_daily_bars_table(engine) -> str:
    """
    Detects which daily bars table exists in Supabase.
    Tries common names: bars_1d, bars_daily.
    """
    candidates = ["bars_1d", "bars_daily"]
    q = """
    select table_name
    from information_schema.tables
    where table_schema='public' and table_name = any(:names)
    """
    df = pd.read_sql(text(q), engine, params={"names": candidates})
    if df.empty:
        # Fallback: keep bars_1d as default expected name
        return "bars_1d"
    # Prefer bars_1d if present
    names = set(df["table_name"].tolist())
    return "bars_1d" if "bars_1d" in names else df["table_name"].iloc[0]


def read_bars_daily(engine, table_name: str, exchange: str, symbol: str) -> pd.DataFrame:
    df = pd.read_sql(
        text(
            f"""
            select ts, open, high, low, close, volume
            from public.{table_name}
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
    df = df.dropna(subset=["ts"])
    return df.set_index("ts").sort_index()


def resolve_series(engine, table_name: str, inst: dict) -> tuple[str | None, str | None]:
    candidates: list[tuple[str, str | None]] = []

    # (1) STOOQ daily suele ser el mejor para 1D
    for c in (inst.get("stooq_candidates") or []):
        candidates.append(("STOOQ", c))

    # (2) YAHOO (si existe)
    for y in (inst.get("yahoo_candidates") or []):
        candidates.append(("YAHOO", y))

    # (3) XETR, PRIMARY como fallback
    candidates.append(("XETR", inst.get("xetra_symbol")))
    candidates.append(("PRIMARY", inst.get("primary_symbol")))

    for ex, sym in candidates:
        if not ex or not sym:
            continue
        bars = read_bars_daily(engine, table_name, ex, sym)
        if bars is not None and not bars.empty:
            print(f"[SERIES] {inst['name']} using {ex}:{sym} bars={len(bars)}")
            return ex, sym

    return None, None


def compute_walk_forward_metrics(
    feat: pd.DataFrame,
    min_train: int,
    fee_bps: float,
    slippage_bps: float,
    max_oos_points: int = 252,
):
    """
    Walk-forward OOS:
    - entrena hasta i
    - predice i
    - aplica misma lógica BUY/HOLD/SELL
    - evalúa equity con posición binaria
    """
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
            "notes": "feat vacío o faltan columnas",
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
            "notes": f"pocas filas tras dropna: {len(df)}",
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

        # riesgo: usa ATR/price si puede, si no, 2%
        risk_est = None
        try:
            atr = float(test["atr"].iloc[0])
            price = float(test["close"].iloc[0]) if "close" in test.columns else None
            if np.isfinite(atr) and price and price > 0:
                risk_est = atr / price
        except Exception:
            risk_est = None
        if risk_est is None:
            risk_est = 0.02

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
            "notes": f"pocas muestras OOS: {len(realized)}",
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

    # Detect daily table once
    daily_table = detect_daily_bars_table(engine)

    # Horizon: tu config lo usa como “días” (ej: 10)
    horizon_days = int(cfg["ml"]["horizon_bars"])
    min_rows = int(cfg["ml"]["min_train_rows"])
    model_type = cfg["ml"]["model_type"]

    fee_bps = float(cfg["backtest"]["fee_bps"])
    slippage_bps = float(cfg["backtest"]["slippage_bps"])

    inserted = 0

    for inst in cfg["universe"]:
        name = inst["name"]

        ex, sym = resolve_series(engine, daily_table, inst)
        if not ex or not sym:
            print(f"[TRAIN SKIP] {name}: sin datos daily")
            continue

        bars = read_bars_daily(engine, daily_table, ex, sym)
        if bars is None or bars.empty:
            print(f"[TRAIN SKIP] {name}: sin barras daily")
            continue

        feat = compute_features(bars, horizon_bars=horizon_days)
        if feat is None or feat.empty:
            print(f"[TRAIN SKIP] {name}: features vacío")
            continue

        clf, reg, train_rows = train_models(feat, model_type=model_type, min_rows=min_rows)
        model_id = f"{model_type}_{horizon_days}d_1d"
        meta = {
            "name": name,
            "train_rows": int(train_rows),
            "horizon_days": int(horizon_days),
            "tf": "1d",
            "daily_table": daily_table,
        }

        # ✅ CLAVE: guarda en model_store_1d (tf="1d")
        save_models(engine, ex, sym, model_id, clf, reg, meta, tf="1d")

        trained_at = datetime.now(timezone.utc)

        m = compute_walk_forward_metrics(
            feat,
            min_train=min_rows,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_oos_points=252,
        )

        insert_metrics_1d(engine, ex, sym, model_id, trained_at, f"{horizon_days}d", m)
        inserted += 1

        print(
            f"[METRICS OK] {name} {ex}:{sym} "
            f"n_test={m.get('n_test')} sharpe={m.get('sharpe')} "
            f"max_dd={m.get('max_dd')} hit_rate={m.get('hit_rate')} notes={m.get('notes')}"
        )

    with engine.begin() as conn:
        c = conn.execute(text("select count(*) from public.model_metrics_1d")).scalar()

    print(f"[TRAIN DONE] metrics_inserted={inserted} total_metrics_rows={c}")


if __name__ == "__main__":
    main()
