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
# Data access
# ----------------------------
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
    df = df.dropna(subset=["ts"]).set_index("ts").sort_index()
    return df


def pick_best_daily_series_from_db(engine, asset_name: str) -> tuple[str | None, str | None, int]:
    """
    Si hay datos en bars_1d para este asset_name, elegimos la "mejor" serie.
    Preferencias:
      1) STOOQ
      2) YAHOO
      3) otras
    Y dentro de cada una: más filas + más reciente
    """
    q = """
    with s as (
      select
        exchange,
        symbol,
        count(*) as n,
        max(ts) as newest
      from public.bars_1d
      where asset_name = :asset_name
      group by exchange, symbol
    )
    select exchange, symbol, n
    from s
    order by
      case when exchange='STOOQ' then 3
           when exchange='YAHOO' then 2
           else 1 end desc,
      n desc,
      newest desc
    limit 1
    """
    df = pd.read_sql(text(q), engine, params={"asset_name": asset_name})
    if df.empty:
        return None, None, 0
    return df.loc[0, "exchange"], df.loc[0, "symbol"], int(df.loc[0, "n"])


def resolve_series_1d(engine, inst: dict) -> tuple[str | None, str | None]:
    """
    Resolver serie daily de forma robusta:

    A) Primero intenta desde DB por asset_name (lo más fiable).
       - Esto arregla tu caso Lundin: aunque no esté en STOOQ, si existe YAHOO:LUN.TO en bars_1d, lo usa.

    B) Si no hay nada en DB, fallback a candidates del config:
       - stooq_candidates, yahoo_candidates
    """
    name = inst["name"]

    # A) DB-first
    ex, sym, n = pick_best_daily_series_from_db(engine, name)
    if ex and sym and n > 0:
        bars = read_bars_1d(engine, ex, sym)
        if bars is not None and not bars.empty:
            print(f"[SERIES] {name} using {ex}:{sym} bars={len(bars)} (db-best)")
            return ex, sym

    # B) Candidates fallback
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
            print(f"[SERIES] {name} using {ex}:{sym} bars={len(bars)} (candidates)")
            return ex, sym

    return None, None


# ----------------------------
# Walk-forward metrics
# ----------------------------
def compute_walk_forward_metrics(
    feat: pd.DataFrame,
    min_train: int,
    fee_bps: float,
    slippage_bps: float,
    max_oos_points: int = 260,  # daily: más puntos OOS que 15m
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
            "notes": "Faltan columnas o feat vacío",
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
            "notes": f"Insuficientes filas tras dropna: {len(df)}",
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

        # risk_est para EV (simple, estable)
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
            "notes": f"Pocas muestras OOS: {len(realized)}",
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


def insert_metrics(engine, exchange: str, symbol: str, model_id: str, trained_at: datetime, horizon: str, m: dict):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.model_metrics(
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


# ----------------------------
# Main
# ----------------------------
def main():
    cfg = load_config()
    engine = get_engine()

    # daily horizon y training mins: soporta tus claves nuevas o fallback a las viejas
    horizon = int(cfg["ml"].get("horizon_bars_1d", cfg["ml"]["horizon_bars"]))
    min_rows = int(cfg["ml"].get("min_train_rows_1d", cfg["ml"]["min_train_rows"]))
    model_type = cfg["ml"]["model_type"]

    fee_bps = float(cfg["backtest"]["fee_bps"])
    slippage_bps = float(cfg["backtest"]["slippage_bps"])

    inserted = 0
    print("[BOOT] TRAIN_1D v3 (db-first series selection + candidates fallback)")

    for inst in cfg["universe"]:
        name = inst["name"]

        ex, sym = resolve_series_1d(engine, inst)
        if not ex or not sym:
            print(f"[TRAIN SKIP] {name}: sin datos daily")
            continue

        bars = read_bars_1d(engine, ex, sym)
        if bars is None or bars.empty:
            print(f"[TRAIN SKIP] {name}: barras vacías ({ex}:{sym})")
            continue

        print(f"[SERIES] {name} using {ex}:{sym} bars={len(bars)}")

        feat = compute_features(bars, horizon_bars=horizon)
        if feat is None or feat.empty:
            print(f"[TRAIN SKIP] {name}: features vacío")
            continue

        clf, reg, train_rows = train_models(feat, model_type=model_type, min_rows=min_rows)
        model_id = f"{model_type}_{horizon}d_1d"

        meta = {
            "name": name,
            "train_rows": int(train_rows),
            "horizon_days": int(horizon),
            "tf": "1d",
            "series": f"{ex}:{sym}",
        }

        # IMPORTANT: tf="1d" para que guarde en model_store_1d
        save_models(engine, ex, sym, model_id, clf, reg, meta, tf="1d")

        trained_at = datetime.now(timezone.utc)

        m = compute_walk_forward_metrics(
            feat,
            min_train=min_rows,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_oos_points=260,
        )

        insert_metrics(engine, ex, sym, model_id, trained_at, f"{horizon}d_1d", m)
        inserted += 1

        print(
            f"[METRICS OK] {name} {ex}:{sym} "
            f"n_test={m.get('n_test')} sharpe={m.get('sharpe')} "
            f"max_dd={m.get('max_dd')} hit_rate={m.get('hit_rate')} notes={m.get('notes')}"
        )

    with engine.begin() as conn:
        c = conn.execute(text("select count(*) from public.model_metrics where model_id like '%_1d'")).scalar()

    print(f"[TRAIN DONE] metrics_inserted={inserted} total_metrics_rows_1d={c}")


if __name__ == "__main__":
    main()
