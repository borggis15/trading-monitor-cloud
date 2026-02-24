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


FEATURES_DEFAULT = ["rsi", "ema_fast", "ema_slow", "atr", "zscore"]


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


def resolve_series_1d(engine, inst: dict) -> tuple[str | None, str | None]:
    """
    v3: db-first (elige la serie con más barras ya en DB)
    y luego fallback a candidates (stooq/yahoo) si no hay nada.
    """
    name = inst.get("name", "—")

    # 1) DB-first: mira qué series existen en DB para este asset_name, escoge la de mayor conteo
    try:
        q = """
        select exchange, symbol, count(*) as n
        from public.bars_1d
        where asset_name = :asset_name
        group by exchange, symbol
        order by n desc
        limit 1
        """
        row = pd.read_sql(text(q), engine, params={"asset_name": name})
        if not row.empty:
            ex = str(row.loc[0, "exchange"])
            sym = str(row.loc[0, "symbol"])
            bars = read_bars_1d(engine, ex, sym)
            if not bars.empty:
                print(f"[SERIES] {name} using {ex}:{sym} bars={len(bars)} (db-best)")
                return ex, sym
    except Exception:
        pass

    # 2) Fallback: candidates en config
    candidates: list[tuple[str, str]] = []
    for s in inst.get("stooq_candidates", []) or []:
        candidates.append(("STOOQ", s))
    for y in inst.get("yahoo_candidates", []) or []:
        candidates.append(("YAHOO", y))

    for ex, sym in candidates:
        if not sym:
            continue
        bars = read_bars_1d(engine, ex, sym)
        if not bars.empty:
            print(f"[SERIES] {name} using {ex}:{sym} bars={len(bars)}")
            return ex, sym

    return None, None


def compute_walk_forward_metrics(
    feat: pd.DataFrame,
    min_train: int,
    fee_bps: float,
    slippage_bps: float,
    max_oos_points: int = 260,
):
    """
    Walk-forward sencillo (reentrena y evalúa forward). Métricas básicas.
    """
    df = feat.copy()
    need = FEATURES_DEFAULT + ["ret_fwd"]
    if df is None or df.empty or any(c not in df.columns for c in need):
        return {
            "n_test": 0,
            "hit_rate": None,
            "avg_ret": None,
            "sharpe": None,
            "max_dd": None,
            "profit_factor": None,
            "notes": "missing_cols_or_empty_feat",
        }

    df = df.dropna(subset=need).copy()
    if len(df) < (min_train + 40):
        return {
            "n_test": 0,
            "hit_rate": None,
            "avg_ret": None,
            "sharpe": None,
            "max_dd": None,
            "profit_factor": None,
            "notes": f"too_few_rows_after_dropna={len(df)}",
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

        # train_models v2 puede devolver 3 o 4+ items → lo manejamos robusto
        tm = train_models(train, model_type="hgb", min_rows=min_train)
        if isinstance(tm, tuple) and len(tm) >= 3:
            clf, reg, _train_rows = tm[0], tm[1], tm[2]
            meta_extra = tm[3] if len(tm) >= 4 else {}
        else:
            # fallback ultra defensivo
            clf, reg, meta_extra = None, None, {}

        if clf is None or reg is None:
            continue

        proba, ret_exp = predict(clf, reg, test, features=(meta_extra or {}).get("features"))
        atr = float(test["atr"].iloc[0]) if np.isfinite(test["atr"].iloc[0]) else None
        risk_est = 0.02 if atr else None
        ev = ev_bps(ret_exp, risk_est, fee_bps, slippage_bps)

        # umbrales: si train_models devuelve thresholds, úsalos; si no, defaults
        thr = (meta_extra or {}).get("thresholds") or {}
        buy_p = float(thr.get("buy_proba", 0.62))
        sell_p = float(thr.get("sell_proba", 0.40))
        buy_ev = float(thr.get("buy_ev_bps", 25.0))
        sell_ev = float(thr.get("sell_ev_bps", -25.0))

        signal = "HOLD"
        if proba is not None and ev is not None:
            if float(proba) >= buy_p and float(ev) >= buy_ev:
                signal = "BUY"
            elif float(proba) <= sell_p and float(ev) <= sell_ev:
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

    if len(realized) < 40:
        return {
            "n_test": int(len(realized)),
            "hit_rate": None,
            "avg_ret": None,
            "sharpe": None,
            "max_dd": None,
            "profit_factor": None,
            "notes": f"too_few_oos={len(realized)}",
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

    print("[BOOT] TRAIN_1D v3 (db-first series selection + candidates fallback)")

    horizon = int(cfg["ml"].get("horizon_bars", 10))       # tu horizonte “10d”
    min_rows = int(cfg["ml"].get("min_train_rows", 200))
    model_type = str(cfg["ml"].get("model_type", "hgb"))

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
        if bars.empty:
            print(f"[TRAIN SKIP] {name}: empty bars")
            continue

        feat = compute_features(bars, horizon_bars=horizon)
        if feat is None or feat.empty:
            print(f"[TRAIN SKIP] {name}: empty features")
            continue

        # ✅ train_models v2: soporta 3 o 4+ outputs
        tm = train_models(feat, model_type=model_type, min_rows=min_rows)

        if not isinstance(tm, tuple) or len(tm) < 3:
            print(f"[TRAIN SKIP] {name}: train_models returned unexpected format")
            continue

        clf, reg, train_rows = tm[0], tm[1], tm[2]
        meta_extra = tm[3] if len(tm) >= 4 and isinstance(tm[3], dict) else {}

        model_id = f"{model_type}_{horizon}d_1d"
        trained_at = datetime.now(timezone.utc)

        meta_base = {"name": name, "train_rows": int(train_rows), "horizon_days": int(horizon)}
        meta = {**meta_base, **(meta_extra or {})}
        meta["model_id"] = model_id

        # ✅ guarda en model_store_1d (tf="1d")
        save_models(engine, ex, sym, model_id, clf, reg, meta, tf="1d")

        # ✅ métricas OOS (walk-forward)
        m = compute_walk_forward_metrics(
            feat,
            min_train=min_rows,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            max_oos_points=260,
        )
        insert_metrics_1d(engine, ex, sym, model_id, trained_at, f"{horizon}d", m)

        inserted += 1
        print(
            f"[METRICS OK] {name} {ex}:{sym} n_test={m.get('n_test')} "
            f"sharpe={m.get('sharpe')} max_dd={m.get('max_dd')} hit_rate={m.get('hit_rate')} notes={m.get('notes')}"
        )

    with engine.begin() as conn:
        c = conn.execute(text("select count(*) from public.model_metrics_1d")).scalar()

    print(f"[TRAIN DONE] metrics_inserted={inserted} total_metrics_rows={c}")


if __name__ == "__main__":
    main()
