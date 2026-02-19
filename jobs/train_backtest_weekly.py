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


def walk_forward_metrics(feat: pd.DataFrame, min_train: int, step: int = 20):
    """
    Walk-forward simple:
      - entrenamos hasta i
      - predecimos siguiente ventana [i, i+step)
    """
    df = feat.dropna().copy()
    if "ret_fwd" not in df.columns:
        return None

    # etiquetas: subida/no subida
    df["y_cls"] = (df["ret_fwd"] > 0).astype(int)
    df["y_reg"] = df["ret_fwd"]

    X_cols = [c for c in df.columns if c in ["rsi", "ema_fast", "ema_slow", "atr", "zscore"]]
    if len(X_cols) == 0:
        return None

    preds = []
    rets = []

    for i in range(min_train, len(df) - step, step):
        train = df.iloc[:i]
        test = df.iloc[i:i+step]
        if len(train) < min_train or len(test) == 0:
            continue

        # entreno rápido con tu train_models
        clf, reg, _ = train_models(train, model_type="hgb", min_rows=min_train)

        proba, ret_exp = predict(clf, reg, test)
        # predict() devuelve escalar para último punto; aquí tomamos último test
        # para walk-forward sencillo: usamos señal del último punto de test
        if ret_exp is None:
            continue

        r = float(test["ret_fwd"].iloc[-1])
        rets.append(r)
        preds.append(float(ret_exp))

    if len(rets) < 10:
        return None

    rets = np.array(rets)
    avg = float(np.mean(rets))
    std = float(np.std(rets) + 1e-9)
    sharpe = float((avg / std) * np.sqrt(252))

    # equity curve (compuesto)
    eq = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(eq)
    dd = (eq / peak) - 1.0
    max_dd = float(np.min(dd))

    hit = float(np.mean(rets > 0))
    pf = float(np.sum(rets[rets > 0]) / (abs(np.sum(rets[rets < 0])) + 1e-9))

    return {
        "n_test": int(len(rets)),
        "hit_rate": hit,
        "avg_ret": avg,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "profit_factor": pf,
    }


def upsert_metrics(engine, exchange: str, symbol: str, model_id: str, trained_at: str, horizon: str, m: dict):
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.model_metrics(exchange,symbol,model_id,trained_at,horizon,n_test,hit_rate,avg_ret,sharpe,max_dd,profit_factor,notes)
                values (:exchange,:symbol,:model_id,:trained_at,:horizon,:n_test,:hit_rate,:avg_ret,:sharpe,:max_dd,:profit_factor,:notes)
                on conflict (exchange,symbol,model_id,trained_at) do nothing
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
                "notes": "",
            },
        )


def main():
    cfg = load_config()
    engine = get_engine()

    for inst in cfg["universe"]:
        name = inst["name"]
        ex, sym = resolve_series(engine, inst)
        if not ex or not sym:
            print(f"[TRAIN SKIP] {name}: sin datos")
            continue

        bars = read_bars(engine, ex, sym)
        feat = compute_features(bars, horizon_bars=int(cfg["ml"]["horizon_bars"]))
        if feat is None or feat.empty:
            print(f"[TRAIN SKIP] {name}: features vacío")
            continue

        # entreno final con todo el histórico disponible (pero validación walk-forward aparte)
        clf, reg, train_rows = train_models(feat, model_type=cfg["ml"]["model_type"], min_rows=int(cfg["ml"]["min_train_rows"]))
        model_id = f"{cfg['ml']['model_type']}_{cfg['ml']['horizon_bars']}d"
        meta = {"name": name, "train_rows": int(train_rows), "horizon_days": int(cfg["ml"]["horizon_bars"])}

        save_models(engine, ex, sym, model_id, clf, reg, meta)

        # walk-forward
        m = walk_forward_metrics(feat, min_train=int(cfg["ml"]["min_train_rows"]), step=20)
        trained_at = datetime.now(timezone.utc).isoformat()
        if m:
            upsert_metrics(engine, ex, sym, model_id, trained_at, f"{int(cfg['ml']['horizon_bars'])}d", m)
            print(f"[TRAIN OK] {name} {ex}:{sym} -> sharpe={m['sharpe']:.2f}, dd={m['max_dd']:.2f}")
        else:
            print(f"[TRAIN OK] {name} {ex}:{sym} -> métricas insuficientes")

    print("[TRAIN DONE]")


if __name__ == "__main__":
    main()
