# core/model_store.py
from __future__ import annotations

import json
import pickle
import pandas as pd
from sqlalchemy import text


def _table(tf: str) -> str:
    tf = (tf or "15m").lower()
    return "public.model_store_1d" if tf in ("1d", "d", "day", "daily") else "public.model_store"


def save_models(engine, exchange: str, symbol: str, model_id: str, clf, reg, meta: dict, tf: str = "15m"):
    tbl = _table(tf)
    meta_json = json.dumps(meta or {}, ensure_ascii=False)

    clf_pickle = pickle.dumps(clf)
    reg_pickle = pickle.dumps(reg)

    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                insert into {tbl}(exchange, symbol, model_id, trained_at, clf_pickle, reg_pickle, meta)
                values (:exchange, :symbol, :model_id, now(), :clf_pickle, :reg_pickle, :meta::jsonb)
                """
            ),
            {
                "exchange": exchange,
                "symbol": symbol,
                "model_id": model_id,
                "clf_pickle": clf_pickle,
                "reg_pickle": reg_pickle,
                "meta": meta_json,
            },
        )


def load_latest_models(engine, exchange: str, symbol: str, tf: str = "15m"):
    tbl = _table(tf)

    df = pd.read_sql(
        text(
            f"""
            select model_id, trained_at, clf_pickle, reg_pickle, meta
            from {tbl}
            where exchange=:exchange and symbol=:symbol
            order by trained_at desc
            limit 1
            """
        ),
        engine,
        params={"exchange": exchange, "symbol": symbol},
    )

    if df.empty:
        return None, None, None

    row = df.iloc[0]
    clf = pickle.loads(row["clf_pickle"])
    reg = pickle.loads(row["reg_pickle"])
    meta = row["meta"] if isinstance(row["meta"], dict) else (json.loads(row["meta"]) if row["meta"] else {})
    meta["model_id"] = row["model_id"]
    meta["trained_at"] = row["trained_at"]

    return clf, reg, meta
