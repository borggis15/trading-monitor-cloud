# core/model_store.py
from __future__ import annotations

import json
import pandas as pd
from sqlalchemy import text
from typing import Any, Tuple


def _table_for_tf(tf: str | None) -> str:
    """
    tf:
      - None / "15m" -> model_store
      - "1d"        -> model_store_1d
    """
    if (tf or "").lower() in ("1d", "daily"):
        return "public.model_store_1d"
    return "public.model_store"


def save_models(
    engine,
    exchange: str,
    symbol: str,
    model_id: str,
    clf: Any,
    reg: Any,
    meta: dict,
    tf: str | None = None,
):
    """
    Guarda modelos serializados en la tabla correspondiente (15m o 1d).

    IMPORTANTE:
    - NO usar %(...)s dentro del SQL.
    - Usar :param (SQLAlchemy) y CAST(:meta AS jsonb).
    """
    import pickle

    table = _table_for_tf(tf)

    clf_blob = pickle.dumps(clf)
    reg_blob = pickle.dumps(reg)

    # Guardamos meta como string JSON y lo casteamos a jsonb en SQL
    meta_json = json.dumps(meta or {}, ensure_ascii=False)

    sql = text(
        f"""
        insert into {table}(
            exchange,
            symbol,
            model_id,
            trained_at,
            clf_pickle,
            reg_pickle,
            meta
        )
        values (
            :exchange,
            :symbol,
            :model_id,
            now(),
            :clf_pickle,
            :reg_pickle,
            CAST(:meta AS jsonb)
        )
        """
    )

    with engine.begin() as conn:
        conn.execute(
            sql,
            {
                "exchange": exchange,
                "symbol": symbol,
                "model_id": model_id,
                "clf_pickle": clf_blob,
                "reg_pickle": reg_blob,
                "meta": meta_json,
            },
        )


def load_latest_models(
    engine,
    exchange: str,
    symbol: str,
    tf: str | None = None,
) -> Tuple[Any, Any, dict | None]:
    """
    Carga el último modelo entrenado para exchange/symbol desde la tabla correspondiente (15m o 1d).
    Devuelve: (clf, reg, meta)
    """
    import pickle

    table = _table_for_tf(tf)

    df = pd.read_sql(
        text(
            f"""
            select clf_pickle, reg_pickle, meta, model_id, trained_at
            from {table}
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

    meta = row.get("meta")

    # meta puede venir como dict (jsonb) o como string
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {"raw_meta": meta}

    if meta is None or not isinstance(meta, dict):
        meta = {}

    if "model_id" not in meta and row.get("model_id") is not None:
        meta["model_id"] = row.get("model_id")

    return clf, reg, meta
