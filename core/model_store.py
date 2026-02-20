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
    - psycopg2 NO soporta ':meta::jsonb' como bind.
    - usamos %(meta)s::jsonb y pasamos meta como string JSON.
    """
    import pickle

    table = _table_for_tf(tf)

    clf_blob = pickle.dumps(clf)
    reg_blob = pickle.dumps(reg)

    meta_json = json.dumps(meta or {}, ensure_ascii=False)

    with engine.begin() as conn:
        conn.execute(
            text(
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
                    %(exchange)s,
                    %(symbol)s,
                    %(model_id)s,
                    now(),
                    %(clf_pickle)s,
                    %(reg_pickle)s,
                    %(meta)s::jsonb
                )
                """
            ),
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
    # meta llega ya como dict (jsonb) o como string dependiendo del driver/config
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {"raw_meta": meta}

    # aseguramos que meta sea dict
    if meta is None or not isinstance(meta, dict):
        meta = {}

    # si el model_id no está dentro, lo añadimos
    if "model_id" not in meta and row.get("model_id") is not None:
        meta["model_id"] = row.get("model_id")

    return clf, reg, meta
