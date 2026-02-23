from __future__ import annotations

import json
import pickle
from typing import Any, Dict, Tuple, Optional

import pandas as pd
from sqlalchemy import text


def _table_for_tf(tf: str | None) -> str:
    """
    tf=None or '15m' -> model_store
    tf='1d' -> model_store_1d
    """
    tf = (tf or "").strip().lower()
    return "public.model_store_1d" if tf == "1d" else "public.model_store"


def save_models(
    engine,
    exchange: str,
    symbol: str,
    model_id: str,
    clf,
    reg,
    meta: Dict[str, Any] | None = None,
    tf: str | None = None,
):
    """
    Guarda modelos en Postgres como pickles + meta jsonb.
    Compatible con SQLAlchemy 2.x (sin engine.execute).
    """
    table = _table_for_tf(tf)

    meta = meta or {}
    meta_json = json.dumps(meta, ensure_ascii=False)

    clf_blob = pickle.dumps(clf) if clf is not None else None
    reg_blob = pickle.dumps(reg) if reg is not None else None

    q = f"""
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
        :meta::jsonb
    )
    on conflict (exchange, symbol, model_id) do update set
        trained_at = excluded.trained_at,
        clf_pickle = excluded.clf_pickle,
        reg_pickle = excluded.reg_pickle,
        meta = excluded.meta
    """

    params = {
        "exchange": exchange,
        "symbol": symbol,
        "model_id": model_id,
        "clf_pickle": clf_blob,
        "reg_pickle": reg_blob,
        "meta": meta_json,
    }

    with engine.begin() as conn:
        conn.execute(text(q), params)


def load_latest_models(
    engine,
    exchange: str,
    symbol: str,
    tf: str | None = None,
) -> Tuple[Optional[Any], Optional[Any], Optional[Dict[str, Any]]]:
    """
    Carga el último par (clf, reg, meta) por trained_at desc.
    Devuelve (clf, reg, meta_dict) o (None, None, None) si no hay.
    """
    table = _table_for_tf(tf)

    q = f"""
    select model_id, trained_at, clf_pickle, reg_pickle, meta
    from {table}
    where exchange=:exchange and symbol=:symbol
    order by trained_at desc
    limit 1
    """

    with engine.connect() as conn:
        res = conn.execute(text(q), {"exchange": exchange, "symbol": symbol})
        row = res.mappings().first()  # ✅ para acceso por claves

    if not row:
        return None, None, None

    clf = pickle.loads(row["clf_pickle"]) if row.get("clf_pickle") is not None else None
    reg = pickle.loads(row["reg_pickle"]) if row.get("reg_pickle") is not None else None

    meta = row.get("meta")
    # meta puede venir como dict (psycopg2 json) o como str
    if meta is None:
        meta_dict = {"model_id": row.get("model_id")}
    elif isinstance(meta, dict):
        meta_dict = meta
    else:
        try:
            meta_dict = json.loads(meta)
        except Exception:
            meta_dict = {"raw_meta": str(meta)}

    # Enriquecemos con model_id por si no viene
    meta_dict.setdefault("model_id", row.get("model_id"))

    return clf, reg, meta_dict
