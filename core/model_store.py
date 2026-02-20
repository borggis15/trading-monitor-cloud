from __future__ import annotations

import json
import pickle
from sqlalchemy import text


def save_models(engine, exchange: str, symbol: str, model_id: str, clf, reg, meta: dict, tf: str = "15m"):
    """
    Guarda en:
      public.model_store      (tf=15m)
      public.model_store_1d   (tf=1d)
    """
    table = "public.model_store_1d" if tf == "1d" else "public.model_store"

    clf_blob = pickle.dumps(clf)
    reg_blob = pickle.dumps(reg)
    meta_json = json.dumps(meta or {})

    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                insert into {table}(exchange, symbol, model_id, trained_at, clf_pickle, reg_pickle, meta)
                values (:exchange, :symbol, :model_id, now(), :clf_pickle, :reg_pickle, cast(:meta as jsonb))
                on conflict (exchange, symbol, model_id) do update set
                  trained_at = excluded.trained_at,
                  clf_pickle = excluded.clf_pickle,
                  reg_pickle = excluded.reg_pickle,
                  meta = excluded.meta
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


def load_latest_models(engine, exchange: str, symbol: str, tf: str = "15m"):
    """
    Devuelve (clf, reg, meta) del modelo más reciente por trained_at.
    """
    table = "public.model_store_1d" if tf == "1d" else "public.model_store"

    row = engine.execute(
        text(
            f"""
            select model_id, clf_pickle, reg_pickle, meta, trained_at
            from {table}
            where exchange=:exchange and symbol=:symbol
            order by trained_at desc
            limit 1
            """
        ),
        {"exchange": exchange, "symbol": symbol},
    ).fetchone()

    if not row:
        return None, None, None

    model_id, clf_blob, reg_blob, meta, _trained_at = row
    clf = pickle.loads(clf_blob) if clf_blob is not None else None
    reg = pickle.loads(reg_blob) if reg_blob is not None else None
    meta = dict(meta) if meta is not None else {}

    meta["model_id"] = model_id
    return clf, reg, meta
