from __future__ import annotations

import pickle
from sqlalchemy import text


def save_models(engine, exchange: str, symbol: str, model_id: str, clf, reg, meta: dict):
    clf_b = pickle.dumps(clf) if clf is not None else None
    reg_b = pickle.dumps(reg) if reg is not None else None

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                insert into public.model_store(exchange, symbol, model_id, trained_at, clf_pickle, reg_pickle, meta)
                values (:exchange, :symbol, :model_id, now(), :clf_pickle, :reg_pickle, :meta)
                """
            ),
            {
                "exchange": exchange,
                "symbol": symbol,
                "model_id": model_id,
                "clf_pickle": clf_b,
                "reg_pickle": reg_b,
                "meta": meta,
            },
        )


def load_latest_models(engine, exchange: str, symbol: str):
    with engine.begin() as conn:
        row = conn.execute(
            text(
                """
                select model_id, trained_at, clf_pickle, reg_pickle, meta
                from public.model_store
                where exchange=:exchange and symbol=:symbol
                order by trained_at desc
                limit 1
                """
            ),
            {"exchange": exchange, "symbol": symbol},
        ).fetchone()

    if not row:
        return None, None, None

    clf = pickle.loads(row.clf_pickle) if row.clf_pickle is not None else None
    reg = pickle.loads(row.reg_pickle) if row.reg_pickle is not None else None
    return clf, reg, {"model_id": row.model_id, "trained_at": str(row.trained_at), "meta": row.meta}
