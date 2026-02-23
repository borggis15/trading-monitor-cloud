from __future__ import annotations

import pickle
from sqlalchemy import text


# ----------------------------
# SAVE MODELS
# ----------------------------
def save_models(engine, exchange, symbol, model_id, clf, reg, meta, tf="15m"):
    table = "model_store_1d" if tf == "1d" else "model_store"

    clf_bytes = pickle.dumps(clf)
    reg_bytes = pickle.dumps(reg)

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                insert into public.{table}(
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
                    :meta
                )
            """),
            {
                "exchange": exchange,
                "symbol": symbol,
                "model_id": model_id,
                "clf_pickle": clf_bytes,
                "reg_pickle": reg_bytes,
                "meta": meta,
            },
        )


# ----------------------------
# LOAD LATEST MODELS
# ----------------------------
def load_latest_models(engine, exchange, symbol, tf="15m"):
    table = "model_store_1d" if tf == "1d" else "model_store"

    with engine.begin() as conn:
        result = conn.execute(
            text(f"""
                select model_id, clf_pickle, reg_pickle, meta
                from public.{table}
                where exchange=:exchange and symbol=:symbol
                order by trained_at desc
                limit 1
            """),
            {
                "exchange": exchange,
                "symbol": symbol,
            },
        ).fetchone()

    if not result:
        return None, None, None

    clf = pickle.loads(result["clf_pickle"])
    reg = pickle.loads(result["reg_pickle"])
    meta = result["meta"] or {}

    # añadimos model_id dentro del meta (muy útil)
    meta["model_id"] = result["model_id"]

    return clf, reg, meta
