from __future__ import annotations

import pickle
from sqlalchemy import text


def _table_for_tf(tf: str) -> str:
    """
    tf: "15m" (default) or "1d"
    """
    return "model_store_1d" if tf == "1d" else "model_store"


# ----------------------------
# SAVE MODELS
# ----------------------------
def save_models(engine, exchange: str, symbol: str, model_id: str, clf, reg, meta: dict, tf: str = "15m"):
    table = _table_for_tf(tf)

    clf_bytes = pickle.dumps(clf)
    reg_bytes = pickle.dumps(reg)

    # Nota: meta lo guardamos como JSON (dict), Postgres lo acepta en jsonb
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
                    :meta::jsonb
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
def load_latest_models(engine, exchange: str, symbol: str, tf: str = "15m"):
    table = _table_for_tf(tf)

    with engine.begin() as conn:
        row = (
            conn.execute(
                text(f"""
                    select model_id, clf_pickle, reg_pickle, meta
                    from public.{table}
                    where exchange=:exchange and symbol=:symbol
                    order by trained_at desc
                    limit 1
                """),
                {"exchange": exchange, "symbol": symbol},
            )
            .mappings()          # ✅ convierte a dict-like
            .fetchone()
        )

    if not row:
        return None, None, None

    clf = pickle.loads(row["clf_pickle"])
    reg = pickle.loads(row["reg_pickle"])
    meta = row["meta"] or {}

    # por comodidad para score_*
    meta["model_id"] = row["model_id"]

    return clf, reg, meta
