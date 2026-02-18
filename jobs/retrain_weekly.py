from __future__ import annotations
from sqlalchemy import text
from core.config import load_config
from core.db import get_engine

def main():
    cfg = load_config()
    engine = get_engine()
    model_id = f"{cfg['ml']['model_type']}_{cfg['ml']['horizon_bars']}"
    with engine.begin() as conn:
        conn.execute(text("""
          insert into model_registry(model_id, exchange, horizon_bars, model_type, train_rows, notes)
          values (:model_id,:exchange,:horizon,:type,null,:notes)
          on conflict (model_id) do update set created_at=now(), exchange=excluded.exchange, horizon_bars=excluded.horizon_bars, model_type=excluded.model_type, notes=excluded.notes
        """), {
            "model_id": model_id,
            "exchange": cfg["data"]["exchange"],
            "horizon": int(cfg["ml"]["horizon_bars"]),
            "type": cfg["ml"]["model_type"],
            "notes": "Reentreno programado (auditoría).",
        })
    print("Model registry updated:", model_id)

if __name__ == "__main__":
    main()
