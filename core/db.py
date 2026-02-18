from __future__ import annotations
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

def get_engine() -> Engine:
    return create_engine(os.environ["DATABASE_URL"], future=True, pool_pre_ping=True)
