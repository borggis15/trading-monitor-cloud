from __future__ import annotations
from pathlib import Path
import yaml

def load_config(path: str | Path = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
