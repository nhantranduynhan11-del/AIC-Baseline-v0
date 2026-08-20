"""Load config YAML + resolve duong dan.

Dung: cfg = load_config()  hoac  load_config("configs/exp1.yaml")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

_PATH_KEYS = (
    "data_root", "videos", "shots", "shots_raw", "keyframes", "thumbs", "index_dir",
    "manifest", "clip_embeddings", "faiss_clip", "faiss_siglip",
    "index_meta", "metadata_db",
)


class Config(dict):
    """dict co truy cap kieu thuoc tinh, long nhau."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # Duong dan tuong doi -> tuyet doi theo PROJECT_ROOT
    paths = raw.get("paths", {})
    for key in _PATH_KEYS:
        if key in paths and paths[key] is not None:
            p = Path(paths[key])
            paths[key] = str(p if p.is_absolute() else PROJECT_ROOT / p)

    return Config(raw)
