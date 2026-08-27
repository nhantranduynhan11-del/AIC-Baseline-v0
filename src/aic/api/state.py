"""Trang thai dung chung cua API - nap MOT LAN luc startup.

Hai model chiem khoang 3GB VRAM va mat vai chuc giay de nap. Nap lai moi request
la khong the chap nhan duoc khi dang thi, nen chung song suot vong doi process.

Sync hay async: cac endpoint deu la `def` thuong (khong phai `async def`), nen
FastAPI chay chung trong threadpool va khong khoa event loop. Rieng buoc encode
duoc bao boi mot Lock: mot process chi co mot ban model, hai request vao cung
luc se dung chung buffer cua torch. Chi co mot nguoi ngoi thi nen hang doi nay
khong phai van de.

Neu chua co index (chua chay A.2/A.3) thi state khong nap duoc - API van chay
va tra 503 kem ly do, thay vi crash luc khoi dong.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from aic.retrieval.search import IndexBundle, index_files_ready


@dataclass
class AppState:
    cfg: Any
    bundle: IndexBundle
    encoder: Any
    conn: Any = None                       # sqlite3.Connection, None neu chua co DB OCR
    yolo_model: Any = None                 # ultralytics.YOLOWorld
    lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def load(cls, cfg: Any, device: str | None = None) -> "AppState":
        from aic.retrieval.encode_query import QueryEncoder
        from aic.store import sqlite_store as store
        
        try:
            from ultralytics import YOLO
            print("[API] Dang tai YOLO11s...")
            yolo_model = YOLO("yolo11s.pt")
        except ImportError:
            print("[API] Khong the nap YOLO11s, vui long 'pip install ultralytics'")
            yolo_model = None

        bundle = IndexBundle.from_config(cfg)
        encoder = QueryEncoder(cfg, device=device)

        # check_same_thread=False vi FastAPI chay endpoint trong threadpool.
        # Phia API chi DOC nen khong co tranh chap ghi.
        try:
            conn = store.open_db(cfg.paths.metadata_db, check_same_thread=False)
        except Exception as exc:
            print(f"[API] Khong mo duoc DB OCR ({exc}) - tang B se khong dung duoc")
            conn = None

        return cls(cfg=cfg, bundle=bundle, encoder=encoder, conn=conn)

    @property
    def has_ocr(self) -> bool:
        return self.conn is not None


def can_load(cfg: Any) -> tuple[bool, str]:
    """Kiem tra da du file de nap state chua. Tra ve (duoc khong, ly do)."""
    if not index_files_ready(cfg):
        return False, (
            "Chua co manifest / FAISS index. Chay scripts/00_run_all.py truoc."
        )
    return True, ""
