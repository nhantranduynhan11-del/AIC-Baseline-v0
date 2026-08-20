"""Chay UI voi DU LIEU GIA - de sua giao dien tren may khong co GPU.

    python scripts/ui_demo.py            # http://127.0.0.1:8010

Dung 60 keyframe tong hop (anh mau co so thu tu), FAISS index that, DB OCR that,
va mot encoder GIA tra ve vector ngau nhien. Nho vay toan bo duong di cua UI -
search, grid, panel chi tiet, keyframe lan can, OCR, gio nop bai, export CSV -
deu chay that, chi co model la gia.

KHONG dung cho thi dau. Chay that:
    uvicorn aic.api.app:app --app-dir src --port 8000
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

import numpy as np

N_VIDEOS = 6
KF_PER_VIDEO = 10
DIM_CLIP, DIM_SIGLIP = 8, 16

PALETTE = [(198, 76, 72), (72, 132, 198), (86, 166, 104), (196, 148, 62),
           (146, 96, 186), (72, 168, 176)]


def build_demo_state(root: Path):
    from PIL import Image, ImageDraw

    from aic.api.state import AppState
    from aic.manifest import KeyframeEntry
    from aic.retrieval.search import IndexBundle
    from aic.store import faiss_store
    from aic.store import sqlite_store as store

    entries, total = [], N_VIDEOS * KF_PER_VIDEO
    for v in range(N_VIDEOS):
        video_id = f"L01_V{v:03d}"
        for j in range(KF_PER_VIDEO):
            idx = v * KF_PER_VIDEO + j
            frame_idx = j * 40
            rel = f"{video_id}/{frame_idx}.jpg"
            entries.append(KeyframeEntry(idx, video_id, frame_idx, frame_idx / 25.0, rel))

            path = root / "keyframes" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            base = PALETTE[v % len(PALETTE)]
            shade = 0.55 + 0.045 * j
            img = Image.new("RGB", (480, 270), tuple(int(c * shade) % 256 for c in base))
            draw = ImageDraw.Draw(img)
            draw.text((16, 14), f"{video_id}", fill=(255, 255, 255))
            draw.text((16, 34), f"frame {frame_idx}", fill=(235, 235, 235))
            draw.text((16, 54), f"idx {idx}", fill=(210, 210, 210))
            img.save(path, "JPEG", quality=85)

    rng = np.random.RandomState(0)

    def unit(dim: int) -> np.ndarray:
        v = rng.rand(total, dim).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    bundle = IndexBundle(
        {
            "clip": faiss_store.build_flat_ip(unit(DIM_CLIP)),
            "siglip2": faiss_store.build_flat_ip(unit(DIM_SIGLIP)),
        },
        entries,
    )

    conn = store.open_db(root / "metadata.db", check_same_thread=False)
    samples = ["Đường Điện Biên Phủ", "TIN NÓNG", "Quận Bình Thạnh",
               "Thời sự 19 giờ", "Dự báo thời tiết"]
    rows = []
    for e in entries:
        if e.idx % 3 == 0:
            rows.append((e.video_id, e.frame_idx, samples[e.idx % len(samples)], 0.55 + (e.idx % 4) * 0.1))
    store.insert_ocr(conn, rows)
    conn.commit()

    class FakeEncoder:
        """Vector ngau nhien - ket qua vo nghia, nhung duong di cua UI thi that."""

        def encode(self, text=None, image_rgb=None):
            seed = abs(hash(text)) % 2**31 if text is not None else int(image_rgb.mean() * 1000)
            r = np.random.RandomState(seed)

            def q(dim):
                v = r.rand(dim).astype(np.float32)
                return v / np.linalg.norm(v)

            return {"clip": q(DIM_CLIP), "siglip2": q(DIM_SIGLIP)}

    class DemoCfg:
        paths = type("P", (), {
            "keyframes": str(root / "keyframes"),
            "thumbs": str(root / "thumbs"),
            "metadata_db": str(root / "metadata.db"),
        })()
        retrieval = type("R", (), {"top_k_per_model": 60, "rrf_k": 60, "final_top_n": 100})()
        ocr = type("O", (), {"query_min_confidence": 0.3})()

    return AppState(cfg=DemoCfg(), bundle=bundle, encoder=FakeEncoder(), conn=conn)


def main() -> int:
    p = argparse.ArgumentParser(description="UI demo voi du lieu gia")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    import uvicorn

    from aic.api import app as api

    root = Path(tempfile.mkdtemp(prefix="aic_ui_demo_"))
    print(f"Dang dung du lieu gia trong {root} ...")
    state = build_demo_state(root)
    # Gan thang vao module: get_state() VA /health deu doc bien nay, nen UI thay
    # server "san sang" giong het khi chay that.
    api._state = state
    print(f"{state.bundle.ntotal} keyframe gia. Mo http://{args.host}:{args.port}")
    print("DU LIEU GIA - ket qua search vo nghia, chi de sua giao dien.")

    uvicorn.run(api.app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
