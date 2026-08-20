"""A.3 Indexing - encode SigLIP2 + build 2 FAISS index tu manifest.

Hai buoc tach roi:

  1. `encode_siglip_video()` - encode anh keyframe bang SigLIP2, ghi ra
     <video_id>/siglip2.npy nam canh clip.npy cua A.2. Lam theo tung video de
     resume duoc: dut giua chung thi chay lai chi lam nhung video con thieu.

  2. `build_indexes()` - doc manifest, gop embedding, build 2 IndexFlatIP.
     - Vector CLIP: doc tu clip_embeddings.npy cua A.2, KHONG encode lai.
     - Vector SigLIP2: gop tu cac file siglip2.npy.

Bat bien ID (A.6): ham gop KHONG glob thu muc roi doan thu tu. No duyet DUNG
tung dong manifest, mang theo con tro trong tung video, va assert dung so hang.
Manifest la nguon su that; moi thu khac phai khop voi no chu khong nguoc lai.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aic.manifest import iter_manifest, write_index_meta
from aic.preprocess.keyframe import KEYFRAME_META, read_keyframe_meta
from aic.store import faiss_store

SIGLIP_EMB = "siglip2.npy"


def encode_siglip_video(
    encoder,
    keyframes_dir: str | Path,
    video_id: str,
    *,
    batch_size: int = 32,
) -> int:
    """Encode SigLIP2 cho toan bo keyframe cua mot video. Tra ve so vector.

    Doc DUNG thu tu keyframe trong keyframes.json cua A.2 - do cung la thu tu
    cac dong cua video nay trong manifest.
    """
    import cv2

    video_dir = Path(keyframes_dir) / video_id
    meta = read_keyframe_meta(video_dir / KEYFRAME_META)

    feats: list[np.ndarray] = []
    buffer: list[np.ndarray] = []

    def flush() -> None:
        if buffer:
            feats.append(encoder.encode_images(buffer))
            buffer.clear()

    for kf in meta["keyframes"]:
        img_path = Path(keyframes_dir) / kf["path"]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Khong doc duoc keyframe: {img_path}")
        buffer.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if len(buffer) >= batch_size:
            flush()
    flush()

    array = (
        np.concatenate(feats).astype(np.float32)
        if feats
        else np.zeros((0, encoder.dim), dtype=np.float32)
    )
    if len(array) != meta["n_keyframes"]:
        raise AssertionError(
            f"{video_id}: encode duoc {len(array)} vector nhung co {meta['n_keyframes']} keyframe"
        )
    np.save(video_dir / SIGLIP_EMB, array)
    return len(array)


def video_ids_in_manifest(manifest_path: str | Path) -> list[str]:
    """Danh sach video_id theo THU TU XUAT HIEN trong manifest, khong trung."""
    seen: list[str] = []
    last: str | None = None
    for entry in iter_manifest(manifest_path):
        if entry.video_id != last:
            if entry.video_id in seen:
                raise ValueError(
                    f"manifest khong lien tuc theo video: {entry.video_id} xuat hien lai. "
                    "Cac dong cua mot video phai nam lien nhau."
                )
            seen.append(entry.video_id)
            last = entry.video_id
    return seen


def gather_embeddings(
    manifest_path: str | Path,
    keyframes_dir: str | Path,
    filename: str,
    n_manifest: int,
) -> np.ndarray:
    """Gop embedding tung video thanh mot mang KHOP TUNG DONG voi manifest.

    Duyet manifest tu tren xuong; moi khi doi video thi nap file cua video do va
    dat lai con tro. Nho vay thu tu ket qua duoc quyet dinh boi manifest chu
    khong phai boi thu tu glob thu muc.

    Cap phat san mang dich (n_manifest x D) roi ghi tai cho, thay vi gom list rai
    rac roi stack - o quy mo 500k x 1024 float32 (~2GB) thi stack se nhan doi
    dinh bo nho.
    """
    keyframes_dir = Path(keyframes_dir)
    out: np.ndarray | None = None
    current_id: str | None = None
    current: np.ndarray | None = None
    cursor = 0
    written = 0

    def close_video() -> None:
        if current is not None and cursor != len(current):
            raise AssertionError(
                f"{current_id}: manifest dung {cursor} hang nhung {filename} co "
                f"{len(current)} hang. Chay lai buoc encode cho video nay."
            )

    for entry in iter_manifest(manifest_path):
        if entry.video_id != current_id:
            close_video()
            path = keyframes_dir / entry.video_id / filename
            if not path.exists():
                raise FileNotFoundError(f"Thieu {path}. Chay buoc encode truoc.")
            current = np.load(path)
            current_id = entry.video_id
            cursor = 0
            if out is None:
                out = np.zeros((n_manifest, current.shape[1]), dtype=np.float32)
            elif current.shape[1] != out.shape[1]:
                raise ValueError(
                    f"{entry.video_id}: dim {current.shape[1]} != {out.shape[1]}"
                )
        if written >= n_manifest:
            raise AssertionError(f"manifest dai hon {n_manifest} dong da bao")
        out[written] = current[cursor]
        cursor += 1
        written += 1
    close_video()

    if out is None:
        return np.zeros((0, 0), dtype=np.float32)
    if written != n_manifest:
        raise AssertionError(f"gop duoc {written} vector nhung manifest co {n_manifest} dong")
    return out


def load_clip_embeddings(clip_emb_path: str | Path, n_manifest: int) -> np.ndarray:
    """Doc lai embedding CLIP tu A.2. KHONG encode lai - do la ca diem cua A.2."""
    path = Path(clip_emb_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chua co {path}. Chay `02_keyframe.py --build-manifest` truoc."
        )
    emb = np.load(path)
    if len(emb) != n_manifest:
        raise AssertionError(
            f"clip_embeddings.npy co {len(emb)} hang nhung manifest co {n_manifest} dong. "
            "Build lai manifest."
        )
    return emb.astype(np.float32)


def build_indexes(
    manifest_path: str | Path,
    keyframes_dir: str | Path,
    clip_emb_path: str | Path,
    faiss_clip_path: str | Path,
    faiss_siglip_path: str | Path,
    index_meta_path: str | Path,
) -> dict[str, Any]:
    """Build ca hai FAISS index tu cung mot manifest, roi ghi meta de assert sau nay."""
    n_manifest = sum(1 for _ in iter_manifest(manifest_path))
    if n_manifest == 0:
        raise ValueError(f"{manifest_path} rong. Chay A.2 truoc.")

    clip_emb = load_clip_embeddings(clip_emb_path, n_manifest)
    siglip_emb = gather_embeddings(manifest_path, keyframes_dir, SIGLIP_EMB, n_manifest)
    if len(siglip_emb) != n_manifest:
        raise AssertionError(
            f"SigLIP2 co {len(siglip_emb)} vector nhung manifest co {n_manifest} dong"
        )

    clip_index = faiss_store.build_flat_ip(clip_emb, name="clip")
    faiss_store.save_index(clip_index, faiss_clip_path)

    siglip_index = faiss_store.build_flat_ip(siglip_emb, name="siglip2")
    faiss_store.save_index(siglip_index, faiss_siglip_path)

    ntotal = {"clip": int(clip_index.ntotal), "siglip2": int(siglip_index.ntotal)}
    write_index_meta(index_meta_path, n_manifest=n_manifest, ntotal=ntotal)

    return {
        "n_manifest": n_manifest,
        "ntotal": ntotal,
        "dim": {"clip": int(clip_emb.shape[1]), "siglip2": int(siglip_emb.shape[1])},
    }
