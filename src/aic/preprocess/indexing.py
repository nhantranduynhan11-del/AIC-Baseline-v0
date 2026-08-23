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


def iter_video_blocks(
    manifest_path: str | Path,
    keyframes_dir: str | Path,
    filename: str,
    n_manifest: int,
):
    """Yield embedding cua TUNG VIDEO, dung thu tu manifest.

    Cung phep kiem tra nhu gather_embeddings nhung khong dung ca mang dich, nen
    dinh bo nho chi bang mot video thay vi ca bo. Dung cho may it RAM.
    """
    keyframes_dir = Path(keyframes_dir)
    current_id: str | None = None
    cursor = 0
    written = 0
    dim = 0
    current: np.ndarray | None = None

    def close_video() -> None:
        if current is not None and cursor != len(current):
            raise AssertionError(
                f"{current_id}: manifest dung {cursor} hang nhung {filename} co "
                f"{len(current)} hang. Chay lai buoc encode cho video nay."
            )

    for entry in iter_manifest(manifest_path):
        if entry.video_id != current_id:
            close_video()
            if current is not None:
                yield current            # tra ca video vua duyet xong
            path = keyframes_dir / entry.video_id / filename
            if not path.exists():
                raise FileNotFoundError(f"Thieu {path}. Chay buoc encode truoc.")
            current = np.load(path)
            if dim and current.shape[1] != dim:
                raise ValueError(f"{entry.video_id}: dim {current.shape[1]} != {dim}")
            dim = current.shape[1]
            current_id = entry.video_id
            cursor = 0
        cursor += 1
        written += 1

    close_video()
    if current is not None:
        yield current
    if written != n_manifest:
        raise AssertionError(f"duyet duoc {written} hang nhung manifest co {n_manifest} dong")


def iter_array_chunks(path: str | Path, n_manifest: int, chunk: int = 50_000):
    """Yield tung khoi cua mot file .npy lon, doc bang mmap.

    mmap_mode="r" khong nap ca file vao RAM; moi khoi duoc sao chep ra rieng roi
    tha ngay, nen doc file 0.8 GB chi ton bo nho bang mot khoi.
    """
    array = np.load(path, mmap_mode="r")
    if len(array) != n_manifest:
        raise AssertionError(
            f"{Path(path).name} co {len(array)} hang nhung manifest co {n_manifest} dong. "
            "Build lai manifest."
        )
    for start in range(0, len(array), chunk):
        yield np.array(array[start : start + chunk], dtype=np.float32)


def embedding_dim(path: str | Path) -> int:
    """So chieu cua file .npy ma khong nap noi dung."""
    return int(np.load(path, mmap_mode="r").shape[1])


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
    """Build ca hai FAISS index tu cung mot manifest, roi ghi meta de assert sau nay.

    Build TUNG INDEX MOT roi tha ngay, va nap embedding theo dong thay vi nap ca
    bo. O 255k keyframe, giu ca 4 mang cung luc la 3.4 GB; lam theo dong thi dinh
    bo nho chi con bang index lon nhat (~1 GB).
    """
    import gc

    n_manifest = sum(1 for _ in iter_manifest(manifest_path))
    if n_manifest == 0:
        raise ValueError(f"{manifest_path} rong. Chay A.2 truoc.")

    if not Path(clip_emb_path).exists():
        raise FileNotFoundError(
            f"Chua co {clip_emb_path}. Chay `02_keyframe.py --build-manifest` truoc."
        )

    dims: dict[str, int] = {}
    ntotal: dict[str, int] = {}

    # CLIP: doc lai file da co tu A.2 bang mmap, KHONG encode lai.
    dims["clip"] = embedding_dim(clip_emb_path)
    index = faiss_store.build_flat_ip_streaming(
        dims["clip"], iter_array_chunks(clip_emb_path, n_manifest), name="clip"
    )
    ntotal["clip"] = int(index.ntotal)
    faiss_store.save_index(index, faiss_clip_path)
    del index
    gc.collect()

    # SigLIP2: gom tung video theo dung thu tu manifest.
    dims["siglip2"] = embedding_dim(
        Path(keyframes_dir) / next(iter_manifest(manifest_path)).video_id / SIGLIP_EMB
    )
    index = faiss_store.build_flat_ip_streaming(
        dims["siglip2"],
        iter_video_blocks(manifest_path, keyframes_dir, SIGLIP_EMB, n_manifest),
        name="siglip2",
    )
    ntotal["siglip2"] = int(index.ntotal)
    faiss_store.save_index(index, faiss_siglip_path)
    del index
    gc.collect()

    for name, total in ntotal.items():
        if total != n_manifest:
            raise AssertionError(
                f"{name}: index co {total} vector nhung manifest co {n_manifest} dong"
            )

    write_index_meta(index_meta_path, n_manifest=n_manifest, ntotal=ntotal)
    return {"n_manifest": n_manifest, "ntotal": ntotal, "dim": dims}
