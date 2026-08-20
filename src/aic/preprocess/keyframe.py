"""A.2 Keyframe Extraction - CLIP ViT-L-14-quickgelu + L2-Norm.

Thuat toan, trong tung shot:
  1. Cu moi `sample_every` (=8) frame thi trich dac trung bang CLIP.
  2. Frame lay mau DAU TIEN cua shot = anchor: giu ngay, khong so sanh (chua co
     gi de so).
  3. Cac frame sau: tinh khoang cach Euclidean toi embedding cua KEYFRAME GAN
     NHAT DA CHON.
        > 0.4  -> giu lam keyframe moi, va no thanh moc so sanh moi
        <= 0.4 -> loai vi qua giong keyframe hien co

Embedding da normalize L2 (xem clip_encoder) nen khoang cach nam trong [0, 2].

TAI SU DUNG EMBEDDING - bat buoc. Embedding cua nhung frame duoc chon duoc ghi
ra dia (`<video_id>/clip.npy`) va dung THANG lam vector index CLIP o A.3.
Khong encode lai o A.3.

Bo nho: xu ly theo lo `keyframe.batch_size` frame lay mau. Viec chia lo KHONG
lam doi ket qua chon, vi trang thai `_ref` duoc mang qua giua cac lo; chi reset
khi sang shot moi.

torch/cv2/open_clip import LAZY -> module import duoc tren may khong cai torch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from aic.manifest import KeyframeEntry, write_index_meta, write_manifest

KEYFRAME_META = "keyframes.json"
KEYFRAME_EMB = "clip.npy"
KEYFRAME_META_VERSION = 1


class KeyframeSelector:
    """Trang thai chon keyframe. Tach rieng de test duoc ma khong can torch."""

    def __init__(self, l2_threshold: float = 0.4):
        if l2_threshold <= 0:
            raise ValueError(f"l2_threshold phai > 0, nhan {l2_threshold}")
        self.l2_threshold = float(l2_threshold)
        self._ref: np.ndarray | None = None

    def start_shot(self) -> None:
        """Sang shot moi -> bo moc so sanh, frame lay mau dau tien se la anchor."""
        self._ref = None

    def consider(self, embedding: np.ndarray) -> bool:
        """True neu frame nay duoc giu lam keyframe."""
        if self._ref is None:
            self._ref = embedding
            return True
        distance = float(np.linalg.norm(embedding - self._ref))
        if distance > self.l2_threshold:
            self._ref = embedding
            return True
        return False


def build_sample_plan(
    shots: Sequence[dict[str, Any]], sample_every: int = 8
) -> list[tuple[int, int]]:
    """Tra ve [(frame_idx, shot_id), ...] tang dan theo frame_idx.

    Trong moi shot lay start_frame, start_frame + 8, ... <= end_frame.
    """
    if sample_every < 1:
        raise ValueError(f"sample_every phai >= 1, nhan {sample_every}")
    plan: list[tuple[int, int]] = []
    for shot in shots:
        start, end = int(shot["start_frame"]), int(shot["end_frame"])
        plan.extend((idx, int(shot["shot_id"])) for idx in range(start, end + 1, sample_every))
    plan.sort(key=lambda item: item[0])
    return plan


def iter_planned_frames(
    video_path: str | Path, plan: Sequence[tuple[int, int]]
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Doc video MOT LUOT tuan tu, chi giai ma nhung frame nam trong plan.

    cv2.grab() bo qua frame khong can (khong giai ma ra RGB), cv2.retrieve() chi
    goi cho frame can lay -> nhanh hon nhieu so voi seek tung frame.

    Yield (frame_idx, shot_id, frame_bgr).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    try:
        targets = iter(plan)
        target = next(targets, None)
        idx = 0
        while target is not None:
            if not cap.grab():
                break
            if idx == target[0]:
                ok, frame = cap.retrieve()
                if ok:
                    yield idx, target[1], frame
                target = next(targets, None)
            idx += 1
    finally:
        cap.release()


def extract_video(
    encoder,
    video_path: str | Path,
    record: dict[str, Any],
    out_root: str | Path,
    *,
    sample_every: int = 8,
    l2_threshold: float = 0.4,
    batch_size: int = 32,
    jpeg_quality: int = 95,
) -> dict[str, Any]:
    """Chay A.2 cho mot video. Ghi anh keyframe + clip.npy + keyframes.json.

    `record`: ket qua A.1 doc tu data/shots/<video_id>.json (can fps va shots).
    """
    import cv2

    video_path = Path(video_path)
    video_id = record["video_id"]
    fps = float(record["fps"])
    if fps <= 0:
        raise ValueError(f"{video_id}: fps khong hop le ({fps})")

    out_dir = Path(out_root) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = build_sample_plan(record["shots"], sample_every)
    selector = KeyframeSelector(l2_threshold)
    state = {"shot": None}

    kept: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    buffer: list[tuple[int, int, np.ndarray]] = []

    def flush() -> None:
        if not buffer:
            return
        feats = encoder.encode_images(
            [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for _, _, frame in buffer]
        )
        for (frame_idx, shot_id, frame_bgr), emb in zip(buffer, feats):
            if shot_id != state["shot"]:
                selector.start_shot()
                state["shot"] = shot_id
            if not selector.consider(emb):
                continue
            cv2.imwrite(
                str(out_dir / f"{frame_idx}.jpg"),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
            )
            kept.append(
                {
                    "frame_idx": frame_idx,
                    "shot_id": shot_id,
                    "pts_time": round(frame_idx / fps, 4),
                    "path": f"{video_id}/{frame_idx}.jpg",
                }
            )
            embeddings.append(emb)
        buffer.clear()

    for item in iter_planned_frames(video_path, plan):
        buffer.append(item)
        if len(buffer) >= batch_size:
            flush()
    flush()

    emb_array = (
        np.stack(embeddings).astype(np.float32)
        if embeddings
        else np.zeros((0, encoder.dim), dtype=np.float32)
    )
    np.save(out_dir / KEYFRAME_EMB, emb_array)

    meta = {
        "version": KEYFRAME_META_VERSION,
        "video_id": video_id,
        "fps": fps,
        "sample_every": sample_every,
        "l2_threshold": l2_threshold,
        "clip_model": f"{encoder.name}/{encoder.pretrained}",
        "dim": int(emb_array.shape[1]) if emb_array.size else encoder.dim,
        "n_sampled": len(plan),
        "n_keyframes": len(kept),
        "keyframes": kept,
    }
    write_keyframe_meta(out_dir / KEYFRAME_META, meta)
    return meta


def write_keyframe_meta(path: str | Path, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)


def read_keyframe_meta(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("version") != KEYFRAME_META_VERSION:
        raise ValueError(
            f"{path}: version {meta.get('version')} != {KEYFRAME_META_VERSION}. Chay lai A.2."
        )
    return meta


def build_manifest(
    keyframes_dir: str | Path,
    manifest_path: str | Path,
    clip_emb_path: str | Path,
    index_meta_path: str | Path | None = None,
) -> tuple[int, int]:
    """Gop ket qua tung video thanh MOT manifest + MOT mang embedding CLIP.

    Day la buoc sinh ra bat bien ID cua A.6: dong i cua manifest ung voi hang i
    cua clip_embeddings.npy, va sau nay la row i cua ca hai FAISS index.

    Duyet thu muc video theo THU TU TEN da sap xep, trong moi video theo dung
    thu tu keyframe da ghi. Thu tu nay co dinh, A.3 khong bao gio duoc sort lai.

    Tra ve (so keyframe, so chieu embedding).
    """
    keyframes_dir = Path(keyframes_dir)
    entries: list[KeyframeEntry] = []
    blocks: list[np.ndarray] = []
    dim = 0

    for meta_path in sorted(keyframes_dir.glob(f"*/{KEYFRAME_META}")):
        meta = read_keyframe_meta(meta_path)
        emb = np.load(meta_path.parent / KEYFRAME_EMB)
        if len(emb) != meta["n_keyframes"]:
            raise ValueError(
                f"{meta['video_id']}: clip.npy co {len(emb)} hang nhung meta ghi "
                f"{meta['n_keyframes']} keyframe. Chay lai A.2 cho video nay."
            )
        if emb.size:
            if dim and emb.shape[1] != dim:
                raise ValueError(f"{meta['video_id']}: dim {emb.shape[1]} != {dim}")
            dim = emb.shape[1]
            blocks.append(emb)
        for kf in meta["keyframes"]:
            entries.append(
                KeyframeEntry(
                    idx=-1,  # write_manifest gan lai theo dung thu tu ghi
                    video_id=meta["video_id"],
                    frame_idx=int(kf["frame_idx"]),
                    pts_time=float(kf["pts_time"]),
                    path=kf["path"],
                )
            )

    n = write_manifest(manifest_path, entries)
    all_emb = (
        np.concatenate(blocks).astype(np.float32)
        if blocks
        else np.zeros((0, dim), dtype=np.float32)
    )
    if len(all_emb) != n:
        raise AssertionError(f"embedding {len(all_emb)} hang != manifest {n} dong")

    Path(clip_emb_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(clip_emb_path, all_emb)
    if index_meta_path is not None:
        write_index_meta(index_meta_path, n_manifest=n, ntotal={"clip_embeddings": n})
    return n, dim
