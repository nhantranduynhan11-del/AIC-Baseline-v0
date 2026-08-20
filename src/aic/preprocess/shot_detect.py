"""A.1 Shot Detection - TransNetV2, threshold = 0.5.

Dung package `transnetv2-pytorch` (PyPI v1.0.5). Weights bundle san trong wheel
va tu load trong __init__ -> khong can tai tay, khong can vendor repo goc.

Output moi video: paths.shots/<video_id>.json  (xem SHOT_JSON_VERSION)
Kem theo (neu bat): paths.shots_raw/<video_id>.npy chua single_frame_pred tho.
Co file .npy nay thi doi threshold ve sau chi ton vai giay - khong phai chay lai
inference tren GPU thue.

torch va transnetv2_pytorch duoc import LAZY trong tung ham, de module nay
import duoc tren may dev khong cai torch (chay test, doc config).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

SHOT_JSON_VERSION = 1


def build_model(device: str = "auto"):
    """Tao TransNetV2. Weights tu load trong __init__.

    device: "auto" | "cuda" | "cpu". "auto" tu do CUDA -> tren vast.ai se ra cuda,
    tren may dev khong GPU se ra cpu.
    """
    from transnetv2_pytorch import TransNetV2

    model = TransNetV2(device=device)
    model.eval()
    return model


def predict_raw_frames(model, video_path: str | Path) -> tuple[np.ndarray, float]:
    """Chay inference, tra ve (single_frame_pred, fps).

    `predict_video` tra ve (video_frames, single_frame_pred, all_frame_pred).
    `video_frames` la ca video da resize 48x27 nam tren device (~3.9KB/frame,
    100k frame ~ 390MB VRAM) - ta khong dung toi nen tha tham chieu ngay.

    ⚠️ get_video_fps() cua package tra ve 25.0 khi ffmpeg probe that bai, KHONG
    raise. Ham nay doi chieu voi so frame de it nhat phat hien duoc truong hop la.
    """
    import torch

    fps = float(model.get_video_fps(str(video_path)))

    with torch.no_grad():
        out = model.predict_video(str(video_path), quiet=True)
        single = out[1].cpu().detach().numpy().astype(np.float32)
    del out

    return single, fps


def predictions_to_shots(
    predictions: np.ndarray, fps: float, threshold: float = 0.5
) -> list[dict[str, Any]]:
    """Bien prediction tho thanh danh sach shot.

    Dung thang `TransNetV2.predictions_to_scenes` (staticmethod cua package) chu
    khong tu cai dat lai, de khong bao gio lech voi thuat toan goc.
    `start_time`/`end_time` cua package la CHUOI timestamp, nen o day tu tinh lai
    bang giay dang float tu frame/fps - A.2 can so de dua vao `pts_time`.
    """
    from transnetv2_pytorch import TransNetV2

    predictions = np.asarray(predictions).reshape(-1)
    scenes = TransNetV2.predictions_to_scenes(predictions, threshold)
    return _scenes_to_shots(scenes, predictions, fps)


def _scenes_to_shots(
    scenes: Iterable[Sequence[int]], predictions: np.ndarray, fps: float
) -> list[dict[str, Any]]:
    """Phan thuan tuy - tach ra de test duoc ma khong can torch."""
    if fps <= 0:
        raise ValueError(f"fps khong hop le: {fps}")

    shots: list[dict[str, Any]] = []
    for i, (start, end) in enumerate(scenes):
        start, end = int(start), int(end)
        window = predictions[start : end + 1]
        shots.append(
            {
                "shot_id": i + 1,
                "start_frame": start,
                "end_frame": end,
                "start_time": round(start / fps, 4),
                "end_time": round(end / fps, 4),
                "probability": round(float(window.max()), 6) if window.size else 0.0,
            }
        )
    return shots


def detect_shots(
    model, video_path: str | Path, threshold: float = 0.5
) -> tuple[dict[str, Any], np.ndarray]:
    """Chay A.1 cho mot video. Tra ve (record de ghi JSON, prediction tho)."""
    video_path = Path(video_path)
    predictions, fps = predict_raw_frames(model, video_path)
    shots = predictions_to_shots(predictions, fps, threshold)
    return make_record(video_path, predictions, fps, threshold, shots), predictions


def make_record(
    video_path: Path,
    predictions: np.ndarray,
    fps: float,
    threshold: float,
    shots: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": SHOT_JSON_VERSION,
        "video_id": video_path.stem,
        "video_name": video_path.name,
        "model": "transnetv2-pytorch",
        "threshold": threshold,
        "fps": fps,
        "n_frames": int(len(predictions)),
        "n_shots": len(shots),
        "shots": shots,
    }


def write_shots(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)


def read_shots(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        record = json.load(f)
    if record.get("version") != SHOT_JSON_VERSION:
        raise ValueError(
            f"{path}: version {record.get('version')} != {SHOT_JSON_VERSION}. Chay lai A.1."
        )
    return record


def save_raw(path: str | Path, predictions: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, predictions.astype(np.float32))


def rethreshold(raw_path: str | Path, video_path: Path, fps: float, threshold: float) -> dict:
    """Doi threshold tu file .npy da luu, khong chay lai inference."""
    predictions = np.load(raw_path)
    shots = predictions_to_shots(predictions, fps, threshold)
    return make_record(video_path, predictions, fps, threshold, shots)


def find_videos(videos_dir: str | Path, exts: Sequence[str]) -> list[Path]:
    videos_dir = Path(videos_dir)
    exts = {e.lower() for e in exts}
    return sorted(
        p for p in videos_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )
