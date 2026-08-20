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


# MP4/MOV hop le co box 'ftyp' o byte 4-8. Matroska/WebM va AVI dung magic khac.
_FTYP_EXTS = {".mp4", ".m4v", ".mov"}
_NOT_VIDEO_PREFIXES = (b"<!DOCTYPE", b"<html", b"<HTML", b"{", b"version https://git-lfs")


def check_video_readable(video_path: str | Path) -> None:
    """Kiem tra file truoc khi dua vao ffmpeg, de loi ro rang thay vi 'ffmpeg error'.

    Chi doc 64 byte dau nen chay tren ca nghin file van gan nhu tuc thi. Doc 64
    chu khong phai 16 vi con tro Git LFS bat dau bang mot dong 23 ky tu.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Khong co file: {path}")

    size = path.stat().st_size
    if size == 0:
        raise ValueError(f"{path.name}: file rong (0 byte) - tai lai video")
    if size < 1024:
        raise ValueError(f"{path.name}: chi {size} byte - gan nhu chac chan tai loi")

    with open(path, "rb") as f:
        head = f.read(64)

    # Tai hong hay tra ve trang HTML bao loi / con tro Git LFS thay vi video.
    for prefix in _NOT_VIDEO_PREFIXES:
        if head.startswith(prefix):
            raise ValueError(
                f"{path.name}: khong phai file video ({size:,} byte, bat dau bang "
                f"{head[:24]!r}). Nhieu kha nang tai ve trang loi hoac con tro LFS."
            )

    if path.suffix.lower() in _FTYP_EXTS:
        if head[4:8] != b"ftyp":
            raise ValueError(
                f"{path.name}: khong tim thay box 'ftyp' o dau file ({size:,} byte). "
                "File MP4 bi cat cut hoac hong - tai lai."
            )
        check_mp4_complete(path, size)


def iter_mp4_boxes(path: str | Path, file_size: int):
    """Duyet cac box cap cao nhat cua MP4. Yield (loai, offset, kich thuoc).

    MP4 la chuoi box lien tiep, moi box mo dau bang 4 byte kich thuoc + 4 byte
    loai. Nho vay chi can vai lan seek la di het file, khong phai doc 24MB.
    """
    with open(path, "rb") as f:
        offset = 0
        while offset + 8 <= file_size:
            f.seek(offset)
            header = f.read(8)
            if len(header) < 8:
                return
            box_size = int.from_bytes(header[:4], "big")
            box_type = header[4:8]

            if box_size == 1:                       # kich thuoc 64-bit nam ngay sau
                ext = f.read(8)
                if len(ext) < 8:
                    return
                box_size = int.from_bytes(ext, "big")
            elif box_size == 0:                     # box keo den het file
                box_size = file_size - offset

            if box_size < 8:
                return                              # header hong, dung lai
            yield box_type, offset, box_size
            offset += box_size


def check_mp4_complete(path: str | Path, file_size: int) -> None:
    """Bat MP4 tai cat cut: co 'ftyp' o dau nhung thieu 'moov'.

    `moov` chua toan bo chi muc (so frame, codec, vi tri du lieu). Trong phan lon
    MP4 no nam o CUOI file, nen tai dut o giua cho ra dung trieu chung nay: dau
    file trong van hop le, ffmpeg chet ngay voi 'moov atom not found'.
    """
    name = Path(path).name
    types: list[bytes] = []
    for box_type, offset, box_size in iter_mp4_boxes(path, file_size):
        types.append(box_type)
        if box_type == b"moov":
            return
        if offset + box_size > file_size:
            raise ValueError(
                f"{name}: box '{box_type.decode('ascii', 'replace')}' khai bao "
                f"{box_size:,} byte tai offset {offset:,} nhung file chi co "
                f"{file_size:,} byte - file bi CAT CUT, tai lai."
            )

    seen = ", ".join(t.decode("ascii", "replace") for t in types[:6]) or "(khong doc duoc box nao)"
    raise ValueError(
        f"{name}: khong co box 'moov' ({file_size:,} byte, cac box: {seen}). "
        "'moov' chua chi muc cua video va thuong nam o cuoi file -> tai bi cat cut. "
        "Tai lai bang `wget -c` hoac doi chieu kich thuoc voi nguon."
    )


def probe_fps(video_path: str | Path) -> float:
    """Lay fps bang ffprobe. RAISE khi probe that bai.

    KHONG dung model.get_video_fps() cua package: ham do nuot moi loi va tra ve
    25.0, nen mot video probe hong se lang le co fps sai, keo theo `pts_time` sai
    o toan bo pipeline ma khong co dau hieu gi.
    """
    import ffmpeg

    try:
        probe = ffmpeg.probe(str(video_path))
    except ffmpeg.Error as exc:
        raise RuntimeError(_ffmpeg_message(exc, video_path, "ffprobe")) from exc

    stream = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    if stream is None:
        raise ValueError(f"{Path(video_path).name}: khong co luong video nao trong file")

    rate = stream.get("r_frame_rate") or stream.get("avg_frame_rate") or ""
    try:
        if "/" in rate:
            num, den = rate.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(rate)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{Path(video_path).name}: r_frame_rate la '{rate}'") from exc

    if fps <= 0:
        raise ValueError(f"{Path(video_path).name}: fps = {fps}")
    return fps


def _ffmpeg_message(exc: Exception, video_path: str | Path, tool: str = "ffmpeg") -> str:
    """Boc stderr cua ffmpeg vao thong bao loi.

    ffmpeg-python bat stderr vao thuoc tinh cua exception nhung __str__ chi in
    'ffmpeg error (see stderr output for detail)' - cau do khong noi len dieu gi.
    """
    raw = getattr(exc, "stderr", None) or b""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    lines = [line for line in raw.strip().splitlines() if line.strip()]
    tail = "\n    ".join(lines[-6:]) if lines else "(khong co stderr)"
    return f"{tool} that bai voi {Path(video_path).name}:\n    {tail}"


def predict_raw_frames(model, video_path: str | Path) -> tuple[np.ndarray, float]:
    """Chay inference, tra ve (single_frame_pred, fps).

    `predict_video` tra ve (video_frames, single_frame_pred, all_frame_pred).
    `video_frames` la ca video da resize 48x27 nam tren device (~3.9KB/frame,
    100k frame ~ 390MB VRAM) - ta khong dung toi nen tha tham chieu ngay.
    """
    import ffmpeg
    import torch

    check_video_readable(video_path)
    fps = probe_fps(video_path)

    try:
        with torch.no_grad():
            out = model.predict_video(str(video_path), quiet=True)
            single = out[1].cpu().detach().numpy().astype(np.float32)
        del out
    except ffmpeg.Error as exc:
        raise RuntimeError(_ffmpeg_message(exc, video_path)) from exc

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
