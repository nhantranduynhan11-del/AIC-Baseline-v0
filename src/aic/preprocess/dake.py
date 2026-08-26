"""DAKE - trích keyframe bằng heuristic, để CHẠY THỬ SO SÁNH với baseline.

⚠️ ĐÂY KHÔNG PHẢI BASELINE. Baseline đã chốt là TransNetV2 (A.1) + CLIP ViT-L
với ngưỡng L2 0.4 (A.2). DAKE thay cả hai bước đó bằng một lượt quét duy nhất:

  - Cắt cảnh: độ lệch KÍCH THƯỚC JPEG giữa hai frame liên tiếp, so với ngưỡng
    thích nghi tính từ cửa sổ trượt 60 frame. Nén JPEG là proxy rẻ cho "độ phức
    tạp ảnh", nên ảnh đổi nhiều thì kích thước đổi nhiều.
  - Phân loại trạng thái: biến động dưới sàn nhiễu -> coi là video TĨNH
    (slide/bảng giảng), ngược lại là HÀNH ĐỘNG.
  - Khi TĨNH: bật lưới 4x4 absdiff với ngưỡng thấp để bắt chữ mới xuất hiện.
  - Khi HÀNH ĐỘNG: chỉ bắt cut cảnh, khoá luồng lưới.
  - Cooldown khác nhau theo trạng thái.

Thuật toán giữ NGUYÊN VẸN từ scripts/DAKE.py. Ở đây chỉ đổi ba thứ về mặt kỹ
thuật, không đụng tới quyết định chọn frame:

  1. `yield (frame_idx, frame)` thay vì tự ghi file — để phía gọi đặt tên theo
     `<frame_idx>.jpg` đúng chuẩn pipeline. Nộp bài cho BTC cần frame_idx, mà
     bản gốc đặt tên theo số thứ tự shot + timestamp nên không truy ra được.
  2. fps lấy bằng ffprobe (cùng nguồn với cả pipeline) thay vì
     cv2.CAP_PROP_FPS với mặc định 30.0 khi hỏng. Chi phí thêm ~0.1s/video,
     không đáng kể so với việc nén JPEG mọi frame.
  3. Tham số nhận từ ngoài để dò ngưỡng mà không phải sửa code.

Chi phí: `cv2.imencode` chạy trên MỌI frame - đó là tín hiệu chính của thuật
toán nên không bỏ được. Video 27k frame là 27k lần nén JPEG trên CPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# Mặc định lấy đúng từ scripts/DAKE.py, không đổi.
K_GLOBAL = 2
LOCAL_THRESHOLD_STATIC = 10.0
MIN_FRAME_GAP_ACTION = 5
MIN_FRAME_GAP_STATIC = 45
WINDOW_SIZE = 60
GRID_ROWS, GRID_COLS = 4, 4


@dataclass(frozen=True)
class DakeParams:
    k_global: float = K_GLOBAL
    local_threshold_static: float = LOCAL_THRESHOLD_STATIC
    min_frame_gap_action: int = MIN_FRAME_GAP_ACTION
    min_frame_gap_static: int = MIN_FRAME_GAP_STATIC
    window_size: int = WINDOW_SIZE


@dataclass(frozen=True)
class DakeKeyframe:
    frame_idx: int
    trigger: str          # "FIRST" | "GLOBAL_CUT_ACTION" | "LOCAL_TEXT_STATIC"


def iter_keyframes(
    video_path: str | Path, params: DakeParams | None = None
) -> Iterator[tuple[DakeKeyframe, np.ndarray]]:
    """Duyệt video, yield (thông tin keyframe, ảnh BGR) cho mỗi frame được chọn.

    `frame_idx` là số thứ tự frame theo lượt đọc tuần tự — cùng định nghĩa với
    A.2 của pipeline, nên hai bên so sánh được với nhau và nộp bài dùng được.
    """
    import cv2

    params = params or DakeParams()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {video_path}")

    try:
        global_delta_window: deque = deque(maxlen=params.window_size)
        prev_gray = None
        prev_jpeg_size = None
        last_saved_gray = None
        frame_count = 0
        last_keyframe_frame = -params.min_frame_gap_static

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            success, encoded = cv2.imencode(".jpg", frame)
            if not success:
                continue
            current_jpeg_size = len(encoded.tobytes())

            # Frame đầu tiên luôn được giữ - chưa có gì để so sánh.
            if prev_gray is None:
                yield DakeKeyframe(frame_count, "FIRST"), frame
                prev_gray = gray
                last_saved_gray = gray
                prev_jpeg_size = current_jpeg_size
                last_keyframe_frame = frame_count
                frame_count += 1
                continue

            global_delta = abs(current_jpeg_size - prev_jpeg_size)

            # Chưa đủ cửa sổ thì chỉ nạp thống kê, chưa quyết định gì.
            if len(global_delta_window) < params.window_size:
                global_delta_window.append(global_delta)
                prev_gray = gray
                prev_jpeg_size = current_jpeg_size
                frame_count += 1
                continue

            local_mean = float(np.mean(global_delta_window))
            local_std = float(np.std(global_delta_window))
            noise_floor = max(current_jpeg_size * 0.012, 2500.0)
            is_video_static = global_delta < noise_floor

            current_gap = (
                params.min_frame_gap_static if is_video_static else params.min_frame_gap_action
            )
            trigger = ""

            if (frame_count - last_keyframe_frame) >= current_gap:
                if not is_video_static:
                    threshold = local_mean + params.k_global * max(
                        local_std, noise_floor / params.k_global
                    )
                    if global_delta > threshold:
                        trigger = "GLOBAL_CUT_ACTION"
                else:
                    content_diff = cv2.absdiff(gray, last_saved_gray)
                    h, w = content_diff.shape
                    row_step, col_step = h // GRID_ROWS, w // GRID_COLS
                    max_grid_change = 0.0
                    for r in range(GRID_ROWS):
                        for c in range(GRID_COLS):
                            roi = content_diff[
                                r * row_step : (r + 1) * row_step,
                                c * col_step : (c + 1) * col_step,
                            ]
                            max_grid_change = max(max_grid_change, float(np.mean(roi)))
                    if max_grid_change > params.local_threshold_static:
                        trigger = "LOCAL_TEXT_STATIC"

            if trigger:
                yield DakeKeyframe(frame_count, trigger), frame
                last_keyframe_frame = frame_count
                last_saved_gray = gray

            global_delta_window.append(global_delta)
            prev_gray = gray
            prev_jpeg_size = current_jpeg_size
            frame_count += 1
    finally:
        cap.release()


def extract_video(
    video_path: str | Path,
    out_dir: str | Path,
    *,
    params: DakeParams | None = None,
    jpeg_quality: int = 95,
) -> dict:
    """Chạy DAKE cho một video, ghi ảnh + keyframes.json ĐÚNG SCHEMA của A.2.

    Cùng schema để so sánh trực tiếp với kết quả baseline, và để sau này nếu
    nhóm chốt dùng DAKE thì chỉ việc trỏ pipeline sang thư mục này.
    """
    import cv2

    from aic.preprocess.keyframe import KEYFRAME_META, KEYFRAME_META_VERSION, write_keyframe_meta
    from aic.preprocess.shot_detect import probe_fps

    video_path = Path(video_path)
    video_id = video_path.stem
    params = params or DakeParams()

    fps = probe_fps(video_path)          # cùng nguồn với cả pipeline, raise khi hỏng
    out_dir = Path(out_dir) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    triggers: dict[str, int] = {}
    for info, frame in iter_keyframes(video_path, params):
        cv2.imwrite(
            str(out_dir / f"{info.frame_idx}.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
        )
        kept.append(
            {
                "frame_idx": info.frame_idx,
                "shot_id": len(kept) + 1,      # DAKE không tách shot riêng
                "pts_time": round(info.frame_idx / fps, 4),
                "path": f"{video_id}/{info.frame_idx}.jpg",
                "trigger": info.trigger,
            }
        )
        triggers[info.trigger] = triggers.get(info.trigger, 0) + 1

    meta = {
        "version": KEYFRAME_META_VERSION,
        "video_id": video_id,
        "fps": fps,
        "sample_every": 1,                 # DAKE xét mọi frame
        "l2_threshold": None,              # không dùng
        "clip_model": None,                # không dùng model
        "dim": None,
        "n_sampled": None,
        "n_keyframes": len(kept),
        "method": "dake",
        "params": params.__dict__,
        "triggers": triggers,
        "keyframes": kept,
    }
    write_keyframe_meta(out_dir / KEYFRAME_META, meta)
    return meta
