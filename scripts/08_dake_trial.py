"""Chạy thử DAKE và so sánh với keyframe của baseline. KHÔNG đụng index hiện có.

    python scripts/08_dake_trial.py --videos data/videos/L25 --limit 2
    python scripts/08_dake_trial.py --videos data/videos/L25
    python scripts/08_dake_trial.py --compare-only

Ghi ra `data/keyframes_dake/<video_id>/` — thư mục RIÊNG, tách hẳn khỏi
`data/keyframes/`. Manifest, hai FAISS index và metadata.db đều không bị chạm
tới, nên hệ thống đang chạy vẫn nguyên vẹn trong lúc thử.

DAKE thay cả A.1 lẫn A.2 của baseline (TransNetV2 + CLIP/L2). Muốn chốt dùng nó
thì phải chạy lại cả chuỗi: encode CLIP + SigLIP2, OCR, rồi build lại manifest
và hai index — vì mọi thứ đều khoá theo (video_id, frame_idx).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

from aic.config import load_config
from aic.sharding import select_shard
from aic.preprocess import dake
from aic.preprocess.keyframe import KEYFRAME_META
from aic.preprocess.shot_detect import find_videos

OUT_DIRNAME = "keyframes_dake"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chạy thử DAKE, so với baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--videos", default=None, help="Thư mục video (mặc định paths.videos)")
    p.add_argument("--out", default=None, help="Mặc định <data_root>/keyframes_dake")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--compare-only", action="store_true",
                   help="Chỉ so sánh kết quả đã có, không chạy lại")
    p.add_argument("--k-global", type=float, default=dake.K_GLOBAL)
    p.add_argument("--local-threshold-static", type=float, default=dake.LOCAL_THRESHOLD_STATIC)
    p.add_argument("--min-frame-gap-action", type=int, default=dake.MIN_FRAME_GAP_ACTION)
    p.add_argument("--min-frame-gap-static", type=int, default=dake.MIN_FRAME_GAP_STATIC)
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="Chỉ xử lý phần thứ I trong N phần (vd 0/3), để chạy nhiều tiến trình")
    return p.parse_args()


def baseline_counts(keyframes_dir: Path) -> dict[str, int]:
    """Số keyframe của baseline cho từng video, đọc từ keyframes.json."""
    out = {}
    for meta_path in keyframes_dir.glob(f"*/{KEYFRAME_META}"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            out[meta["video_id"]] = meta["n_keyframes"]
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def compare(out_dir: Path, keyframes_dir: Path) -> int:
    base = baseline_counts(keyframes_dir)
    rows = []
    for meta_path in sorted(out_dir.glob(f"*/{KEYFRAME_META}")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rows.append((meta["video_id"], meta["n_keyframes"], base.get(meta["video_id"]),
                     meta.get("triggers", {})))
    if not rows:
        print(f"Chưa có kết quả DAKE nào trong {out_dir}", file=sys.stderr)
        return 1

    print(f"\n{'video':<14}{'DAKE':>8}{'baseline':>10}{'tỉ lệ':>9}   trigger")
    print("-" * 72)
    total_dake = total_base = 0
    for video_id, n_dake, n_base, triggers in rows:
        total_dake += n_dake
        ratio = f"{n_dake / n_base:.2f}x" if n_base else "—"
        total_base += n_base or 0
        trig = " ".join(f"{k.split('_')[0].lower()}={v}" for k, v in sorted(triggers.items()))
        print(f"{video_id:<14}{n_dake:>8,}{(n_base or 0):>10,}{ratio:>9}   {trig}")

    print("-" * 72)
    overall = f"{total_dake / total_base:.2f}x" if total_base else "—"
    print(f"{'TỔNG':<14}{total_dake:>8,}{total_base:>10,}{overall:>9}")
    print(f"\n{len(rows)} video. DAKE {'nhiều' if total_dake > total_base else 'ít'} keyframe hơn "
          f"baseline {abs(total_dake - total_base):,} ảnh.")
    print("Số lượng chỉ là một mặt — phải xem tận mắt vài video mới biết cái nào bắt đúng cảnh.")
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    keyframes_dir = Path(cfg.paths.keyframes)
    out_dir = Path(args.out) if args.out else Path(cfg.paths.data_root) / OUT_DIRNAME

    if args.compare_only:
        return compare(out_dir, keyframes_dir)

    videos_dir = Path(args.videos or cfg.paths.videos)
    videos = select_shard(find_videos(videos_dir, cfg.shot_detection.video_ext), args.shard)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"Không tìm thấy video nào trong {videos_dir}", file=sys.stderr)
        return 1

    todo = videos if args.overwrite else [
        v for v in videos if not (out_dir / v.stem / KEYFRAME_META).exists()
    ]
    params = dake.DakeParams(
        k_global=args.k_global,
        local_threshold_static=args.local_threshold_static,
        min_frame_gap_action=args.min_frame_gap_action,
        min_frame_gap_static=args.min_frame_gap_static,
    )
    print(f"{len(videos)} video, {len(todo)} cần chạy -> {out_dir}")
    print(f"tham số: {params.__dict__}")
    if not todo:
        return compare(out_dir, keyframes_dir)

    base = baseline_counts(keyframes_dir)
    failed: list[tuple[str, str]] = []
    t0 = time.time()
    for i, video in enumerate(todo, 1):
        try:
            t1 = time.time()
            meta = dake.extract_video(
                video, out_dir, params=params, jpeg_quality=cfg.keyframe.jpeg_quality
            )
            n_base = base.get(video.stem)
            ratio = f", baseline {n_base:,} ({meta['n_keyframes']/n_base:.2f}x)" if n_base else ""
            print(f"[{i}/{len(todo)}] {video.name}: {meta['n_keyframes']:,} keyframe"
                  f"{ratio}, {time.time() - t1:.1f}s  {meta['triggers']}")
        except Exception as exc:
            failed.append((video.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(todo)}] LỖI {video.name}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\nXong {len(todo) - len(failed)}/{len(todo)} video, {time.time() - t0:.1f}s")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}", file=sys.stderr)
    compare(out_dir, keyframes_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
