"""A.1 Shot Detection - chay TransNetV2 tren toan bo video.

Chay tren vast.ai:
    python scripts/01_shot_detect.py
    python scripts/01_shot_detect.py --videos /data/videos --device cuda
    python scripts/01_shot_detect.py --check-only            # chi kiem tra file co doc duoc khong
    python scripts/01_shot_detect.py --rethreshold 0.4      # dung .npy da luu, khong inference lai

Mac dinh BO QUA video da co file JSON -> dut mang giua chung thi chay lai la resume.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

from aic.config import load_config
from aic.sharding import select_shard
from aic.preprocess import shot_detect as sd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A.1 Shot Detection (TransNetV2)")
    p.add_argument("--config", default=None, help="Duong dan config YAML")
    p.add_argument("--videos", default=None, help="Ghi de paths.videos")
    p.add_argument("--out", default=None, help="Ghi de paths.shots")
    p.add_argument("--device", default=None, help="auto | cuda | cpu")
    p.add_argument("--threshold", type=float, default=None, help="Ghi de shot_detection.threshold")
    p.add_argument("--overwrite", action="store_true", help="Chay lai ca video da co JSON")
    p.add_argument("--limit", type=int, default=None, help="Chi xu ly N video dau (de thu)")
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Chi kiem tra file video doc duoc hay khong, khong chay model",
    )
    p.add_argument(
        "--rethreshold",
        type=float,
        default=None,
        metavar="T",
        help="Tao lai JSON tu .npy da luu voi threshold moi. Khong chay inference.",
    )
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="Chi xu ly phan thu I trong N phan (vd 0/2). De chay nhieu GPU.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    videos_dir = Path(args.videos or cfg.paths.videos)
    out_dir = Path(args.out or cfg.paths.shots)
    raw_dir = Path(cfg.paths.shots_raw)
    threshold = args.threshold if args.threshold is not None else cfg.shot_detection.threshold
    device = args.device or cfg.runtime.device
    save_raw = bool(cfg.shot_detection.save_raw_predictions)

    videos = select_shard(sd.find_videos(videos_dir, cfg.shot_detection.video_ext), args.shard)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"Khong tim thay video nao trong {videos_dir}", file=sys.stderr)
        return 1

    if args.rethreshold is not None:
        return rethreshold_all(videos, out_dir, raw_dir, args.rethreshold)

    todo = videos if args.overwrite else [v for v in videos if not (out_dir / f"{v.stem}.json").exists()]
    print(f"{len(videos)} video, {len(todo)} can xu ly (threshold={threshold}, device={device})")
    if not todo:
        return 0

    # Kiem tra ca me TRUOC khi nap model: chi doc 16 byte moi file nen gan nhu
    # tuc thi, va bao het file hong mot lan thay vi lo dan tung cai sau moi lan
    # cho model chay. Voi bo du lieu ca nghin video thi khac biet la rat lon.
    todo, broken = preflight(todo)
    if broken:
        print(f"\n{len(broken)}/{len(broken) + len(todo)} video KHONG DOC DUOC:", file=sys.stderr)
        for _, err in broken:
            print(f"  - {err}", file=sys.stderr)   # err da chua ten file
        print("Tai lai nhung file nay roi chay lai.\n", file=sys.stderr)
    if not todo:
        print("Khong con video nao hop le de xu ly.", file=sys.stderr)
        return 1
    if args.check_only:
        print(f"{len(todo)} video doc duoc.")
        return 1 if broken else 0

    print("Dang load TransNetV2...")
    model = sd.build_model(device)
    print(f"Model san sang tren device: {getattr(model, 'device', '?')}")

    failed: list[tuple[str, str]] = []
    t0 = time.time()
    for i, video in enumerate(todo, 1):
        try:
            t1 = time.time()
            record, predictions = sd.detect_shots(model, video, threshold)
            sd.write_shots(out_dir / f"{video.stem}.json", record)
            if save_raw:
                sd.save_raw(raw_dir / f"{video.stem}.npy", predictions)
            print(
                f"[{i}/{len(todo)}] {video.name}: {record['n_shots']} shot, "
                f"{record['n_frames']} frame, fps={record['fps']:.3f}, {time.time() - t1:.1f}s"
            )
        except Exception as exc:  # mot video hong khong duoc lam chet ca me
            failed.append((video.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(todo)}] LOI {video.name}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\nXong {len(todo) - len(failed)}/{len(todo)} video trong {time.time() - t0:.1f}s")
    if failed:
        print(f"{len(failed)} video LOI:", file=sys.stderr)
        for name, err in failed:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    return 0


def preflight(videos: list[Path]) -> tuple[list[Path], list[tuple[str, str]]]:
    """Tach danh sach thanh (doc duoc, hong). Khong nap model, khong goi ffmpeg."""
    ok: list[Path] = []
    broken: list[tuple[str, str]] = []
    for video in videos:
        try:
            sd.check_video_readable(video)
            ok.append(video)
        except (FileNotFoundError, ValueError) as exc:
            broken.append((video.name, str(exc)))
    return ok, broken


def rethreshold_all(videos: list[Path], out_dir: Path, raw_dir: Path, threshold: float) -> int:
    """Dung lai .npy: doi threshold ma khong cham vao GPU."""
    n = 0
    for video in videos:
        raw_path = raw_dir / f"{video.stem}.npy"
        json_path = out_dir / f"{video.stem}.json"
        if not raw_path.exists():
            print(f"Bo qua {video.stem}: khong co {raw_path.name}", file=sys.stderr)
            continue
        fps = sd.read_shots(json_path)["fps"] if json_path.exists() else 25.0
        if not json_path.exists():
            print(f"  ! {video.stem}: chua co JSON, dung fps=25.0", file=sys.stderr)
        record = sd.rethreshold(raw_path, video, fps, threshold)
        sd.write_shots(json_path, record)
        print(f"{video.stem}: {record['n_shots']} shot @ threshold={threshold}")
        n += 1
    print(f"\nDa tao lai {n} file JSON voi threshold={threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
