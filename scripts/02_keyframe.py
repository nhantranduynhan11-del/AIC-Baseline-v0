"""A.2 Keyframe Extraction - trich keyframe + giu lai embedding CLIP.

Chay tren vast.ai:
    python scripts/02_keyframe.py --device cuda
    python scripts/02_keyframe.py --limit 2            # thu 2 video truoc
    python scripts/02_keyframe.py --build-manifest     # gop thanh manifest.csv (chay sau cung)

Doc data/shots/<video_id>.json cua A.1. Video nao chua co JSON shot thi bo qua.
Mac dinh BO QUA video da co keyframes.json -> chay lai la resume.

`--build-manifest` la buoc RIENG, chay SAU khi moi video da xong: no gop tat ca
lai thanh MOT manifest.csv + MOT clip_embeddings.npy dung thu tu - do chinh la
bat bien ID cua A.6.
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
from aic.preprocess import keyframe as kf
from aic.preprocess import shot_detect as sd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A.2 Keyframe Extraction (CLIP + L2-Norm)")
    p.add_argument("--config", default=None)
    p.add_argument("--videos", default=None, help="Ghi de paths.videos")
    p.add_argument("--shots", default=None, help="Ghi de paths.shots")
    p.add_argument("--out", default=None, help="Ghi de paths.keyframes")
    p.add_argument("--device", default=None, help="auto | cuda | cpu")
    p.add_argument("--sample-every", type=int, default=None)
    p.add_argument("--l2-threshold", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--build-manifest",
        action="store_true",
        help="Chi gop ket qua da co thanh manifest.csv + clip_embeddings.npy",
    )
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="Chi xu ly phan thu I trong N phan (vd 0/2). De chay nhieu GPU.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    keyframes_dir = Path(args.out or cfg.paths.keyframes)

    if args.build_manifest:
        return build_manifest(cfg, keyframes_dir)

    videos_dir = Path(args.videos or cfg.paths.videos)
    shots_dir = Path(args.shots or cfg.paths.shots)
    sample_every = args.sample_every or cfg.keyframe.sample_every
    l2_threshold = args.l2_threshold if args.l2_threshold is not None else cfg.keyframe.l2_threshold
    batch_size = args.batch_size or cfg.keyframe.batch_size
    device = args.device or cfg.runtime.device

    videos = select_shard(sd.find_videos(videos_dir, cfg.shot_detection.video_ext), args.shard)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        print(f"Khong tim thay video nao trong {videos_dir}", file=sys.stderr)
        return 1

    todo, missing_shots = [], []
    for video in videos:
        if not (shots_dir / f"{video.stem}.json").exists():
            missing_shots.append(video.name)
            continue
        if args.overwrite or not (keyframes_dir / video.stem / kf.KEYFRAME_META).exists():
            todo.append(video)

    if missing_shots:
        print(
            f"{len(missing_shots)} video chua co ket qua A.1, bo qua: "
            f"{', '.join(missing_shots[:5])}{' ...' if len(missing_shots) > 5 else ''}",
            file=sys.stderr,
        )
    print(
        f"{len(videos)} video, {len(todo)} can xu ly "
        f"(sample_every={sample_every}, l2={l2_threshold}, batch={batch_size}, device={device})"
    )
    if not todo:
        return 0

    print("Dang load CLIP ViT-L-14-quickgelu (dfn2b)...")
    from aic.models.clip_encoder import ClipEncoder

    encoder = ClipEncoder(device=device)
    print(f"Model san sang tren {encoder.device}, dim={encoder.dim}")

    failed: list[tuple[str, str]] = []
    total_kf = 0
    t0 = time.time()
    for i, video in enumerate(todo, 1):
        try:
            t1 = time.time()
            record = sd.read_shots(shots_dir / f"{video.stem}.json")
            meta = kf.extract_video(
                encoder,
                video,
                record,
                keyframes_dir,
                sample_every=sample_every,
                l2_threshold=l2_threshold,
                batch_size=batch_size,
                jpeg_quality=cfg.keyframe.jpeg_quality,
            )
            total_kf += meta["n_keyframes"]
            ratio = meta["n_keyframes"] / meta["n_sampled"] if meta["n_sampled"] else 0.0
            print(
                f"[{i}/{len(todo)}] {video.name}: {meta['n_keyframes']} keyframe "
                f"/ {meta['n_sampled']} frame lay mau ({ratio:.1%} giu lai), "
                f"{record['n_shots']} shot, {time.time() - t1:.1f}s"
            )
        except Exception as exc:
            failed.append((video.name, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(todo)}] LOI {video.name}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(
        f"\nXong {len(todo) - len(failed)}/{len(todo)} video, "
        f"{total_kf} keyframe, {time.time() - t0:.1f}s"
    )
    print("Buoc tiep theo: python scripts/02_keyframe.py --build-manifest")
    if failed:
        print(f"{len(failed)} video LOI:", file=sys.stderr)
        for name, err in failed:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    return 0


def build_manifest(cfg, keyframes_dir: Path) -> int:
    print(f"Gop keyframe tu {keyframes_dir} ...")
    n, dim = kf.build_manifest(
        keyframes_dir,
        cfg.paths.manifest,
        cfg.paths.clip_embeddings,
        cfg.paths.index_meta,
    )
    if n == 0:
        print("Khong co keyframe nao. Chay A.2 truoc.", file=sys.stderr)
        return 1
    print(f"manifest: {cfg.paths.manifest}  ({n} dong)")
    print(f"embedding CLIP: {cfg.paths.clip_embeddings}  ({n} x {dim} float32, "
          f"{n * dim * 4 / 1e9:.2f} GB)")
    print(f"meta: {cfg.paths.index_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
