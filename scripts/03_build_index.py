"""A.3 Indexing - encode SigLIP2 + build 2 FAISS index.

Chay tren vast.ai, hai buoc:

    python scripts/03_build_index.py --encode          # encode SigLIP2 tung video (resume duoc)
    python scripts/03_build_index.py --build           # build 2 FAISS index tu manifest

Hoac ca hai:

    python scripts/03_build_index.py --encode --build --device cuda

Buoc --build KHONG encode lai CLIP: no doc thang clip_embeddings.npy ma A.2 da
ghi ra. Do la ly do A.2 phai giu embedding lai.
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

import numpy as np

from aic.config import load_config
from aic.sharding import select_shard
from aic.preprocess import indexing
from aic.preprocess.keyframe import KEYFRAME_META


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A.3 Indexing (SigLIP2 + 2 FAISS index)")
    p.add_argument("--config", default=None)
    p.add_argument("--keyframes", default=None, help="Ghi de paths.keyframes")
    p.add_argument("--device", default=None, help="auto | cuda | cpu")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--encode", action="store_true", help="Encode SigLIP2 cho tung video")
    p.add_argument("--encode-clip", action="store_true",
                   help="Encode LAI CLIP tu anh keyframe (chi dung khi tap keyframe da doi sau A.2)")
    p.add_argument("--build", action="store_true", help="Build 2 FAISS index tu manifest")
    p.add_argument("--overwrite", action="store_true", help="Encode lai ca video da co siglip2.npy")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", default=None, metavar="I/N",
                   help="Chi xu ly phan thu I trong N phan (vd 0/2). De chay nhieu GPU.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.encode and not args.build and not args.encode_clip:
        print("Can it nhat mot trong --encode / --encode-clip / --build", file=sys.stderr)
        return 2

    cfg = load_config(args.config) if args.config else load_config()
    keyframes_dir = Path(args.keyframes or cfg.paths.keyframes)

    if args.encode_clip:
        code = run_encode(cfg, keyframes_dir, args, model="clip")
        if code != 0:
            return code
    if args.encode:
        code = run_encode(cfg, keyframes_dir, args, model="siglip2")
        if code != 0:
            return code
    if args.build:
        return run_build(cfg, keyframes_dir)
    return 0


def run_encode(cfg, keyframes_dir: Path, args, model: str = "siglip2") -> int:
    video_ids = select_shard(
        sorted(p.parent.name for p in keyframes_dir.glob(f"*/{KEYFRAME_META}")), args.shard
    )
    if args.limit:
        video_ids = video_ids[: args.limit]
    if not video_ids:
        print(f"Khong tim thay keyframe nao trong {keyframes_dir}. Chay A.2 truoc.", file=sys.stderr)
        return 1

    from aic.preprocess.keyframe import KEYFRAME_EMB

    out_name = indexing.SIGLIP_EMB if model == "siglip2" else KEYFRAME_EMB
    todo = (
        video_ids
        if args.overwrite
        else [v for v in video_ids if not (keyframes_dir / v / out_name).exists()]
    )
    device = args.device or cfg.runtime.device
    batch_size = args.batch_size or cfg.keyframe.batch_size
    print(f"{len(video_ids)} video, {len(todo)} can encode {model} -> {out_name} (device={device})")
    if not todo:
        return 0

    spec = cfg.models.siglip2 if model == "siglip2" else cfg.models.clip
    print(f"Dang load {model} {spec.name} ({spec.pretrained})...")
    if model == "siglip2":
        from aic.models.siglip_encoder import SiglipEncoder as Encoder
    else:
        from aic.models.clip_encoder import ClipEncoder as Encoder

    encoder = Encoder(device=device, name=spec.name, pretrained=spec.pretrained)
    print(f"Model san sang tren {encoder.device}, dim={encoder.dim}")
    if encoder.dim != spec.dim:
        print(
            f"  ! dim thuc te {encoder.dim} khac config ({spec.dim}) "
            "- cap nhat configs/default.yaml",
            file=sys.stderr,
        )

    failed: list[tuple[str, str]] = []
    total = 0
    t0 = time.time()
    for i, video_id in enumerate(todo, 1):
        try:
            t1 = time.time()
            n = indexing.encode_video_images(
                encoder, keyframes_dir, video_id, out_name, batch_size=batch_size
            )
            total += n
            print(f"[{i}/{len(todo)}] {video_id}: {n} vector, {time.time() - t1:.1f}s")
        except Exception as exc:
            failed.append((video_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(todo)}] LOI {video_id}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\nEncode xong {len(todo) - len(failed)}/{len(todo)} video, "
          f"{total} vector, {time.time() - t0:.1f}s")
    if failed:
        print(f"{len(failed)} video LOI:", file=sys.stderr)
        for name, err in failed:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    return 0


def run_build(cfg, keyframes_dir: Path) -> int:
    print("Build FAISS index tu manifest...")
    t0 = time.time()
    result = indexing.build_indexes(
        cfg.paths.manifest,
        keyframes_dir,
        cfg.paths.clip_embeddings,
        cfg.paths.faiss_clip,
        cfg.paths.faiss_siglip,
        cfg.paths.index_meta,
    )
    n = result["n_manifest"]
    for name, path, dim in (
        ("clip", cfg.paths.faiss_clip, result["dim"]["clip"]),
        ("siglip2", cfg.paths.faiss_siglip, result["dim"]["siglip2"]),
    ):
        print(
            f"  {name:<8} ntotal={result['ntotal'][name]}  dim={dim}  "
            f"RAM~{n * dim * 4 / 1e9:.2f}GB  ->  {path}"
        )
    print(f"  manifest {n} dong, meta -> {cfg.paths.index_meta}")
    print(f"Xong trong {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
