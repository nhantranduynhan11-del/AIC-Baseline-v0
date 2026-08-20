"""Sinh thumbnail cho grid UI.

    python scripts/05_thumbnails.py
    python scripts/05_thumbnails.py --size 240 --workers 8

Vi sao can buoc rieng: keyframe duoc luu o do phan giai GOC, JPEG quality 95 -
co y, vi OCR can doc duoc chu nho. Grid UI hien ~60 anh cung luc, serve anh goc
se lam cuon grid giat.

Thumbnail giu nguyen cau truc thu muc va TEN FILE cua keyframe, chi doi thu muc
goc (data/keyframes -> data/thumbs). Nho vay khong dinh dang gi toi bat bien ID:
UI xin /thumb/{idx}, server tra manifest[idx].path trong thu muc thumbs.

Chi dung CPU, chay duoc song song, khong can GPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

from aic.config import load_config
from aic.manifest import iter_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sinh thumbnail cho grid UI")
    p.add_argument("--config", default=None)
    p.add_argument("--size", type=int, default=None, help="Ghi de thumbs.size (canh dai nhat)")
    p.add_argument("--quality", type=int, default=None, help="Ghi de thumbs.quality")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def make_thumb(src: Path, dst: Path, size: int, quality: int) -> bool:
    from PIL import Image

    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        img.save(dst, "JPEG", quality=quality, optimize=True)
    return True


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    keyframes = Path(cfg.paths.keyframes)
    thumbs = Path(cfg.paths.thumbs)
    size = args.size or cfg.thumbs.size
    quality = args.quality or cfg.thumbs.quality
    workers = args.workers or cfg.runtime.num_workers

    manifest = Path(cfg.paths.manifest)
    if not manifest.exists():
        print(f"Chua co {manifest}. Chay A.2 --build-manifest truoc.", file=sys.stderr)
        return 1

    jobs: list[tuple[Path, Path]] = []
    for entry in iter_manifest(manifest):
        dst = thumbs / entry.path
        if args.overwrite or not dst.exists():
            jobs.append((keyframes / entry.path, dst))
        if args.limit and len(jobs) >= args.limit:
            break

    print(f"{len(jobs)} thumbnail can sinh ({size}px, q{quality}, {workers} worker)")
    if not jobs:
        return 0

    failed: list[tuple[str, str]] = []
    done = 0
    t0 = time.time()

    def run(job: tuple[Path, Path]) -> None:
        src, dst = job
        try:
            make_thumb(src, dst, size, quality)
        except Exception as exc:
            failed.append((str(src), f"{type(exc).__name__}: {exc}"))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(run, jobs):
            done += 1
            if done % 2000 == 0:
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(jobs)}  ({rate:.0f} anh/s)")

    print(f"\nXong {len(jobs) - len(failed)}/{len(jobs)} trong {time.time() - t0:.1f}s -> {thumbs}")
    if failed:
        print(f"{len(failed)} anh LOI:", file=sys.stderr)
        for name, err in failed[:10]:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
