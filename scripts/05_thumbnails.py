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

KHONG can manifest: neu chua co manifest.csv thi doc thang keyframes.json cua
tung video. Thumbnail dat ten theo duong dan anh chu khong theo idx nen khong
dinh gi toi bat bien ID.
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


def list_keyframe_paths(manifest: Path, keyframes_dir: Path) -> tuple[list[str], str]:
    """Danh sach duong dan anh keyframe (tuong doi so voi keyframes_dir).

    Uu tien manifest neu co. KHONG bat buoc phai co: thumbnail duoc dat ten theo
    DUONG DAN anh chu khong theo idx, nen no khong dinh gi toi bat bien ID va
    chay duoc truoc ca buoc --build-manifest. Nho vay moi thanh vien sinh
    thumbnail cho phan cua minh duoc ngay, khong phai doi may gop.
    """
    if manifest.exists():
        return [entry.path for entry in iter_manifest(manifest)], f"manifest ({manifest.name})"

    from aic.preprocess.keyframe import KEYFRAME_META, read_keyframe_meta

    paths: list[str] = []
    for meta_path in sorted(keyframes_dir.glob(f"*/{KEYFRAME_META}")):
        paths.extend(kf["path"] for kf in read_keyframe_meta(meta_path)["keyframes"])
    return paths, f"keyframes.json cua {len(list(keyframes_dir.glob('*/' + KEYFRAME_META)))} video (chua co manifest)"


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

    rel_paths, source = list_keyframe_paths(Path(cfg.paths.manifest), keyframes)
    if not rel_paths:
        print(f"Khong tim thay keyframe nao trong {keyframes}. Chay A.2 truoc.", file=sys.stderr)
        return 1
    print(f"Nguon danh sach keyframe: {source}")

    jobs: list[tuple[Path, Path]] = []
    for rel in rel_paths:
        dst = thumbs / rel
        if args.overwrite or not dst.exists():
            jobs.append((keyframes / rel, dst))
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
