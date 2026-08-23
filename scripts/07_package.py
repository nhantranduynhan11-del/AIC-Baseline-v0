"""Đóng gói ảnh keyframe thành nhiều phần vừa sức chuyển đi.

    python scripts/07_package.py --part-gb 8

Vì sao chia ngay lúc tạo chứ không tar rồi mới cắt: bộ ảnh 48 GB thì tar-rồi-cắt
cần 48 GB cho file trung gian CỘNG 48 GB cho các mảnh. Ghi thẳng ra từng phần
chỉ cần đúng một lần dung lượng.

Mỗi phần chứa TRỌN VẸN các video, không cắt ngang một video. Nhờ vậy giải nén
thiếu một phần vẫn cho ra dữ liệu nhất quán: chỉ thiếu hẳn vài video chứ không
có video nào bị mất một nửa số ảnh.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

from aic.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Đóng gói ảnh keyframe thành nhiều phần")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="dist", help="Thư mục chứa gói")
    p.add_argument("--part-gb", type=float, default=8.0, help="Kích thước mỗi phần (GB)")
    p.add_argument("--prefix", default="aic_keyframes_part")
    p.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch chia")
    return p.parse_args()


def plan_parts(videos: list[tuple[str, list[Path], int]], budget: int) -> list[list[tuple]]:
    """Gom video vào các phần sao cho mỗi phần không vượt `budget` byte.

    Video nào tự nó đã lớn hơn budget thì vẫn đứng riêng một phần — thà một phần
    quá cỡ còn hơn cắt đôi một video.
    """
    parts: list[list[tuple]] = []
    current: list[tuple] = []
    current_size = 0

    for video in videos:
        size = video[2]
        if current and current_size + size > budget:
            parts.append(current)
            current, current_size = [], 0
        current.append(video)
        current_size += size

    if current:
        parts.append(current)
    return parts


def collect_videos(keyframes: Path, manifest: Path) -> list[tuple[str, list[Path], int]]:
    """(video_id, danh sách ảnh, tổng byte) cho từng video, theo THỨ TỰ MANIFEST.

    Lấy danh sách từ manifest chứ không quét thư mục: chạy lại A.2 với tham số
    khác sẽ ghi JPEG mới mà không xoá JPEG cũ, nên trong thư mục có thể còn ảnh
    thừa không thuộc index nào. Đóng gói theo manifest thì gói chứa ĐÚNG những
    ảnh mà index tham chiếu, không hơn không kém.
    """
    from aic.manifest import iter_manifest

    grouped: dict[str, list[Path]] = {}
    for entry in iter_manifest(manifest):
        grouped.setdefault(entry.video_id, []).append(keyframes / entry.path)

    out = []
    for video_id, paths in grouped.items():
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"{video_id}: thiếu {len(missing)}/{len(paths)} ảnh, "
                f"ví dụ {missing[0]}"
            )
        out.append((video_id, paths, sum(p.stat().st_size for p in paths)))
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    keyframes = Path(cfg.paths.keyframes)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = collect_videos(keyframes, Path(cfg.paths.manifest))
    if not videos:
        print(f"Không có ảnh nào trong {keyframes}", file=sys.stderr)
        return 1

    total_images = sum(len(v[1]) for v in videos)
    total_bytes = sum(v[2] for v in videos)
    parts = plan_parts(videos, int(args.part_gb * 1e9))

    print(f"{len(videos)} video, {total_images:,} ảnh (theo manifest), {total_bytes/1e9:.2f} GB")
    print(f"-> {len(parts)} phần, mỗi phần tối đa {args.part_gb} GB\n")
    for i, part in enumerate(parts, 1):
        size = sum(v[2] for v in part)
        print(f"  phần {i:02d}: {len(part):>3} video, {sum(len(v[1]) for v in part):>7,} ảnh, "
              f"{size/1e9:>5.2f} GB   {part[0][0]} .. {part[-1][0]}")

    if args.dry_run:
        return 0

    print()
    t_all = time.time()
    for i, part in enumerate(parts, 1):
        path = out_dir / f"{args.prefix}{i:02d}.tar"
        t0 = time.time()
        with tarfile.open(path, "w") as tar:
            for video_id, jpgs, _ in part:
                for jpg in jpgs:
                    tar.add(jpg, arcname=f"keyframes/{video_id}/{jpg.name}")
        print(f"  {path.name}  {path.stat().st_size/1e9:.2f} GB  ({time.time()-t0:.0f}s)")

    print(f"\nXong {len(parts)} phần trong {(time.time()-t_all)/60:.1f} phút -> {out_dir}")
    print("Giải nén ở máy đích:")
    print(f'  for f in {args.prefix}*.tar; do tar xf "$f" -C data; done')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
