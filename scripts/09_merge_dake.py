"""Gộp keyframe DAKE vào bộ keyframe baseline (hợp của hai tập frame_idx).

    python scripts/09_merge_dake.py --dry-run
    python scripts/09_merge_dake.py
    python scripts/09_merge_dake.py --restore      # lùi về bản baseline

Hai bộ dùng CÙNG hệ frame_idx (số thứ tự frame khi đọc tuần tự) nên hợp nhất là
phép hợp tập hợp thuần tuý: cùng frame_idx tức cùng một khung hình của cùng một
video, ảnh giống hệt nhau.

Thao tác CỘNG THÊM và ĐẢO NGƯỢC ĐƯỢC:
  - Chỉ chép ảnh DAKE chưa có, không ghi đè ảnh baseline.
  - keyframes.json cũ được lưu thành keyframes.json.baseline trước khi ghi đè,
    nên `--restore` trả về nguyên trạng.

⚠️ Sau khi gộp, clip.npy và siglip2.npy của các video này KHÔNG còn khớp: chúng
có số hàng của tập cũ. Script đổi tên chúng thành *.stale để bước build index
báo thiếu file ngay thay vì âm thầm dùng vector sai. Phải encode lại trên GPU
(xem kaggle/run_dake_encode.py), rồi OCR lại, rồi build lại manifest và 2 index.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

from aic.config import load_config
from aic.preprocess.indexing import SIGLIP_EMB
from aic.preprocess.keyframe import KEYFRAME_EMB, KEYFRAME_META, KEYFRAME_META_VERSION

BACKUP_SUFFIX = ".baseline"
STALE_SUFFIX = ".stale"
DAKE_DIRNAME = "keyframes_dake"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gộp keyframe DAKE vào baseline")
    p.add_argument("--config", default=None)
    p.add_argument("--dake", default=None, help="Mặc định <data_root>/keyframes_dake")
    p.add_argument("--dry-run", action="store_true", help="Chỉ in kế hoạch")
    p.add_argument("--restore", action="store_true", help="Lùi về keyframes.json.baseline")
    return p.parse_args()


def merge_one(base_dir: Path, dake_dir: Path, dry_run: bool) -> dict:
    """Gộp một video. Trả về thống kê."""
    dake_meta = json.loads((dake_dir / KEYFRAME_META).read_text(encoding="utf-8"))
    base_path = base_dir / KEYFRAME_META
    base_meta = json.loads(base_path.read_text(encoding="utf-8"))
    video_id = base_meta["video_id"]

    by_frame = {k["frame_idx"]: dict(k) for k in base_meta["keyframes"]}
    for k in base_meta["keyframes"]:
        by_frame[k["frame_idx"]].setdefault("source", "baseline")

    n_base = len(by_frame)
    new_frames = []
    for k in dake_meta["keyframes"]:
        idx = k["frame_idx"]
        if idx in by_frame:
            by_frame[idx]["source"] = "both"
            continue
        new_frames.append(idx)
        by_frame[idx] = {
            "frame_idx": idx,
            "shot_id": None,                 # DAKE không tách shot
            "pts_time": k["pts_time"],
            "path": f"{video_id}/{idx}.jpg",
            "source": "dake",
            "trigger": k.get("trigger"),
        }

    # Sắp theo frame_idx: thứ tự thời gian, và là thứ tự manifest sẽ dùng.
    merged = [by_frame[i] for i in sorted(by_frame)]

    if not dry_run:
        for idx in new_frames:
            src = dake_dir / f"{idx}.jpg"
            dst = base_dir / f"{idx}.jpg"
            if not src.exists():
                raise FileNotFoundError(f"{video_id}: thiếu ảnh DAKE {src}")
            if not dst.exists():
                shutil.copy2(src, dst)

        backup = base_path.with_name(base_path.name + BACKUP_SUFFIX)
        if not backup.exists():
            shutil.copy2(base_path, backup)

        out = dict(base_meta)
        out.update({
            "version": KEYFRAME_META_VERSION,
            "n_keyframes": len(merged),
            "keyframes": merged,
            "method": "baseline+dake",
            "n_baseline": n_base,
            "n_dake_added": len(new_frames),
            "dake_params": dake_meta.get("params"),
        })
        base_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        # Vector cũ không còn khớp số hàng -> đổi tên để build index báo thiếu ngay.
        for name in (KEYFRAME_EMB, SIGLIP_EMB):
            emb = base_dir / name
            if emb.exists():
                emb.rename(emb.with_name(name + STALE_SUFFIX))

    return {"video_id": video_id, "base": n_base, "dake": len(dake_meta["keyframes"]),
            "added": len(new_frames), "merged": len(merged)}


def restore(base_root: Path) -> int:
    """Trả keyframes.json về bản baseline và khôi phục vector .stale.

    Ảnh DAKE đã chép vào thì để nguyên - chúng không nằm trong keyframes.json
    nên không lọt vào manifest, chỉ chiếm chỗ.
    """
    n = 0
    for backup in sorted(base_root.glob(f"*/{KEYFRAME_META}{BACKUP_SUFFIX}")):
        target = backup.with_name(KEYFRAME_META)
        shutil.copy2(backup, target)
        backup.unlink()
        for name in (KEYFRAME_EMB, SIGLIP_EMB):
            stale = backup.parent / (name + STALE_SUFFIX)
            if stale.exists():
                stale.rename(backup.parent / name)
        n += 1
    print(f"Đã lùi {n} video về bản baseline (ảnh DAKE đã chép vẫn nằm trên đĩa, vô hại)")
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    base_root = Path(cfg.paths.keyframes)

    if args.restore:
        return restore(base_root)

    dake_root = Path(args.dake) if args.dake else Path(cfg.paths.data_root) / DAKE_DIRNAME
    dake_videos = sorted(p for p in dake_root.iterdir() if (p / KEYFRAME_META).exists())
    if not dake_videos:
        print(f"Không có kết quả DAKE trong {dake_root}", file=sys.stderr)
        return 1

    missing = [p.name for p in dake_videos if not (base_root / p.name / KEYFRAME_META).exists()]
    if missing:
        print(f"{len(missing)} video có DAKE nhưng không có baseline: {missing[:5]}", file=sys.stderr)
        return 1

    already = [p for p in dake_videos
               if (base_root / p.name / f"{KEYFRAME_META}{BACKUP_SUFFIX}").exists()]
    if already and not args.dry_run:
        print(f"{len(already)} video đã gộp rồi (có {KEYFRAME_META}{BACKUP_SUFFIX}). "
              f"Chạy --restore trước nếu muốn gộp lại.", file=sys.stderr)
        return 1

    stats = [merge_one(base_root / p.name, p, args.dry_run) for p in dake_videos]

    print(f"{'video':<14}{'baseline':>10}{'DAKE':>8}{'thêm':>8}{'gộp':>9}")
    print("-" * 50)
    for s in stats[:10]:
        print(f"{s['video_id']:<14}{s['base']:>10,}{s['dake']:>8,}{s['added']:>8,}{s['merged']:>9,}")
    if len(stats) > 10:
        print(f"  ... và {len(stats) - 10} video nữa")
    print("-" * 50)
    tot = {k: sum(s[k] for s in stats) for k in ("base", "dake", "added", "merged")}
    print(f"{'TỔNG':<14}{tot['base']:>10,}{tot['dake']:>8,}{tot['added']:>8,}{tot['merged']:>9,}")

    if args.dry_run:
        print("\n(--dry-run: chưa chép ảnh, chưa sửa keyframes.json)")
        return 0

    print(f"\nĐã chép {tot['added']:,} ảnh và cập nhật {len(stats)} keyframes.json")
    print(f"clip.npy/siglip2.npy của {len(stats)} video đã đổi tên thành *{STALE_SUFFIX}")
    print("\nBước tiếp theo (cần GPU):")
    print("  1. Đóng gói L25 gửi lên Kaggle")
    print("  2. kaggle/run_dake_encode.py  -> encode CLIP + SigLIP2 + OCR")
    print("  3. Về máy gộp: 02_keyframe.py --build-manifest && 03_build_index.py --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
