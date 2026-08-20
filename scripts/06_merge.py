"""Gộp kết quả tiền xử lý từ nhiều máy, và kiểm tra đã đủ chưa.

Chia việc cho nhóm: mỗi người bỏ phần video của mình vào `data/videos/` rồi chạy
`00_run_all.py` như bình thường. Mọi sản phẩm của A.1-A.3 đều theo TỪNG VIDEO
nên chỉ cần copy thư mục lại là xong. Riêng DB OCR là một file dùng chung nên
phải gộp bằng script này.

    # 1. Gộp DB OCR của các thành viên
    python scripts/06_merge.py --merge-db /mnt/share/an.db /mnt/share/binh.db

    # 2. Kiểm tra đã đủ mọi thứ chưa TRƯỚC khi build manifest
    python scripts/06_merge.py --check

    # 3. Chỉ khi bước 2 báo sạch mới chạy:
    python scripts/02_keyframe.py --build-manifest
    python scripts/03_build_index.py --build

⚠️ Bước --check quan trọng vì `--build-manifest` KHÔNG biết là nó đang thiếu dữ
liệu: thiếu video nào thì manifest chỉ đơn giản là không có video đó, và mọi thứ
sau đó khớp với manifest thiếu ấy mà không báo lỗi gì.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

from aic.config import load_config
from aic.preprocess.indexing import SIGLIP_EMB
from aic.preprocess.keyframe import KEYFRAME_EMB, KEYFRAME_META
from aic.store import sqlite_store as store


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gộp kết quả từ nhiều máy")
    p.add_argument("--config", default=None)
    p.add_argument("--merge-db", nargs="+", metavar="DB", default=None,
                   help="Các file metadata.db cần gộp vào DB chính")
    p.add_argument("--check", action="store_true",
                   help="Kiểm tra từng video đã đủ sản phẩm của A.1-A.4 chưa")
    p.add_argument("--overwrite", action="store_true",
                   help="Khi gộp DB: ghi đè video đã có thay vì bỏ qua")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.merge_db and not args.check:
        print("Cần ít nhất một trong --merge-db / --check", file=sys.stderr)
        return 2

    cfg = load_config(args.config) if args.config else load_config()

    if args.merge_db:
        code = merge_databases(cfg, [Path(p) for p in args.merge_db], args.overwrite)
        if code != 0:
            return code
    if args.check:
        return check_coverage(cfg)
    return 0


def merge_databases(cfg, sources: list[Path], overwrite: bool) -> int:
    """Gộp bảng ocr + ocr_done từ các DB khác vào DB chính.

    Bỏ qua video đã có trong DB đích (theo ocr_done) để chạy lại lệnh này nhiều
    lần không sinh dòng trùng. `--overwrite` thì xoá bản cũ trước khi chép.
    """
    target = store.open_db(cfg.paths.metadata_db)
    print(f"DB đích: {cfg.paths.metadata_db}")
    print(f"  trước khi gộp: {store.stats(target)}")

    total_rows = 0
    for src in sources:
        if not src.exists():
            print(f"  ! không có {src}", file=sys.stderr)
            return 1

        src_videos = video_ids_in(src)
        have = store.done_videos(target)
        incoming = [v for v in src_videos if overwrite or v not in have]
        skipped = len(src_videos) - len(incoming)

        if not incoming:
            print(f"  {src.name}: {len(src_videos)} video, đã có hết -> bỏ qua")
            continue

        if overwrite:
            for video_id in incoming:
                store.delete_video(target, video_id)

        # ATTACH rồi INSERT ... SELECT: trigger FTS5 của bảng đích tự chạy nên
        # chỉ mục full-text được cập nhật đúng, không phải rebuild.
        target.execute("ATTACH DATABASE ? AS src", (str(src),))
        try:
            placeholders = ",".join("?" * len(incoming))
            cur = target.execute(
                f"INSERT INTO ocr(video_id, frame_idx, text, text_norm, confidence) "
                f"SELECT video_id, frame_idx, text, text_norm, confidence FROM src.ocr "
                f"WHERE video_id IN ({placeholders})",
                incoming,
            )
            rows = cur.rowcount
            target.execute(
                f"INSERT OR REPLACE INTO ocr_done(video_id, n_rows, engine, created_at) "
                f"SELECT video_id, n_rows, engine, created_at FROM src.ocr_done "
                f"WHERE video_id IN ({placeholders})",
                incoming,
            )
            target.commit()
        finally:
            target.execute("DETACH DATABASE src")

        total_rows += rows
        note = f", bỏ qua {skipped} video đã có" if skipped else ""
        print(f"  {src.name}: +{len(incoming)} video, +{rows} dòng OCR{note}")

    print(f"  sau khi gộp:   {store.stats(target)}  (+{total_rows} dòng)")
    return 0


def video_ids_in(db_path: Path) -> list[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return [r[0] for r in conn.execute("SELECT video_id FROM ocr_done ORDER BY video_id")]
    finally:
        conn.close()


def check_coverage(cfg) -> int:
    """Đối chiếu từng video xem đã đủ sản phẩm của mọi bước chưa."""
    shots_dir = Path(cfg.paths.shots)
    kf_dir = Path(cfg.paths.keyframes)

    video_ids = sorted(
        {p.stem for p in shots_dir.glob("*.json")}
        | {p.parent.name for p in kf_dir.glob(f"*/{KEYFRAME_META}")}
    )
    if not video_ids:
        print("Không tìm thấy kết quả nào. Chưa chạy A.1?", file=sys.stderr)
        return 1

    try:
        ocr_done = store.done_videos(store.open_db(cfg.paths.metadata_db))
    except Exception:
        ocr_done = set()

    columns = ("A.1 shot", "A.2 keyframe", "A.2 clip.npy", "A.3 siglip2", "A.4 OCR")
    missing: dict[str, list[str]] = {c: [] for c in columns}

    for video_id in video_ids:
        checks = {
            "A.1 shot": (shots_dir / f"{video_id}.json").exists(),
            "A.2 keyframe": (kf_dir / video_id / KEYFRAME_META).exists(),
            "A.2 clip.npy": (kf_dir / video_id / KEYFRAME_EMB).exists(),
            "A.3 siglip2": (kf_dir / video_id / SIGLIP_EMB).exists(),
            "A.4 OCR": video_id in ocr_done,
        }
        for name, ok in checks.items():
            if not ok:
                missing[name].append(video_id)

    print(f"{len(video_ids)} video trong data/\n")
    print(f"  {'bước':<16} {'đủ':>6} {'thiếu':>7}")
    for name in columns:
        n_missing = len(missing[name])
        mark = "" if n_missing == 0 else "   <-- THIẾU"
        print(f"  {name:<16} {len(video_ids) - n_missing:>6} {n_missing:>7}{mark}")

    incomplete = {v for names in missing.values() for v in names}
    if incomplete:
        print(f"\n{len(incomplete)} video chưa xong:")
        for video_id in sorted(incomplete)[:20]:
            lack = [n for n in columns if video_id in missing[n]]
            print(f"  - {video_id}: thiếu {', '.join(lack)}")
        if len(incomplete) > 20:
            print(f"  ... và {len(incomplete) - 20} video nữa")
        print("\nCHƯA được chạy --build-manifest: manifest sẽ thiếu và không có gì báo lỗi.")
        return 1

    print("\nĐủ hết. Chạy được:")
    print("  python scripts/02_keyframe.py --build-manifest")
    print("  python scripts/03_build_index.py --build")
    print("  python scripts/05_thumbnails.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
