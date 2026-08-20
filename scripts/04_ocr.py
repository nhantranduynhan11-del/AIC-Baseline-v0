"""A.4 OCR - chay EasyOCR tren keyframe, ghi vao SQLite FTS5.

Chay tren vast.ai (sau khi A.2 xong):
    python scripts/04_ocr.py --limit 2          # thu 2 video truoc
    python scripts/04_ocr.py

Thu ket qua ngay tren DB, khong can model:
    python scripts/04_ocr.py --query "dien bien phu"
    python scripts/04_ocr.py --query "tin nong" --no-phrase --stats

Mac dinh BO QUA video da co trong bang ocr_done -> chay lai la resume.
Video khong doc duoc chu nao van duoc danh dau xong, nen khong bi chay lai mai.
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
from aic.preprocess import ocr as ocr_mod
from aic.store import sqlite_store as store


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A.4 OCR (EasyOCR -> SQLite FTS5)")
    p.add_argument("--config", default=None)
    p.add_argument("--keyframes", default=None, help="Ghi de paths.keyframes")
    p.add_argument("--db", default=None, help="Ghi de paths.metadata_db")
    p.add_argument("--cpu", action="store_true", help="Ep chay CPU thay vi GPU")
    p.add_argument("--overwrite", action="store_true", help="Chay lai ca video da xong")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--query", default=None, help="Chi tra cuu DB, khong chay OCR")
    p.add_argument("--no-phrase", action="store_true", help="Tim theo tung tu (AND) thay vi ca cum")
    p.add_argument("--stats", action="store_true", help="In thong ke DB roi thoat")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    db_path = args.db or cfg.paths.metadata_db
    conn = store.open_db(db_path)

    if args.stats or args.query:
        if args.stats:
            print(f"DB {db_path}: {store.stats(conn)}")
        if args.query:
            return run_query(conn, cfg, args)
        return 0

    keyframes_dir = Path(args.keyframes or cfg.paths.keyframes)
    video_ids = ocr_mod.video_ids_with_keyframes(keyframes_dir)
    if args.limit:
        video_ids = video_ids[: args.limit]
    if not video_ids:
        print(f"Khong tim thay keyframe nao trong {keyframes_dir}. Chay A.2 truoc.", file=sys.stderr)
        return 1

    done = set() if args.overwrite else store.done_videos(conn)
    todo = [v for v in video_ids if v not in done]
    gpu = not args.cpu
    print(f"{len(video_ids)} video, {len(todo)} can chay OCR (engine={cfg.ocr.engine}, gpu={gpu})")
    if not todo:
        print(f"DB: {store.stats(conn)}")
        return 0

    print(f"Dang load EasyOCR {list(cfg.ocr.languages)}...")
    reader = ocr_mod.build_reader(tuple(cfg.ocr.languages), gpu=gpu)

    failed: list[tuple[str, str]] = []
    total_rows = 0
    t0 = time.time()
    for i, video_id in enumerate(todo, 1):
        try:
            t1 = time.time()
            result = ocr_mod.run_video(
                reader, conn, keyframes_dir, video_id,
                min_confidence=cfg.ocr.min_confidence,
            )
            total_rows += result["n_rows"]
            ratio = (
                result["n_frames_with_text"] / result["n_frames"] if result["n_frames"] else 0.0
            )
            print(
                f"[{i}/{len(todo)}] {video_id}: {result['n_rows']} dong chu tren "
                f"{result['n_frames_with_text']}/{result['n_frames']} keyframe "
                f"({ratio:.1%} co chu), {time.time() - t1:.1f}s"
            )
        except Exception as exc:
            failed.append((video_id, f"{type(exc).__name__}: {exc}"))
            print(f"[{i}/{len(todo)}] LOI {video_id}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

    print(f"\nXong {len(todo) - len(failed)}/{len(todo)} video, {total_rows} dong, "
          f"{time.time() - t0:.1f}s")
    print(f"DB: {store.stats(conn)}")
    if failed:
        print(f"{len(failed)} video LOI:", file=sys.stderr)
        for name, err in failed:
            print(f"  - {name}: {err}", file=sys.stderr)
        return 1
    return 0


def run_query(conn, cfg, args) -> int:
    rows = store.search_text(
        conn, args.query,
        phrase=not args.no_phrase,
        min_confidence=cfg.ocr.query_min_confidence,
        limit=30,
    )
    mode = "tung tu (AND)" if args.no_phrase else "ca cum (phrase)"
    print(f'\n"{args.query}" [{mode}], confidence >= {cfg.ocr.query_min_confidence}: '
          f"{len(rows)} ket qua")
    for row in rows:
        print(f"  {row['bm25']:>8.3f}  conf={row['confidence']:.2f}  "
              f"{row['video_id']} frame={row['frame_idx']:<7} {row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
