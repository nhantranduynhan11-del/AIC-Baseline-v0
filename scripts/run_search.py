"""B.1 - search hoan chinh: encode 2 model -> FAISS x2 -> RRF -> [OCR filter] -> top-100.

    python scripts/run_search.py "nguoi dan ong mac ao do dang chay"
    python scripts/run_search.py "bien bao giao thong" --ocr "dien bien phu"
    python scripts/run_search.py --image path/to/frame.jpg --show 20
    python scripts/run_search.py "..." --per-model      # xem them ranked list tung model

Xuat file nop bai:
    python scripts/run_search.py "..." --export sub.csv                    # KIS
    python scripts/run_search.py "..." --export qa.csv --task qa --answer "Mau do"

--ocr la OPTIONAL theo tung query: chi ap khi cau truy van thuc su co yeu cau ve
chu tren man hinh. Khong truyen thi khong loc gi.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aic.console import use_utf8

use_utf8()

from aic.config import load_config
from aic.retrieval import pipeline
from aic.retrieval.filters import apply_filter
from aic.retrieval.fusion import reciprocal_rank_fusion
from aic.retrieval.search import IndexBundle, index_files_ready


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B.1 - search (RRF, top-100)")
    p.add_argument("query", nargs="?", default=None, help="Cau truy van text")
    p.add_argument("--image", default=None, help="Anh query (Video KIS) thay cho text")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--k", type=int, default=None, help="Ghi de retrieval.top_k_per_model")
    p.add_argument("--rrf-k", type=int, default=None, help="Ghi de retrieval.rrf_k")
    p.add_argument("--top-n", type=int, default=None, help="Ghi de retrieval.final_top_n")
    p.add_argument("--ocr", default=None, help="Cum chu phai co tren man hinh (hard filter)")
    p.add_argument("--ocr-no-phrase", action="store_true",
                   help="Chu OCR chi can du tu, khong can lien nhau")
    p.add_argument("--ocr-min-conf", type=float, default=None,
                   help="Ghi de ocr.query_min_confidence")
    p.add_argument("--show", type=int, default=20, help="So dong in ra")
    p.add_argument("--per-model", action="store_true", help="In them ranked list tung model")
    p.add_argument("--export", default=None, help="Ghi ket qua ra file CSV nop bai")
    p.add_argument("--task", default="kis", choices=["kis", "qa"],
                   help="Loai task khi export. TRAKE can nhieu query nen dung API rieng.")
    p.add_argument("--answer", default=None, help="Answer cho task qa (toi da 100 ky tu)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if (args.query is None) == (args.image is None):
        print("Truyen dung mot trong hai: cau truy van text hoac --image", file=sys.stderr)
        return 2

    cfg = load_config(args.config) if args.config else load_config()
    if not index_files_ready(cfg):
        print("Chua co manifest / FAISS index. Chay A.2 va A.3 truoc.", file=sys.stderr)
        return 1

    k = args.k or cfg.retrieval.top_k_per_model
    rrf_k = args.rrf_k or cfg.retrieval.rrf_k
    top_n = args.top_n or cfg.retrieval.final_top_n

    bundle = IndexBundle.from_config(cfg)
    print(f"Index: {bundle.ntotal} keyframe, dim={bundle.dims()}")

    print("Dang load 2 model...")
    from aic.retrieval.encode_query import QueryEncoder

    encoder = QueryEncoder(cfg, device=args.device)

    image_rgb = None
    if args.image is not None:
        import cv2

        img = cv2.imread(args.image)
        if img is None:
            print(f"Khong doc duoc anh: {args.image}", file=sys.stderr)
            return 1
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"\nQuery: anh {args.image}")
    else:
        print(f'\nQuery: "{args.query}"')

    allowed = None
    if args.ocr:
        from aic.retrieval.filters import describe, ocr_allowed_idxs
        from aic.store import sqlite_store as store

        conn = store.open_db(cfg.paths.metadata_db)
        min_conf = args.ocr_min_conf if args.ocr_min_conf is not None else cfg.ocr.query_min_confidence
        result = ocr_allowed_idxs(
            conn, bundle, args.ocr,
            phrase=not args.ocr_no_phrase, min_confidence=min_conf,
        )
        allowed = result.allowed
        print(f'OCR filter "{args.ocr}" (conf>={min_conf}): {describe(result)}')
        if not result:
            print("Khong keyframe nao thoa dieu kien OCR -> ket qua se rong.", file=sys.stderr)

    t0 = time.time()
    vectors = encoder.encode(text=args.query, image_rgb=image_rgb)
    per_model = bundle.search(vectors, k)
    # RRF tren TOAN BO ung vien -> loc -> moi cat top_n. Cat truoc khi loc se ra
    # it hon top_n mot cach vo ly.
    hits = apply_filter(reciprocal_rank_fusion(per_model, k=rrf_k, top_n=None), allowed)[:top_n]
    elapsed = time.time() - t0

    if args.per_model:
        for model_key, model_hits in per_model.items():
            print(f"\n--- {model_key}: {len(model_hits)} ket qua, {args.show} dong dau ---")
            for row in bundle.hydrate(model_hits[: args.show]):
                print(f"  #{row['rank']:<3} {row['score']:.4f}  {row['video_id']}  "
                      f"frame={row['frame_idx']}")

    label = f"RRF (k={rrf_k})" + (" + OCR filter" if allowed is not None else "")
    print(f"\n=== {label}, top-{top_n}, in {min(args.show, len(hits))} dong dau ===")
    for row in pipeline.hydrate(bundle, hits[: args.show]):
        sources = " ".join(f"{name}#{rank}" for name, rank in sorted(row["ranks"].items()))
        print(
            f"  #{row['rank']:<3} {row['score']:.6f}  {row['video_id']}  "
            f"frame={row['frame_idx']:<7} t={row['pts_time']:.2f}s  [{sources}]"
        )

    if args.export:
        code = do_export(bundle, hits, args)
        if code != 0:
            return code

    both = sum(1 for h in hits if len(h.ranks) == len(per_model))
    print(f"\n{len(hits)} ket qua, {both} xuat hien o ca {len(per_model)} model")
    print(f"Encode + search + fusion: {elapsed:.2f}s (K={k})")
    return 0


def do_export(bundle, hits, args) -> int:
    from aic.submit import export

    if args.task == "qa" and not args.answer:
        print("Task qa can --answer", file=sys.stderr)
        return 2
    if not hits:
        print("Khong co ket qua nao de xuat.", file=sys.stderr)
        return 1

    rows = pipeline.hydrate(bundle, hits)
    if args.task == "kis":
        n = export.write_kis(args.export, export.hits_to_kis(rows))
    else:
        n = export.write_qa(args.export, export.hits_to_qa(rows, args.answer))

    info = export.validate_file(args.export, args.task)
    print(f"\nDa ghi {n} dong ra {args.export} (task={args.task}), doc lai kiem tra OK: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
