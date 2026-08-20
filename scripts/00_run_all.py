"""Orchestrator - chay het pipeline tien xu ly theo dung thu tu.

    python scripts/00_run_all.py --device cuda
    python scripts/00_run_all.py --limit 2 --device cuda     # thu 2 video truoc
    python scripts/00_run_all.py --from 3                    # chay lai tu buoc 3
    python scripts/00_run_all.py --skip-ocr

Thu tu bat buoc:
    1. Shot detection (A.1)          - tuan tu, phai xong truoc
    2. Keyframe extraction (A.2)     - tuan tu, can ket qua buoc 1
    3. build-manifest (A.2)          - GOP, phai chay sau khi MOI video xong buoc 2
    4. Encode SigLIP2 (A.3)          -+
    5. Build 2 FAISS index (A.3)      | nhanh doc lap, nhung 5 can 4 xong
    6. OCR (A.4)                     -+ doc lap hoan toan voi 4-5

Buoc 3 la ranh gioi quan trong: no sinh ra manifest - nguon su that cua bat bien
ID. Chay no khi con video chua xong buoc 2 se cho ra manifest thieu, va moi thu
sau do deu khop voi manifest thieu do ma khong bao loi.

Moi buoc deu resume duoc, nen dut giua chung thi chay lai lenh nay la du.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aic.console import use_utf8

use_utf8()

STEPS = [
    (1, "Shot detection (A.1)", ["01_shot_detect.py"], True),
    (2, "Keyframe extraction (A.2)", ["02_keyframe.py"], True),
    (3, "Build manifest (A.2)", ["02_keyframe.py", "--build-manifest"], False),
    (4, "Encode SigLIP2 (A.3)", ["03_build_index.py", "--encode"], True),
    (5, "Build FAISS index (A.3)", ["03_build_index.py", "--build"], False),
    (6, "OCR (A.4)", ["04_ocr.py"], True),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chay het pipeline tien xu ly A.1 -> A.4")
    p.add_argument("--config", default=None)
    p.add_argument("--device", default=None, help="auto | cuda | cpu")
    p.add_argument("--limit", type=int, default=None, help="Chi xu ly N video dau (de thu)")
    p.add_argument("--from", dest="start", type=int, default=1, help="Bat dau tu buoc thu may")
    p.add_argument("--to", dest="end", type=int, default=len(STEPS), help="Dung o buoc thu may")
    p.add_argument("--skip-ocr", action="store_true", help="Bo qua buoc 6")
    p.add_argument("--dry-run", action="store_true", help="Chi in lenh se chay")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t_all = time.time()

    for number, name, command, per_video in STEPS:
        if number < args.start or number > args.end:
            continue
        if number == 6 and args.skip_ocr:
            print(f"\n### Buoc {number}: {name} - BO QUA (--skip-ocr)")
            continue

        argv = [sys.executable, str(ROOT / "scripts" / command[0]), *command[1:]]
        if args.config:
            argv += ["--config", args.config]
        # --device: buoc 6 dung --cpu chu khong nhan --device
        if args.device and command[0] != "04_ocr.py":
            argv += ["--device", args.device]
        if args.device == "cpu" and command[0] == "04_ocr.py":
            argv += ["--cpu"]
        # --limit chi co nghia voi cac buoc chay theo tung video
        if args.limit and per_video:
            argv += ["--limit", str(args.limit)]

        print(f"\n{'=' * 70}\n### Buoc {number}/{len(STEPS)}: {name}\n### {' '.join(argv[1:])}\n{'=' * 70}")
        if args.dry_run:
            continue

        t0 = time.time()
        code = subprocess.call(argv, cwd=str(ROOT))
        elapsed = time.time() - t0
        if code != 0:
            print(
                f"\nBuoc {number} ({name}) that bai voi exit code {code} sau {elapsed:.1f}s. "
                f"Sua xong thi chay lai:\n"
                f"  python scripts/00_run_all.py --from {number}",
                file=sys.stderr,
            )
            return code
        print(f"### Buoc {number} xong trong {elapsed:.1f}s")

    print(f"\n{'=' * 70}\nToan bo pipeline xong trong {time.time() - t_all:.1f}s")
    print("Thu search:  python scripts/run_search.py \"cau truy van cua ban\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
