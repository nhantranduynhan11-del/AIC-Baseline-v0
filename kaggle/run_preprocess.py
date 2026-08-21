"""Chạy A.1–A.4 trên Kaggle với 2×T4, hai GPU chạy song song.

═══════════════════════════════════════════════════════════════════════════
DÁN Ô NÀY VÀO NOTEBOOK KAGGLE
═══════════════════════════════════════════════════════════════════════════

    !git clone -q https://github.com/nhantranduynhan11-del/AIC-Baseline-v0.git /kaggle/working/repo
    !python /kaggle/working/repo/kaggle/run_preprocess.py --setup
    # (khởi động lại session nếu script yêu cầu, rồi chạy tiếp)
    !python /kaggle/working/repo/kaggle/run_preprocess.py --run

Cài đặt notebook bắt buộc:
  - Accelerator = **GPU T4 ×2**
  - Internet = **On**  (cần để pip install và tải weights model)
  - Dataset video gắn vào phần Input

═══════════════════════════════════════════════════════════════════════════

Vì sao cần script này thay vì gọi thẳng 00_run_all.py:

1. **Hai GPU.** Pipeline chạy một GPU. Script tách video làm 2 phần rời nhau
   (`--shard 0/2`, `--shard 1/2`) rồi chạy hai tiến trình, mỗi tiến trình ghim
   một GPU bằng CUDA_VISIBLE_DEVICES. Không làm vậy thì T4 thứ hai nằm không.

2. **Giới hạn 9 giờ/phiên.** Script dừng có kiểm soát trước hạn và đóng gói
   phần đã làm được, thay vì để Kaggle giết ngang rồi mất sạch.

3. **Đường dẫn Kaggle.** Video ở /kaggle/input (chỉ đọc), kết quả phải nằm ở
   /kaggle/working. Script sinh file config riêng trỏ đúng chỗ.

4. **DB OCR riêng cho từng phần.** Hai tiến trình cùng ghi một file SQLite sẽ
   tranh khoá; mỗi phần ghi DB riêng rồi gộp lại ở cuối bằng 06_merge.py.

CHẠY TIẾP PHIÊN SAU: gắn output của phiên trước vào Input, rồi thêm
`--resume-from /kaggle/input/<tên-dataset>`. Mọi bước đều bỏ qua video đã xong.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from aic.console import use_utf8

use_utf8()

WORK = Path("/kaggle/working")
DATA = WORK / "aic-data"
CONFIG = WORK / "kaggle.yaml"
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

# Kaggle giết phiên GPU ở mốc 9 giờ. Chừa 30 phút để đóng gói kết quả.
DEFAULT_DEADLINE_HOURS = 8.5


# --------------------------------------------------------------------------
# thiết lập
# --------------------------------------------------------------------------


def setup() -> int:
    """Cài dependency. Chạy một lần cho mỗi phiên."""
    packages = [
        "transnetv2-pytorch", "open_clip_torch", "timm",
        "faiss-cpu", "easyocr", "ffmpeg-python", "pyyaml",
    ]
    print("Cài dependency (torch/opencv/transformers đã có sẵn trong image Kaggle)...")
    code = subprocess.call([sys.executable, "-m", "pip", "install", "-q", *packages])
    if code != 0:
        print("pip install thất bại", file=sys.stderr)
        return code

    print("\nKiểm tra:")
    check = subprocess.run(
        [sys.executable, "-c",
         "import torch, faiss, sqlite3;"
         "print(' torch', torch.__version__, '| GPU:', torch.cuda.device_count());"
         "[print('  -', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())];"
         "print(' faiss', faiss.__version__);"
         "c=sqlite3.connect(':memory:');"
         "print(' FTS5', 'ENABLE_FTS5' in [o[0] for o in c.execute('pragma compile_options')])"],
        capture_output=True, text=True,
    )
    print(check.stdout or check.stderr)
    print("Xong. Chạy tiếp:  !python kaggle/run_preprocess.py --run")
    return 0


def find_input_videos() -> list[Path]:
    """Tìm video trong /kaggle/input (mọi dataset đã gắn vào)."""
    root = Path("/kaggle/input")
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def write_config(videos_dir: Path) -> Path:
    """Sinh config trỏ mọi đường dẫn vào /kaggle/working.

    Đọc lại configs/default.yaml để mọi tham số đã chốt (threshold 0.5, L2 0.4,
    tên model...) giữ nguyên - chỉ thay phần paths.
    """
    import yaml

    with open(REPO / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["paths"] = {
        "data_root": str(DATA),
        "videos": str(videos_dir),
        "shots": str(DATA / "shots"),
        "shots_raw": str(DATA / "shots" / "raw"),
        "keyframes": str(DATA / "keyframes"),
        "thumbs": str(DATA / "thumbs"),
        "index_dir": str(DATA / "index"),
        "clip_embeddings": str(DATA / "index" / "clip_embeddings.npy"),
        "manifest": str(DATA / "index" / "manifest.csv"),
        "faiss_clip": str(DATA / "index" / "clip_vitl14.faiss"),
        "faiss_siglip": str(DATA / "index" / "siglip2_vitl16.faiss"),
        "index_meta": str(DATA / "index" / "meta.json"),
        "metadata_db": str(DATA / "metadata.db"),
    }
    cfg["runtime"]["device"] = "cuda"

    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return CONFIG


def restore_previous(source: Path) -> None:
    """Chép kết quả phiên trước vào /kaggle/working để chạy tiếp.

    /kaggle/input chỉ đọc nên không ghi thêm vào đó được; phải chép sang.
    """
    if not source.is_dir():
        print(f"Không có {source}, bỏ qua phần khôi phục", file=sys.stderr)
        return

    print(f"Khôi phục kết quả phiên trước từ {source} ...")
    for name in ("shots", "keyframes"):
        src = source / name
        if src.is_dir():
            shutil.copytree(src, DATA / name, dirs_exist_ok=True)
            print(f"  {name}: {sum(1 for _ in src.rglob('*'))} mục")
    for db in source.glob("*.db"):
        shutil.copy2(db, DATA / db.name)
        print(f"  {db.name}")


# --------------------------------------------------------------------------
# chạy song song hai GPU
# --------------------------------------------------------------------------


def gpu_count() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except Exception:
        return 0


def launch(step: str, shard: int, shards: int, gpu: int, extra: list[str]) -> subprocess.Popen:
    """Chạy một script trên đúng một GPU.

    CUDA_VISIBLE_DEVICES=<gpu> làm tiến trình con chỉ THẤY một GPU, nên nó gọi
    cuda:0 mà thực chất là GPU được chỉ định. Không phải sửa gì trong code model.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HOME", str(WORK / ".cache" / "hf"))
    env.setdefault("EASYOCR_MODULE_PATH", str(WORK / ".cache" / "easyocr"))

    argv = [
        sys.executable, str(REPO / "scripts" / step),
        "--config", str(CONFIG),
        "--shard", f"{shard}/{shards}",
        *extra,
    ]
    log = WORK / f"log_{step.split('_')[0]}_shard{shard}.txt"
    handle = open(log, "w", encoding="utf-8")
    print(f"  GPU{gpu}  shard {shard}/{shards}  ->  {log.name}")
    return subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO))


def run_step(name: str, step: str, shards: int, extra_for, deadline: float) -> bool:
    """Chạy một bước song song trên mọi GPU. Trả về True nếu mọi phần thành công.

    `extra_for(shard)` trả về tham số riêng cho từng phần — cần vì bước OCR phải
    ghi DB riêng cho mỗi phần.
    """
    print(f"\n{'=' * 70}\n### {name}\n{'=' * 70}")
    procs = [launch(step, i, shards, i, extra_for(i)) for i in range(shards)]

    t0 = time.time()
    while any(p.poll() is None for p in procs):
        if time.time() > deadline:
            print("\n!!! Hết thời gian cho phép - dừng có kiểm soát.", file=sys.stderr)
            print("    Video đang xử lý dở sẽ được làm lại ở phiên sau.", file=sys.stderr)
            for p in procs:
                if p.poll() is None:
                    p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
            return False
        time.sleep(10)

    codes = [p.returncode for p in procs]
    elapsed = time.time() - t0
    print(f"### {name}: exit={codes}, {elapsed / 60:.1f} phút")
    for i, code in enumerate(codes):
        if code != 0:
            print(f"  shard {i} lỗi - xem log_{step.split('_')[0]}_shard{i}.txt", file=sys.stderr)
            tail = (WORK / f"log_{step.split('_')[0]}_shard{i}.txt").read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-15:]
            print("  " + "\n  ".join(tail), file=sys.stderr)
    return all(code == 0 for code in codes)


def merge_ocr_dbs(shards: int) -> None:
    """Gộp DB OCR của các phần thành một file duy nhất."""
    parts = [DATA / f"metadata_shard{i}.db" for i in range(shards)]
    existing = [p for p in parts if p.exists()]
    if not existing:
        return

    print(f"\n### Gộp {len(existing)} DB OCR")
    subprocess.call(
        [sys.executable, str(REPO / "scripts" / "06_merge.py"),
         "--config", str(CONFIG), "--merge-db", *[str(p) for p in existing]],
        cwd=str(REPO),
    )


def pack(shards: int) -> None:
    """Đóng gói kết quả thành hai file: phần nhẹ và phần ảnh.

    Tách ra vì phần nhẹ (~100MB/30 video) là ĐỦ để build index, còn ảnh JPEG
    (~2,5GB/30 video) chỉ cần trên máy chạy search và UI.
    """
    print(f"\n{'=' * 70}\n### Đóng gói\n{'=' * 70}")

    meta = WORK / "aic_meta.tar.gz"
    with tarfile.open(meta, "w:gz") as tar:
        for path in sorted(DATA.rglob("*")):
            if path.is_dir() or path.suffix == ".jpg":
                continue
            tar.add(path, arcname=str(path.relative_to(DATA)))
    print(f"  {meta.name}  {meta.stat().st_size / 1e6:.0f} MB   <- đủ để build index")

    images = DATA / "keyframes"
    if images.is_dir():
        jpgs = list(images.rglob("*.jpg"))
        if jpgs:
            archive = WORK / "aic_keyframes.tar"
            with tarfile.open(archive, "w") as tar:      # JPEG nén thêm không lợi
                tar.add(images, arcname="keyframes",
                        filter=lambda t: t if t.name.endswith((".jpg", "/")) or t.isdir() else None)
            print(f"  {archive.name}  {archive.stat().st_size / 1e9:.2f} GB   "
                  f"({len(jpgs)} ảnh) <- chỉ cần ở máy chạy UI")

    print("\nTải hai file trên từ tab Output, hoặc Save Version để thành dataset")
    print("cho phiên sau (--resume-from /kaggle/input/<tên-dataset>).")


# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chạy A.1-A.4 trên Kaggle 2×T4")
    p.add_argument("--setup", action="store_true", help="Cài dependency rồi thoát")
    p.add_argument("--run", action="store_true", help="Chạy pipeline")
    p.add_argument("--shards", type=int, default=None,
                   help="Số phần (mặc định = số GPU nhìn thấy)")
    p.add_argument("--limit", type=int, default=None, help="Chỉ xử lý N video đầu mỗi phần")
    p.add_argument("--deadline-hours", type=float, default=DEFAULT_DEADLINE_HOURS,
                   help=f"Dừng sau bấy nhiêu giờ (mặc định {DEFAULT_DEADLINE_HOURS})")
    p.add_argument("--resume-from", default=None,
                   help="Thư mục kết quả phiên trước trong /kaggle/input")
    p.add_argument("--skip-ocr", action="store_true")
    p.add_argument("--pack-only", action="store_true", help="Chỉ đóng gói kết quả đã có")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.setup:
        return setup()

    DATA.mkdir(parents=True, exist_ok=True)

    if args.pack_only:
        write_config(DATA / "videos")
        pack(args.shards or 2)
        return 0

    if not args.run:
        print("Cần --setup hoặc --run. Xem hướng dẫn ở đầu file.", file=sys.stderr)
        return 2

    videos = find_input_videos()
    if not videos:
        print("Không tìm thấy video nào trong /kaggle/input.", file=sys.stderr)
        print("Gắn dataset video vào phần Input của notebook.", file=sys.stderr)
        return 1

    videos_dir = videos[0].parent
    if len({v.parent for v in videos}) > 1:
        print(f"! Video nằm rải ở nhiều thư mục, dùng thư mục của file đầu: {videos_dir}",
              file=sys.stderr)
    print(f"{len(videos)} video trong {videos_dir}")

    n_gpu = gpu_count()
    shards = args.shards or max(1, n_gpu)
    print(f"{n_gpu} GPU -> chia {shards} phần")
    if n_gpu < 2:
        print("! Chỉ thấy <2 GPU. Kiểm tra Accelerator = GPU T4 x2 trong Settings.",
              file=sys.stderr)

    write_config(videos_dir)
    if args.resume_from:
        restore_previous(Path(args.resume_from))

    deadline = time.time() + args.deadline_hours * 3600
    common = ["--device", "cuda"]
    if args.limit:
        common += ["--limit", str(args.limit)]

    steps = [
        ("Bước 1/4 · A.1 Shot detection", "01_shot_detect.py", lambda _: common),
        ("Bước 2/4 · A.2 Keyframe", "02_keyframe.py", lambda _: common),
        ("Bước 3/4 · A.3 Encode SigLIP2", "03_build_index.py", lambda _: common + ["--encode"]),
    ]
    if not args.skip_ocr:
        # Mỗi phần ghi DB RIÊNG: hai tiến trình cùng ghi một file SQLite sẽ tranh
        # khoá. Gộp lại thành một DB ở cuối bằng 06_merge.py.
        steps.append((
            "Bước 4/4 · A.4 OCR", "04_ocr.py",
            lambda i: ["--db", str(DATA / f"metadata_shard{i}.db")],
        ))

    t0 = time.time()
    for name, step, extra_for in steps:
        if not run_step(name, step, shards, extra_for, deadline):
            print(f"\n{name} chưa xong. Đóng gói phần đã làm được...", file=sys.stderr)
            break

    if not args.skip_ocr:
        merge_ocr_dbs(shards)

    print(f"\nTổng thời gian chạy: {(time.time() - t0) / 3600:.2f} giờ")
    pack(shards)

    print("\nBước cuối (chạy trên MỘT máy sau khi gom đủ mọi phần):")
    print("  python scripts/06_merge.py --check")
    print("  python scripts/02_keyframe.py --build-manifest")
    print("  python scripts/03_build_index.py --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
