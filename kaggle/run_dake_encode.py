"""Encode CLIP + SigLIP2 + OCR cho bộ keyframe ĐÃ CÓ SẴN ẢNH, trên Kaggle 2×T4.

═══════════════════════════════════════════════════════════════════════════
DÁN VÀO NOTEBOOK KAGGLE
═══════════════════════════════════════════════════════════════════════════

    !git clone -q https://github.com/nhantranduynhan11-del/AIC-Baseline-v0.git /kaggle/working/repo
    !python /kaggle/working/repo/kaggle/run_dake_encode.py --setup
    !python /kaggle/working/repo/kaggle/run_dake_encode.py --run

Settings: Accelerator = **GPU T4 ×2**, Internet = **On**,
Input = dataset chứa `keyframes/<video_id>/` (ảnh + keyframes.json).

═══════════════════════════════════════════════════════════════════════════

Khác `run_preprocess.py` ở chỗ nào: script kia đi từ VIDEO (shot detect →
keyframe → encode → OCR). Script này bắt đầu từ ẢNH KEYFRAME đã có, chỉ chạy
ba bước cần GPU:

    encode CLIP  ->  <video_id>/clip.npy
    encode SigLIP2 -> <video_id>/siglip2.npy
    OCR          ->  metadata_dake_shardN.db

Dùng khi tập keyframe thay đổi sau A.2 — ví dụ sau khi gộp thêm keyframe DAKE
vào bộ baseline. Lúc đó clip.npy cũ không còn khớp số hàng nên phải encode lại
CẢ HAI model cho những video bị đổi, không chỉ SigLIP2.

⚠️ Phải encode lại TOÀN BỘ keyframe của video, không thể chỉ encode phần thêm:
vector trong .npy khớp theo THỨ TỰ dòng với keyframes.json, chèn giữa chừng là
lệch hết.

OCR thì chạy lại cả video (xoá rồi ghi lại). Không làm tăng dần được vì keyframe
không có chữ cũng không để lại dòng nào, nên không phân biệt được "đã xử lý,
không có chữ" với "chưa xử lý".
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
CONFIG = WORK / "kaggle_dake.yaml"
EMB_DIR = DATA / "emb"          # .npy ghi o day, tach khoi anh chi doc
DEFAULT_DEADLINE_HOURS = 8.5


def setup() -> int:
    packages = ["open_clip_torch", "timm", "faiss-cpu", "easyocr", "ffmpeg-python", "pyyaml"]
    print("Cài dependency (không cần transnetv2-pytorch vì không đụng tới video)...")
    code = subprocess.call([sys.executable, "-m", "pip", "install", "-q", *packages])
    if code != 0:
        print("pip install thất bại", file=sys.stderr)
        return code

    check = subprocess.run(
        [sys.executable, "-c",
         "import torch, faiss;"
         "print(' torch', torch.__version__, '| GPU:', torch.cuda.device_count());"
         "[print('  -', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"],
        capture_output=True, text=True,
    )
    print(check.stdout or check.stderr)
    print("Xong. Chạy tiếp: --run")
    return 0


def find_keyframes_root() -> Path | None:
    """Tìm thư mục chứa <video_id>/keyframes.json trong /kaggle/input."""
    root = Path("/kaggle/input")
    if not root.is_dir():
        return None
    for meta in root.rglob("keyframes.json"):
        return meta.parent.parent          # <root>/<video_id>/keyframes.json
    return None


def write_config(keyframes_dir: Path) -> Path:
    """Config trỏ keyframes vào /kaggle/working (đã chép ra), giữ nguyên tham số đã chốt."""
    import yaml

    with open(REPO / "configs" / "default.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {
        "data_root": str(DATA),
        "videos": str(DATA / "videos"),
        "shots": str(DATA / "shots"),
        "shots_raw": str(DATA / "shots" / "raw"),
        "keyframes": str(keyframes_dir),
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


def report_disk(keyframes_dir: Path) -> None:
    """In ngân sách đĩa. /kaggle/working giới hạn 20 GB và vượt là mất cả phiên.

    Ảnh KHÔNG được chép sang working: chúng ở lại /kaggle/input (chỉ đọc, không
    tính vào hạn mức) và .npy ghi sang EMB_DIR. Chép ảnh sang working từng làm
    hỏng một phiên 7 giờ vì bộ ảnh 14 GB cộng cache weights vượt luôn 20 GB.
    """
    n_img = sum(1 for _ in keyframes_dir.rglob("*.jpg"))
    size = sum(f.stat().st_size for f in keyframes_dir.rglob("*.jpg"))
    emb = n_img * (768 + 1024) * 4
    print("\nNgân sách /kaggle/working (giới hạn 20 GB):")
    print(f"  ảnh ở /kaggle/input (chỉ đọc, KHÔNG tính) {size/1e9:>6.2f} GB")
    print(f"  cache weights                             {4.0:>6.2f} GB")
    print(f"  .npy CLIP + SigLIP2                       {emb/1e9:>6.2f} GB")
    print(f"  gói tar.gz cuối                           {emb/1e9:>6.2f} GB")
    print(f"  {'-'*44}")
    print(f"  tổng dự kiến                              {(4e9 + 2*emb)/1e9:>6.2f} GB")

    free = shutil.disk_usage(WORK).free
    need = 4e9 + 2 * emb
    print(f"  đĩa còn trống                             {free/1e9:>6.2f} GB")
    if free < need * 1.2:
        print("  ! Sát hạn mức. Cân nhắc --skip-ocr rồi chạy OCR ở phiên riêng.",
              file=sys.stderr)


def set_cache_env() -> None:
    os.environ.setdefault("HF_HOME", str(WORK / ".cache" / "hf"))
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(WORK / ".cache" / "easyocr"))


def warm_caches(skip_ocr: bool) -> None:
    """Tải weights ở tiến trình cha trước khi chia GPU.

    EasyOCR tải về file tên cố định `temp.zip` rồi xoá sau khi giải nén; hai tiến
    trình cùng tải sẽ có một cái gọi remove trên file đã biến mất và chết.
    """
    import yaml

    with open(CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"\n{'=' * 70}\n### Tải sẵn weights\n{'=' * 70}")
    for key in ("clip", "siglip2"):
        name, tag = cfg["models"][key]["name"], cfg["models"][key]["pretrained"]
        t0 = time.time()
        try:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(name, pretrained=tag, device="cpu")
            del model
            print(f"  {key:<8} {name}  {time.time() - t0:.0f}s")
        except Exception as exc:
            print(f"  ! {key}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not skip_ocr:
        t0 = time.time()
        try:
            import easyocr

            easyocr.Reader(list(cfg["ocr"]["languages"]), gpu=False)
            print(f"  easyocr  {time.time() - t0:.0f}s")
        except Exception as exc:
            print(f"  ! easyocr: {type(exc).__name__}: {exc}", file=sys.stderr)


def launch(step: str, shard: int, shards: int, gpu: int, extra: list[str]) -> subprocess.Popen:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    argv = [
        sys.executable, str(REPO / "scripts" / step),
        "--config", str(CONFIG), "--shard", f"{shard}/{shards}", *extra,
    ]
    log = WORK / f"log_{step.split('_')[0]}_shard{shard}.txt"
    handle = open(log, "w", encoding="utf-8")
    print(f"  GPU{gpu}  shard {shard}/{shards}  ->  {log.name}")
    return subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, cwd=str(REPO))


def run_step(name: str, step: str, shards: int, extra_for, deadline: float) -> bool:
    print(f"\n{'=' * 70}\n### {name}\n{'=' * 70}")
    procs = [launch(step, i, shards, i, extra_for(i)) for i in range(shards)]
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        if time.time() > deadline:
            print("\n!!! Hết thời gian - dừng có kiểm soát.", file=sys.stderr)
            for p in procs:
                p.terminate()
            for p in procs:
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
            return False
        time.sleep(10)

    codes = [p.returncode for p in procs]
    print(f"### {name}: exit={codes}, {(time.time() - t0) / 60:.1f} phút")
    for i, code in enumerate(codes):
        if code != 0:
            log = WORK / f"log_{step.split('_')[0]}_shard{i}.txt"
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            print(f"  shard {i} lỗi:\n  " + "\n  ".join(tail), file=sys.stderr)
    return all(c == 0 for c in codes)


def pack(shards: int) -> None:
    """Đóng gói .npy + DB OCR. KHÔNG kèm ảnh - máy gộp đã có ảnh rồi.

    Lấy .npy từ EMB_DIR chứ không từ thư mục ảnh: ảnh nằm ở /kaggle/input chỉ
    đọc, vector được ghi riêng sang /kaggle/working.
    """
    print(f"\n{'=' * 70}\n### Đóng gói\n{'=' * 70}")
    out = WORK / "aic_dake_vectors.tar.gz"
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        if EMB_DIR.is_dir():
            for path in sorted(EMB_DIR.rglob("*.npy")):
                tar.add(path, arcname=f"keyframes/{path.relative_to(EMB_DIR)}")
                n += 1
        else:
            print(f"  ! Không có {EMB_DIR} — chưa encode gì?", file=sys.stderr)
        for i in range(shards):
            db = DATA / f"metadata_dake_shard{i}.db"
            if db.exists():
                tar.add(db, arcname=db.name)
    print(f"  {out.name}  {out.stat().st_size / 1e6:.0f} MB  ({n} file .npy)")
    print("\nTải file này về máy gộp, giải nén đè lên data/, rồi:")
    print("  python scripts/06_merge.py --merge-db data/metadata_dake_shard*.db")
    print("  python scripts/02_keyframe.py --build-manifest")
    print("  python scripts/03_build_index.py --build")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode CLIP+SigLIP2+OCR từ ảnh keyframe có sẵn")
    p.add_argument("--setup", action="store_true")
    p.add_argument("--run", action="store_true")
    p.add_argument("--shards", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--deadline-hours", type=float, default=DEFAULT_DEADLINE_HOURS)
    p.add_argument("--skip-ocr", action="store_true")
    p.add_argument("--pack-only", action="store_true",
                   help="Chỉ đóng gói kết quả đã có, không encode lại")
    p.add_argument("--keyframes", default=None, help="Ghi đè thư mục keyframe trong /kaggle/input")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.setup:
        return setup()
    if args.pack_only:
        pack(args.shards or 2)
        return 0
    if not args.run:
        print("Cần --setup, --run hoặc --pack-only", file=sys.stderr)
        return 2

    src = Path(args.keyframes) if args.keyframes else find_keyframes_root()
    if src is None or not src.is_dir():
        print("Không tìm thấy keyframes.json nào trong /kaggle/input.", file=sys.stderr)
        print("Gắn dataset chứa keyframes/<video_id>/ vào Input.", file=sys.stderr)
        return 1

    DATA.mkdir(parents=True, exist_ok=True)
    set_cache_env()
    keyframes_dir = src                 # ĐỌC THẲNG từ /kaggle/input, không chép
    n_videos = len([p for p in keyframes_dir.iterdir() if p.is_dir()])
    print(f"{n_videos} video, ảnh đọc thẳng từ {keyframes_dir}")
    report_disk(keyframes_dir)

    try:
        import torch

        n_gpu = torch.cuda.device_count()
    except Exception:
        n_gpu = 0
    shards = args.shards or max(1, n_gpu)
    print(f"{n_gpu} GPU -> chia {shards} phần")
    if n_gpu < 2:
        print("! Chỉ thấy <2 GPU. Kiểm tra Accelerator = GPU T4 x2.", file=sys.stderr)

    write_config(keyframes_dir)
    warm_caches(args.skip_ocr)

    deadline = time.time() + args.deadline_hours * 3600
    common = ["--device", "cuda", "--emb-dir", str(EMB_DIR)]
    if args.limit:
        common += ["--limit", str(args.limit)]

    steps = [
        ("Bước 1/3 · Encode CLIP", "03_build_index.py", lambda _: common + ["--encode-clip"]),
        ("Bước 2/3 · Encode SigLIP2", "03_build_index.py", lambda _: common + ["--encode"]),
    ]
    if not args.skip_ocr:
        steps.append((
            "Bước 3/3 · OCR", "04_ocr.py",
            lambda i: (["--limit", str(args.limit)] if args.limit else [])
            + ["--db", str(DATA / f"metadata_dake_shard{i}.db"), "--overwrite"],
        ))

    t0 = time.time()
    for name, step, extra_for in steps:
        if not run_step(name, step, shards, extra_for, deadline):
            print(f"\n{name} chưa xong. Đóng gói phần đã làm được...", file=sys.stderr)
            break

    print(f"\nTổng thời gian: {(time.time() - t0) / 3600:.2f} giờ")
    pack(shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
