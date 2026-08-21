# RUNBOOK — chạy end-to-end

Hướng dẫn chạy toàn bộ hệ thống từ video thô đến file CSV nộp bài.

Mỗi bước có: **lệnh**, **kết quả mong đợi**, **cách kiểm tra**, và **cái gì có thể sai**. Đừng bỏ phần kiểm tra — gần như mọi lỗi trong pipeline này đều thuộc loại *vẫn chạy, vẫn ra kết quả, nhưng sai*.

> **Nguyên tắc chung: luôn chạy `--limit 2` trước.** Mọi script đều có cờ này. Chạy 2 video, soi kết quả, rồi mới chạy cả bộ. Một tham số sai phát hiện ở phút thứ 3 rẻ hơn nhiều so với phát hiện sau 6 tiếng.

---

## 0. Chuẩn bị

### Máy dev (Windows, không GPU)

Chỉ để viết code, chạy test, sửa giao diện.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

Mong đợi: **tất cả pass** (hiện 201, một vài test bị skip nếu máy chưa cài `open_clip`). Test nào đỏ thì dừng lại sửa trước, đừng mang lên GPU thuê.

### Máy chạy thật (GPU vast.ai, Linux)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

Trỏ cache model vào volume, nếu không mỗi lần dựng instance mới sẽ tải lại vài GB:

```bash
export HF_HOME=/workspace/.cache/hf              # open_clip + timm
export EASYOCR_MODULE_PATH=/workspace/.cache/easyocr
```

Kiểm tra GPU và thư viện:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import faiss; print('faiss', faiss.__version__)"
python -c "import sqlite3;c=sqlite3.connect(':memory:');print('FTS5', 'ENABLE_FTS5' in [o[0] for o in c.execute('pragma compile_options')])"
```

Cả ba phải in ra `True` / phiên bản. `cuda.is_available()` là `False` nghĩa là bản torch cài sai (bản CPU) — cài lại đúng bản CUDA trước khi làm gì tiếp.

### Đặt dữ liệu

Video vào `data/videos/` (đệ quy cũng được). `video_id` = **tên file không phần mở rộng** — đây chính là "Tên file video" trong file nộp bài, nên đừng đổi tên file sau khi đã chạy pipeline.

Ổ khác thì sửa `paths.data_root` và các `paths.*` trong `configs/default.yaml`, hoặc dùng cờ `--videos` / `--out` từng bước.

```bash
python -c "
import sys; sys.path.insert(0,'src')
from aic.config import load_config; from aic.preprocess.shot_detect import find_videos
c = load_config(); v = find_videos(c.paths.videos, c.shot_detection.video_ext)
print(len(v), 'video'); [print(' ', p.stem) for p in v[:5]]"
```

---

## Chạy nhanh cả pipeline

Khi đã quen và tin các tham số:

```bash
python scripts/00_run_all.py --device cuda --limit 2      # thử trước
python scripts/00_run_all.py --device cuda                # chạy thật
```

Đứt giữa chừng thì chạy lại đúng lệnh đó — mọi bước đều resume. Muốn chạy tiếp từ một bước: `--from 4`.

Phần dưới đây là từng bước một, kèm cách kiểm tra. **Lần chạy đầu tiên trên dữ liệu thật nên đi từng bước**, đừng dùng `00_run_all.py`.

---

## 1 · A.1 — Shot Detection (TransNetV2)

```bash
python scripts/01_shot_detect.py --limit 2 --device cuda
```

**Kết quả mong đợi**

```
5 video, 2 can xu ly (threshold=0.5, device=cuda)
Dang load TransNetV2...
[1/2] L01_V001.mp4: 187 shot, 27431 frame, fps=25.000, 34.2s
```

Ra `data/shots/<video_id>.json` và `data/shots/raw/<video_id>.npy`.

**Kiểm tra**

```bash
python -c "
import json,glob
for f in sorted(glob.glob('data/shots/*.json'))[:3]:
    d = json.load(open(f))
    dur = d['n_frames']/d['fps']
    print(f\"{d['video_id']}: fps={d['fps']:.3f} {d['n_frames']} frame = {dur/60:.1f} phut, {d['n_shots']} shot, {dur/d['n_shots']:.1f}s/shot\")"
```

Ba con số phải hợp lý:

- **`fps`** — so với fps thật của video. ⚠️ `get_video_fps()` của package trả về **25.0 khi ffmpeg probe thất bại, không báo lỗi**. Thấy đúng `25.000` mà video không phải 25fps thì fps đang sai, và `pts_time` của toàn bộ pipeline sẽ sai theo.
- **Thời lượng** — `n_frames / fps` phải khớp độ dài video thật. Lệch nhiều nghĩa là decode thiếu frame.
- **`s/shot`** — video thời sự thường 3–8 giây/shot. Dưới 1s là threshold quá nhạy (cắt vụn); trên 30s là quá ì (bỏ sót cảnh).

**Đổi threshold không cần chạy lại inference** — nhờ file `.npy`:

```bash
python scripts/01_shot_detect.py --rethreshold 0.4
```

**Có thể sai gì**

| Triệu chứng | Nguyên nhân |
|---|---|
| `fps` đúng 25.000 ở mọi video | ffmpeg không có trong PATH → probe thất bại lặng lẽ |
| CUDA OOM ở video dài | `predict_video` giữ cả video 48×27 trên GPU (~3.9KB/frame) |
| Số shot cực lớn | video có hiệu ứng chuyển cảnh liên tục, hoặc threshold quá thấp |
| `moov atom not found` | **file tải bị cắt cụt** — xem mục dưới |

**Kiểm tra dữ liệu trước, không cần GPU:**

```bash
python scripts/01_shot_detect.py --check-only
```

Chạy trên cả nghìn file trong vài giây. Bắt được: file thiếu / 0 byte, trang HTML báo lỗi tải về thay vì video (hay gặp khi `gdown` đụng quota Google Drive), con trỏ Git LFS chưa pull, MP4 thiếu box `ftyp`, và **MP4 tải cắt cụt** (có `ftyp` nhưng thiếu `moov`).

**Dấu hiệu tải cắt cụt** — kiểm tra kích thước:

```bash
ls -la data/videos/
```

Kích thước là **bội số chẵn của 64KB** (hoặc 1MB), hoặc nhiều file **trùng kích thước đến từng byte**, gần như chắc chắn là tải đứt tại ranh giới buffer. Video thật không có kích thước như vậy. Tải lại bằng `wget -c` và đối chiếu kích thước với nguồn.

Vì sao `ftyp` vẫn đúng mà file lại hỏng: `ftyp` nằm ở **đầu** file, còn `moov` (chứa toàn bộ chỉ mục: số frame, codec, vị trí dữ liệu) thường nằm ở **cuối**. Tải mất đuôi thì đầu file trông vẫn hoàn hảo.

---

## 2 · A.2 — Keyframe Extraction (CLIP + L2)

```bash
python scripts/02_keyframe.py --limit 2 --device cuda
```

**Kết quả mong đợi**

```
Dang load CLIP ViT-L-14-quickgelu (dfn2b)...
Model san sang tren cuda, dim=768
[1/2] L01_V001.mp4: 412 keyframe / 3429 frame lay mau (12.0% giu lai), 187 shot, 88.5s
```

Ra `data/keyframes/<video_id>/`: ảnh `<frame_idx>.jpg`, `clip.npy`, `keyframes.json`.

**Kiểm tra — `dim=768`.** Khác 768 nghĩa là model load sai; dừng lại ngay, đừng chạy tiếp.

**Kiểm tra — tỉ lệ giữ lại.** Đây là con số quan trọng nhất của bước này:

| % giữ lại | Nghĩa là |
|---|---|
| < 5% | ngưỡng 0.4 quá chặt — đang mất cảnh |
| **10–35%** | hợp lý |
| > 60% | ngưỡng quá lỏng — index phình, tốn RAM và thời gian encode vô ích |

Chỉnh bằng `--l2-threshold 0.35` (giữ nhiều hơn) hoặc `0.45` (giữ ít hơn).

**Kiểm tra — lệch frame giữa A.1 và A.2.** A.1 lấy frame bằng **ffmpeg**, A.2 đọc bằng **OpenCV**. Video CFR thì khớp; video VFR có thể lệch, và lệch thì keyframe bị gán sai shot lẫn sai `pts_time`:

```bash
python -c "
import json,glob,os
for f in sorted(glob.glob('data/keyframes/*/keyframes.json'))[:5]:
    m = json.load(open(f))
    s = json.load(open(f\"data/shots/{m['video_id']}.json\"))
    last = m['keyframes'][-1]['frame_idx'] if m['keyframes'] else 0
    flag = 'LECH!' if last > s['n_frames'] else 'ok'
    print(f\"{m['video_id']}: keyframe cuoi={last}, A.1 bao {s['n_frames']} frame -> {flag}\")"
```

**Sau khi TẤT CẢ video xong bước này** — không phải sau mỗi video:

```bash
python scripts/02_keyframe.py --build-manifest
```

```
manifest: data/index/manifest.csv  (48213 dong)
embedding CLIP: data/index/clip_embeddings.npy  (48213 x 768 float32, 0.15 GB)
```

> ⚠️ **Đây là ranh giới quan trọng nhất của cả pipeline.** Lệnh này sinh ra manifest — nguồn sự thật của bất biến ID. Chạy khi còn video chưa xong bước 2 sẽ ra manifest thiếu, và **mọi thứ sau đó khớp với manifest thiếu ấy mà không báo lỗi gì**. Đếm số thư mục trước khi chạy:

```bash
ls data/keyframes | wc -l          # phải bằng số video đã xử lý
head -3 data/index/manifest.csv
```

---

## 3 · A.3 — Indexing (SigLIP2 + 2 FAISS index)

```bash
python scripts/03_build_index.py --encode --limit 2 --device cuda
```

```
Dang load SigLIP2 ViT-L-16-SigLIP2-256 (webli)...
Model san sang tren cuda, dim=1024
[1/2] L01_V001: 412 vector, 41.3s
```

**Kiểm tra — `dim=1024`** (CLIP là 768). Script tự cảnh báo nếu lệch với config. Lần chạy đầu sẽ tải model từ HuggingFace, mất vài phút — chạy `--limit 2` trước để không phát hiện lỗi sau khi đã chờ xong.

Chạy hết rồi mới build index:

```bash
python scripts/03_build_index.py --encode --device cuda    # chạy cả bộ
python scripts/03_build_index.py --build
```

```
  clip     ntotal=48213  dim=768   RAM~0.15GB  ->  data/index/clip_embeddings...
  siglip2  ntotal=48213  dim=1024  RAM~0.20GB  ->  data/index/siglip2_...
  manifest 48213 dong, meta -> data/index/meta.json
```

**Kiểm tra — bất biến ID.** Chạy lệnh này sau mỗi lần build lại bất cứ thứ gì:

```bash
python -c "
import sys; sys.path.insert(0,'src')
from aic.manifest import check_alignment_from_meta
check_alignment_from_meta('data/index/meta.json','data/index/manifest.csv')
print('manifest va 2 index KHOP')"
```

**Kiểm tra — ngân sách RAM.** Con số `RAM~` là bộ nhớ FAISS Flat cần khi search. Cộng cả hai index rồi so với RAM máy thuê. Căng quá thì hỏi trước khi đổi sang float16 hoặc SQ8 — đó là thay đổi lệch khỏi baseline.

---

## 4 · A.4 — OCR (EasyOCR → SQLite FTS5)

```bash
python scripts/04_ocr.py --limit 2
```

```
Dang load EasyOCR ['vi']...
[1/2] L01_V001: 938 dong chu tren 301/412 keyframe (73.1% co chu), 121.4s
```

**Kiểm tra — tỉ lệ `% có chữ`.** Video thời sự Việt Nam gần như luôn có chữ (logo, ticker, tên người phát biểu). Dưới ~30% là dấu hiệu EasyOCR đang bỏ sót — kiểm tra lại xem có đang chạy CPU không, hoặc ảnh keyframe có bị nén hỏng không.

**Kiểm tra — tra thử vài cụm chữ thật:**

```bash
python scripts/04_ocr.py --stats
python scripts/04_ocr.py --query "thoi su"
python scripts/04_ocr.py --query "ha noi" --no-phrase
```

Gõ **không dấu** vẫn phải ra chữ có dấu. Không ra gì thì kiểm tra lại `text_norm` trong DB:

```bash
python -c "
import sqlite3
c = sqlite3.connect('data/metadata.db')
for r in c.execute('SELECT text, text_norm, confidence FROM ocr LIMIT 5'): print(r)"
```

Cột thứ hai phải là bản đã bỏ dấu và `đ→d`. Nếu vẫn còn dấu thì có dòng nào đó vào DB không qua `insert_ocr`.

> Lệnh `sqlite3` trần ở trên in tiếng Việt ra console Windows sẽ báo `UnicodeEncodeError`. Các script trong `scripts/` đã tự ép stdout sang UTF-8 nên không dính; chỉ các lệnh `python -c` tự gõ mới cần thêm `PYTHONIOENCODING=utf-8` ở đầu (hoặc `$env:PYTHONIOENCODING="utf-8"` trong PowerShell).

---

## 5 · Thumbnail cho UI

```bash
python scripts/05_thumbnails.py
```

Chỉ dùng CPU, chạy song song. Bỏ qua được — UI vẫn chạy, chỉ là cuộn grid nặng vì phải tải ảnh gốc quality 95.

---

## 6 · B.1 — Search

```bash
python scripts/run_search.py "người đàn ông mặc áo đỏ đang chạy" --show 10 --per-model
```

```
Index: 48213 keyframe, dim={'clip': 768, 'siglip2': 1024}
=== RRF (k=60), top-100, in 10 dong dau ===
  #1   0.032787  L01_V028  frame=3450    t=138.00s  [clip#1 siglip2#2]
  ...
100 ket qua, 63 xuat hien o ca 2 model
```

**Kiểm tra — dòng cuối "xuất hiện ở cả 2 model".** Đây là chỉ số nhanh cho biết ensemble có hoạt động không:

| Số trùng | Nghĩa là |
|---|---|
| gần 100 | hai model gần như đồng ý hoàn toàn — model thứ hai đóng góp ít |
| **30–70** | bổ trợ nhau tốt, đúng như mong đợi |
| gần 0 | **nghi vấn nghiêm trọng** — nhiều khả năng một index build sai hoặc encode nhầm model |

**Kiểm tra — hard filter:**

```bash
python scripts/run_search.py "biển báo giao thông" --ocr "ha noi"
```

Dòng `OCR filter ...` mà hiện `(N keyframe không có trong manifest - DB và manifest lệch)` thì DB OCR đang lệch với manifest — thường do build lại manifest sau khi đã chạy OCR. Chạy lại A.4.

**Xuất file nộp bài:**

```bash
python scripts/run_search.py "..." --export sub.csv
python scripts/run_search.py "..." --export qa.csv --task qa --answer "Màu đỏ"
```

Script tự đọc lại file vừa ghi để kiểm tra định dạng.

---

## 7 · API + UI

```bash
uvicorn aic.api.app:app --app-dir src --host 0.0.0.0 --port 8000
```

Mở `http://localhost:8000`. Trên vast.ai cần port forwarding hoặc SSH tunnel:

```bash
ssh -L 8000:localhost:8000 -p <port> root@<host>
```

**Kiểm tra trước khi mở UI:**

```bash
curl -s localhost:8000/health
```

Phải thấy `"ready": true`. Nếu `false` thì trường `error` nói rõ thiếu gì. Hai model nạp một lần lúc startup nên request đầu tiên sau khi khởi động sẽ chờ vài chục giây.

Sửa giao diện trên máy không GPU:

```bash
python scripts/ui_demo.py          # http://127.0.0.1:8010, dữ liệu giả
```

---

## Thứ tự phụ thuộc

```
1. Shot detection ──► 2. Keyframe ──► 3. build-manifest ──┬─► 4. Encode SigLIP2 ─► 5. Build FAISS ─┐
                                                          ├─► 6. OCR ─────────────────────────────┼─► Search
                                                          └─► 7. Thumbnail ───────────────────────┘
```

Bước 1→2→3 **bắt buộc tuần tự**. Sau bước 3, các nhánh 4-5 / 6 / 7 độc lập, chạy song song được nếu đủ GPU.

---

## Chia việc cho nhiều người

150 video chia 5 người thì **mỗi người chỉ chạy 30 video của mình**, không ai phải chạy hết.

Mọi sản phẩm của A.1–A.3 đều nằm theo **từng video** trong thư mục riêng, nên gộp lại chỉ là copy file. Chia bằng cách đơn giản nhất: mỗi người bỏ 30 video của mình vào `data/videos/` rồi chạy y hệt hướng dẫn phía trên. Không cần cờ gì đặc biệt.

**Chia danh sách video sao cho không trùng nhau.** Hai người cùng xử lý một video sẽ tạo ra hai bản ghi cùng `video_id`, và bước gộp không phát hiện được đó là lỗi.

### Mỗi người chạy

```bash
python scripts/00_run_all.py --device cuda      # bước 1 → 6, gồm cả OCR
```

Máy có **nhiều GPU** thì chia tiếp bằng `--shard I/N`, mỗi phần một GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/01_shot_detect.py --shard 0/2 --device cuda &
CUDA_VISIBLE_DEVICES=1 python scripts/01_shot_detect.py --shard 1/2 --device cuda &
wait
```

Chia xen kẽ nên hai phần cân tải dù video dài ngắn khác nhau. Riêng bước OCR, mỗi phần phải ghi DB riêng (`--db`) vì hai tiến trình cùng ghi một file SQLite sẽ tranh khoá.

**Chạy trên Kaggle 2×T4**: đã có sẵn script làm hết việc này — xem [kaggle/README.md](kaggle/README.md).

OCR phải chạy trên máy đã có ảnh keyframe, nên để nguyên trong phần việc mỗi người — đừng để dồn về một máy.

### Nộp về máy gộp

| Cần chuyển | Dung lượng / 30 video |
|---|---|
| `data/shots/` | ~5 MB |
| `data/keyframes/<id>/keyframes.json` + `clip.npy` + `siglip2.npy` | **~95 MB** |
| `data/metadata.db` (đổi tên theo người, vd `an.db`) | vài MB |
| `data/keyframes/<id>/*.jpg` | ~2,5 GB |

Chỉ cần **~0,5 GB** cho cả 150 video là đủ để build index — ảnh JPEG (12 GB) chỉ cần trên máy sẽ chạy search và UI. Nếu đường truyền hẹp: chạy `05_thumbnails.py` ở phía mỗi người rồi chỉ chuyển `data/thumbs/` (1,2 GB), chấp nhận ảnh chi tiết ở độ phân giải thấp hơn.

### Máy gộp làm gì

```bash
# 1. Gộp DB OCR của mọi người
python scripts/06_merge.py --merge-db /mnt/share/an.db /mnt/share/binh.db /mnt/share/*.db

# 2. Kiểm tra đã đủ chưa
python scripts/06_merge.py --check
```

`--check` đối chiếu từng video xem có đủ 5 thứ không (shot JSON, keyframes.json, clip.npy, siglip2.npy, OCR) và liệt kê đích danh video nào thiếu gì.

```
150 video trong data/

  bước                 đủ   thiếu
  A.1 shot            150       0
  A.2 keyframe        150       0
  A.2 clip.npy        150       0
  A.3 siglip2         148       2   <-- THIẾU
  A.4 OCR             150       0

2 video chưa xong:
  - L21_V087: thiếu A.3 siglip2
  - L21_V112: thiếu A.3 siglip2
```

**Chỉ khi `--check` báo "Đủ hết" mới được chạy bước gộp cuối:**

```bash
python scripts/02_keyframe.py --build-manifest
python scripts/03_build_index.py --build
python scripts/05_thumbnails.py
```

> ⚠️ Vì sao `--check` là bắt buộc: `--build-manifest` **không biết** là nó đang thiếu dữ liệu. Thiếu video nào thì manifest đơn giản là không có video đó, hai FAISS index vẫn khớp manifest thiếu ấy, mọi assert vẫn xanh, và hệ thống chạy hoàn hảo — chỉ là vĩnh viễn không bao giờ tìm ra video bị thiếu. Đây là lỗi tốn kém nhất có thể xảy ra và không có gì tự báo.

### Thứ tự bắt buộc

Chỉ **một người duy nhất** chạy `--build-manifest`, và chỉ **sau khi** đã thu đủ. Muốn thêm video sau đó thì phải chạy lại `--build-manifest` **và** `03 --build` — FAISS `IndexFlatIP` không thêm dần được, phải build lại toàn bộ. Build lại nhanh (chỉ đọc `.npy` rồi `add`, không encode lại), nên đây không phải vấn đề lớn.

---

## Ước lượng thời gian

Không có con số cố định — phụ thuộc GPU thuê, độ dài video và tỉ lệ giữ keyframe. **Cách đúng là tự đo:**

```bash
python scripts/00_run_all.py --limit 2 --device cuda
```

Lấy thời gian in ra mỗi bước, chia cho tổng số phút video của 2 video đó, rồi nhân với tổng số phút của cả bộ. Cộng thêm ~20% cho video dài hơn trung bình.

Bước tốn nhất thường là 2 (giải mã video + encode CLIP) và 4 (encode SigLIP2). Bước 6 (OCR) chạy trên ảnh nên nhanh hơn nhiều.

---

## Sự cố thường gặp

| Triệu chứng | Xử lý |
|---|---|
| `Chua co manifest / FAISS index` | chưa chạy bước 3 hoặc 5 |
| `FAISS index lech voi manifest` | build lại: bước 3 rồi bước 5. Không sửa tay |
| `Thieu data/keyframes/<id>/siglip2.npy` | video đó chưa encode xong — chạy lại `03 --encode` |
| `clip.npy co N hang nhung meta ghi M` | A.2 của video đó đứt giữa chừng — `02 --overwrite --limit`... cho riêng video đó |
| `Manifest lech ID tai dong N` | file manifest hỏng — chạy lại `02 --build-manifest` |
| `Ban SQLite ... khong bat FTS5` | đổi bản Python; FTS5 là bắt buộc |
| CUDA OOM ở bước 2 hoặc 4 | giảm `keyframe.batch_size` trong config (mặc định 32) |
| Search trả kết quả vô nghĩa | kiểm tra dòng "xuất hiện ở cả 2 model" (mục 6) |
| `/health` báo `ready: false` | đọc trường `error` — nó nói đúng thiếu gì |

Mọi bước đều **bỏ qua phần đã xong**, nên sau khi sửa lỗi chỉ cần chạy lại cùng một lệnh. Muốn ép làm lại: `--overwrite`.

---

## Checklist trước buổi thi

- [ ] `pytest tests/ -q` — tất cả pass
- [ ] `check_alignment_from_meta` — manifest và 2 index khớp
- [ ] `04_ocr.py --stats` — số dòng OCR hợp lý
- [ ] `curl localhost:8000/health` — `ready: true`, `ocr: true`
- [ ] Search thử 3–5 query mẫu, xem "xuất hiện ở cả 2 model" nằm trong 30–70
- [ ] Xuất thử một file CSV mỗi loại task, mở bằng text editor xem tận mắt
- [ ] Thumbnail đã sinh xong (grid mượt)
- [ ] Thuộc phím tắt: `/` `↑↓←→` `Space` `,` `.` `Ctrl+E`

---

## Chưa có trong bản này

Giai đoạn 2 trở đi chưa code: **ASR** (model chưa chốt), **Temporal Search** (3 ô Before/Now/After), **Relevance Feedback** (α = 0.75), **LLM Query Enhancement**. Object Detection đang hoãn.

Hệ quả cần biết khi thi: query mô tả nội dung **được nói ra** mà không nhìn thấy trong khung hình hiện chưa xử lý được (thiếu ASR), và query mô tả **chuỗi sự kiện theo thời gian** phải làm thủ công bằng dải keyframe lân cận trong UI (thiếu Temporal Search).
