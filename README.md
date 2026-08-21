# AIC Challenge HCM 2026 — HCMUT Technologia — Baseline v0

Hệ thống truy xuất video. Mọi model / threshold / kiến trúc theo **skill `aic2026-baseline-v0`** — đó là nguồn sự thật, file này chỉ mô tả cách chạy repo.

> **Chạy lần đầu?** Đọc [RUNBOOK.md](RUNBOOK.md) — hướng dẫn từng bước end-to-end, kèm cách kiểm tra sau mỗi bước và các lỗi thường gặp.

## Setup

**Máy dev (Windows, không GPU)** — chỉ để viết code, chạy test, gọi API:

```
uv venv --python 3.11 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

**Máy chạy tiền xử lý (GPU thuê trên vast.ai, Linux)**:

```
uv venv --python 3.11 .venv && uv pip install -r requirements.txt
```

Đường dẫn interpreter khác nhau giữa hai máy: `.venv/Scripts/python.exe` (Windows) vs `.venv/bin/python` (Linux). Mọi đường dẫn trong code đều đi qua `pathlib` nên chạy được cả hai.

Trên vast.ai nhớ trỏ `paths.data_root` trong config vào volume có đủ dung lượng, và cân nhắc đặt `HF_HOME` sang volume đó để weight open_clip không bị tải lại mỗi lần dựng instance mới.

Kiểm tra nhanh:

```
.venv/Scripts/python.exe -m pytest tests/ -q
```

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `configs/` | `default.yaml` — toàn bộ tham số đã chốt |
| `data/` | video, shot, keyframe, FAISS index, SQLite (không commit) |
| `src/aic/models/` | CLIP ViT-L-14-quickgelu + SigLIP2 ViT-L-16-256, load bằng `open_clip` |
| `src/aic/preprocess/` | A.1 shot detect → A.2 keyframe → A.3 index; A.4 OCR |
| `src/aic/store/` | FAISS `IndexFlatIP`, SQLite FTS5 |
| `src/aic/retrieval/` | encode query → search → RRF → hard filter → top-100 |
| `src/aic/submit/` | export KIS / QA / TRAKE |
| `scripts/` | entrypoint chạy tay từng bước |
| `scripts/06_merge.py` | gộp kết quả nhiều máy + kiểm tra đủ dữ liệu |
| `kaggle/` | chạy tiền xử lý trên Kaggle 2×T4 |

## Bất biến quan trọng nhất

> `row i` của FAISS index CLIP == `row i` của FAISS index SigLIP2 == `manifest[i]`

`data/index/manifest.csv` là nguồn sự thật của thứ tự. Cả hai builder index đọc đúng file đó, đúng thứ tự, **không sort lại, không lọc thêm**. `data/index/meta.json` giữ `ntotal` để `assert_alignment()` chặn lệch mỗi lần load. Xem [manifest.py](src/aic/manifest.py).

## Pipeline

```
video → shot detect (TransNetV2, 0.5) → keyframe (CLIP, mỗi 8 frame, L2 > 0.4)
                                             │
                    ┌────────────────────────┼──────────────┐
                    ▼                        ▼              ▼
             FAISS ×2 (A.3)            OCR → SQLite    object detect (HOÃN)
```

Shot detect → keyframe chạy tuần tự. Sau khi có keyframe, các nhánh chạy song song được.

## Chạy cả pipeline

```bash
python scripts/00_run_all.py --device cuda --limit 2
```

Chạy A.1 → A.2 → build-manifest → A.3 encode → A.3 build → A.4 theo đúng thứ tự. Mọi bước resume được nên đứt giữa chừng chỉ cần chạy lại; `--from N` để chạy tiếp từ bước N, `--dry-run` để xem lệnh trước.

**Bước 3 (build-manifest) là ranh giới quan trọng**: nó sinh manifest — nguồn sự thật của bất biến ID. Chạy khi còn video chưa xong bước 2 sẽ ra manifest thiếu, và mọi thứ sau đó khớp với manifest thiếu ấy mà không báo lỗi.

## A.1 — Shot Detection

```bash
python scripts/01_shot_detect.py --videos /data/videos --device cuda
```

Mỗi video ra `data/shots/<video_id>.json` (fps, n_frames, danh sách shot) và `data/shots/raw/<video_id>.npy` (prediction thô). Mặc định **bỏ qua video đã có JSON** — đứt mạng giữa chừng thì chạy lại là resume; muốn ép chạy lại thì thêm `--overwrite`.

Nhờ file `.npy`, đổi threshold về sau không cần chạm GPU:

```bash
python scripts/01_shot_detect.py --rethreshold 0.4
```

## A.2 — Keyframe Extraction

```bash
python scripts/02_keyframe.py --device cuda
```

Mỗi video ra `data/keyframes/<video_id>/`: ảnh `<frame_idx>.jpg`, `clip.npy` (embedding CLIP của đúng các keyframe được giữ), `keyframes.json`. Resume mặc định như A.1.

Sau khi **tất cả** video xong, chạy bước gộp — đây là bước sinh ra bất biến ID:

```bash
python scripts/02_keyframe.py --build-manifest
```

Ra `data/index/manifest.csv` + `data/index/clip_embeddings.npy`, cùng thứ tự từng hàng. A.3 chỉ việc đọc lại, **không encode lại CLIP**.

## A.3 — Indexing

```bash
python scripts/03_build_index.py --encode --build --device cuda
```

`--encode` chạy SigLIP2 trên ảnh keyframe, ghi `data/keyframes/<video_id>/siglip2.npy` (resume theo video). `--build` đọc manifest, gộp embedding và ghi 2 file FAISS + `meta.json`.

Bước `--build` **không encode lại CLIP** — nó đọc thẳng `clip_embeddings.npy` mà A.2 đã ghi.

## A.4 — OCR

```bash
python scripts/04_ocr.py --limit 2
python scripts/04_ocr.py
```

EasyOCR chạy trên ảnh keyframe (không đụng lại video), ghi vào `data/metadata.db`. Mỗi vùng chữ đọc được là một dòng riêng, kèm `confidence`. Resume theo bảng `ocr_done` — video không có chữ nào vẫn được đánh dấu xong nên không bị chạy lại mãi.

Tra cứu DB không cần model:

```bash
python scripts/04_ocr.py --query "dien bien phu" --stats
python scripts/04_ocr.py --query "tin nong" --no-phrase
```

Gõ không dấu vẫn tìm được chữ có dấu — cả `Đ`. Mặc định tìm **cả cụm liền nhau** (phrase); `--no-phrase` chuyển sang đủ-từ-là-được.

## B.1 — Search (RRF, top-100)

```bash
python scripts/run_search.py "người đàn ông mặc áo đỏ đang chạy"
python scripts/run_search.py "biển báo giao thông" --ocr "dien bien phu"
python scripts/run_search.py --image data/keyframes/L01_V001/240.jpg --show 20
python scripts/run_search.py "..." --per-model
```

Luồng: encode 2 model → 2 lần search FAISS → RRF (`k=60`) → [OCR filter] → top-100. Cột `[clip#3 siglip2#7]` cho thấy thứ hạng gốc ở từng model.

`--ocr` là **optional theo từng query** — chỉ áp khi câu truy vấn thực sự có yêu cầu về chữ trên màn hình. Nó là **post-filter**, chạy sau RRF, không dùng để search trực tiếp.

## Export file nộp bài

```bash
python scripts/run_search.py "..." --export sub.csv
python scripts/run_search.py "..." --export qa.csv --task qa --answer "Màu đỏ"
```

| Task | Định dạng |
|---|---|
| KIS | `<video>,<frame_idx>` |
| Q&A | `<video>,<frame_idx>,<answer>` — answer ≤ 100 ký tự |
| TRAKE | `<video>,<frame_1>,<frame_2>,...,<frame_N>` — thứ tự thời gian |

UTF-8 không BOM, không header, tối đa 100 dòng. Ghi xong tự đọc lại bằng `csv.reader` để kiểm tra — bắt được lỗi quote/escape mà nhìn mắt thường dễ bỏ qua.

TRAKE cần nhiều query nên chưa có ở CLI; dùng `aic.submit.export.write_trake` trực tiếp.

## Thumbnail cho UI

```bash
python scripts/05_thumbnails.py
```

Keyframe lưu ở độ phân giải gốc quality 95 (cố ý, để OCR đọc được chữ nhỏ). Grid UI hiện ~60 ảnh cùng lúc nên cần bản 320px riêng. Chạy CPU, song song, không đụng tới bất biến ID.

## Chạy API

```bash
.venv/Scripts/python.exe -m uvicorn aic.api.app:app --app-dir src --port 8000
```

| Endpoint | Việc |
|---|---|
| `GET /health` | trạng thái + cấu hình đang chạy |
| `POST /search` | query text hoặc `image_idx` → top-100, kèm `ocr` filter tuỳ chọn |
| `GET /keyframe/{idx}` | ảnh độ phân giải gốc |
| `GET /thumb/{idx}` | thumbnail (rơi về ảnh gốc nếu chưa sinh) |
| `GET /neighbors/{idx}?w=5` | keyframe lân cận cùng video, theo thời gian |
| `GET /ocr/{idx}` | chữ đọc được trên keyframe |
| `POST /export` | danh sách đã chọn → CSV nộp bài |

Hai model nạp **một lần lúc startup**. Thiếu index thì API vẫn lên và trả 503 kèm lý do, thay vì crash.

## UI

Mở `http://localhost:8000` sau khi chạy API. Một file HTML + một file JS trong `web/`, FastAPI serve trực tiếp — không build step, sửa là F5 thấy ngay.

| Phím | Việc |
|---|---|
| `/` | nhảy vào ô query |
| `↑↓←→` | di chuyển con trỏ trong grid |
| `Space` | thêm/bỏ khỏi giỏ nộp bài |
| `,` `.` | keyframe trước/sau **trong cùng video** |
| `Enter` | dùng keyframe đang xem làm query (Video KIS) |
| `Ctrl+E` | xuất CSV |

Panel chi tiết hiện thứ hạng gốc ở từng model (`clip#14 siglip2#2`), chữ OCR đọc được, và dải keyframe lân cận. Dải này quan trọng: nhiều query trả về đúng video nhưng lệch vài keyframe — trượt bằng `,`/`.` sửa được mà không phải search lại, và keyframe lân cận vẫn thêm được vào giỏ dù không nằm trong kết quả.

Sửa giao diện trên máy không có GPU:

```bash
python scripts/ui_demo.py
```

Dựng 60 keyframe tổng hợp + FAISS + DB OCR thật, encoder giả. Toàn bộ đường đi của UI chạy thật, chỉ kết quả search là vô nghĩa.

## Trạng thái

| Task | Trạng thái |
|---|---|
| 1. Scaffolding | xong |
| 2. Shot detection | code xong, **chờ chạy thử trên vast.ai** |
| 3. Keyframe extraction | code xong, **chờ chạy thử trên vast.ai** |
| 4. Indexing (SigLIP2 + 2 FAISS) | code xong, **chờ chạy thử trên vast.ai** |
| 5. Retrieval (encode query + search) | code xong, **chờ chạy thử trên vast.ai** |
| 6. RRF + top-100 | xong, có test |
| 7. OCR | code xong (EasyOCR), **chờ chạy thử trên vast.ai** |
| 8. Hard filter | xong, có test |
| 9. Orchestrator + export | xong, có test |
| 10. API layer + thumbnail | xong, có test |
| 11. UI | xong, đã kiểm chứng trên trình duyệt |

Giai đoạn 2 trở đi (ASR, temporal search, relevance feedback, LLM query enhancement) chưa scaffold.
