# Chạy tiền xử lý trên Kaggle (2×T4)

## Cài đặt notebook

| Mục | Giá trị |
|---|---|
| **Accelerator** | GPU T4 ×2 |
| **Internet** | On — bắt buộc, để `pip install` và tải weights model |
| **Input** | dataset chứa video của bạn |

Không bật Internet thì `pip install` chết ngay ô đầu tiên. Không chọn T4 ×2 thì script vẫn chạy nhưng chỉ dùng một GPU, mất một nửa tốc độ.

## Ba ô notebook

**Ô 1 — lấy code**

```python
!git clone -q https://github.com/nhantranduynhan11-del/AIC-Baseline-v0.git /kaggle/working/repo
```

**Ô 2 — cài dependency** (~3 phút)

```python
!python /kaggle/working/repo/kaggle/run_preprocess.py --setup
```

In ra số GPU nhìn thấy. Phải là **2**, nếu là 1 thì kiểm tra lại Accelerator.

**Ô 3 — chạy**

```python
!python /kaggle/working/repo/kaggle/run_preprocess.py --run
```

Chạy thử vài video trước khi chạy cả bộ:

```python
!python /kaggle/working/repo/kaggle/run_preprocess.py --run --limit 2
```

## Script làm gì

```
video trong /kaggle/input
        │
        ├── shard 0/2 ──► GPU 0 ──► A.1 → A.2 → A.3 encode → A.4 OCR
        └── shard 1/2 ──► GPU 1 ──► A.1 → A.2 → A.3 encode → A.4 OCR
                                          │
                                    gộp 2 DB OCR
                                          │
                                    đóng gói kết quả
```

Video được chia **xen kẽ** chứ không cắt thành hai khối liền nhau — độ dài video chênh nhau nhiều, cắt khối dễ khiến một GPU ôm toàn video dài rồi GPU kia ngồi không.

Mỗi tiến trình bị ghim vào một GPU bằng `CUDA_VISIBLE_DEVICES`, nên nó chỉ *nhìn thấy* một GPU và gọi `cuda:0` như bình thường — không phải sửa gì trong code model.

**Script không chạy `--build-manifest`.** Bước đó phải chạy đúng một lần, trên máy đã gom đủ kết quả của mọi người.

## Kết quả

Hai file trong tab Output:

| File | Kích thước (30 video) | Dùng để |
|---|---|---|
| `aic_meta.tar.gz` | ~100 MB | **đủ để build index** |
| `aic_keyframes.tar` | ~2,5 GB | chỉ cần ở máy chạy search và UI |

Tách làm hai vì phần nhẹ là thứ duy nhất cần cho bước gộp cuối. Ảnh JPEG chỉ cần trên máy sẽ phục vụ giao diện.

Log của từng phần nằm ở `/kaggle/working/log_<bước>_shard<i>.txt`.

**Chỉ cần tải hai file tar.** Bỏ qua `.cache/` (weights, tải lại được), `repo/` (đã có trên GitHub), và `aic-data/` — nội dung của nó đã nằm trong hai file tar rồi.

## Tải kết quả về

### Cách 1 — kaggle CLI (khuyến nghị cho file lớn)

Trên máy của bạn:

```bash
pip install kaggle
kaggle kernels output <username>/<kernel-slug> -p ./kaggle_out
```

Lấy `<username>/<kernel-slug>` từ URL notebook. Cần API token: Kaggle → Account → Create New API Token → lưu `kaggle.json` vào `~/.kaggle/`.

Đây là cách ổn định nhất — trình duyệt hay treo khi tải file vài GB từ tab Output.

### Cách 2 — cắt nhỏ rồi tải từ tab Output

```python
!python /kaggle/working/repo/kaggle/run_preprocess.py --pack-only --split 900 --clean
```

`--split 900` cắt gói ảnh thành các mảnh 900MB, `--clean` xoá `.cache/`, `repo/` và ảnh đã đóng gói để Output nhẹ hẳn.

Ghép lại ở máy đích:

```bash
cat aic_keyframes.tar.part* > aic_keyframes.tar
tar xf aic_keyframes.tar
tar xzf aic_meta.tar.gz
```

Thứ tự ghép theo tên file (`part00`, `part01`, …) nên `cat` với `*` không bao giờ ghép lộn.

### Cách 3 — Save Version thành dataset

**Save Version** → output thành một Kaggle Dataset → tải từ trang dataset hoặc `kaggle datasets download`. Cách này cũng chính là cách để chạy tiếp phiên sau.

## Giới hạn 9 giờ và cách chạy tiếp

Kaggle giết phiên GPU ở mốc 9 giờ. Script mặc định dừng ở **8,5 giờ** rồi đóng gói phần đã làm — chừa 30 phút để không mất trắng.

Chạy tiếp ở phiên sau:

1. **Save Version** để output của phiên này thành một dataset.
2. Ở notebook mới, gắn dataset đó vào Input.
3. Chạy với `--resume-from`:

```python
!python /kaggle/working/repo/kaggle/run_preprocess.py --run \
    --resume-from /kaggle/input/<tên-dataset>/aic-data
```

Script chép kết quả cũ sang `/kaggle/working` (vì `/kaggle/input` chỉ đọc) rồi bỏ qua mọi video đã xong.

Quota GPU của Kaggle là **30 giờ/tuần**, nên một tài khoản chạy được khoảng 3 phiên đầy mỗi tuần.

## Sau khi mọi người chạy xong

Trên **một** máy, gom kết quả của mọi người rồi làm **đúng thứ tự này**:

```bash
# 1. Giải nén vào data/
tar xzf aic_meta.tar.gz -C data
tar xf aic_keyframes.tar -C data

# 2. Gộp DB OCR
python scripts/06_merge.py --merge-db data/metadata_shard*.db

# 3. Kiểm tra đã đủ chưa — phải báo "Đủ hết" mới đi tiếp
python scripts/06_merge.py --check

# 4. Sinh manifest  <-- CHỈ CHẠY MỘT LẦN, sau khi đã gom đủ
python scripts/02_keyframe.py --build-manifest

# 5. Build 2 FAISS index
python scripts/03_build_index.py --build

# 6. Thumbnail cho UI
python scripts/05_thumbnails.py
```

⚠️ **Bước 4 phải chạy sau bước 3.** Kaggle cố tình không chạy `--build-manifest` vì manifest là nguồn sự thật của bất biến ID: sinh nó khi mới có một phần dữ liệu sẽ cho ra manifest thiếu, hai FAISS index vẫn khớp manifest thiếu ấy, mọi assert vẫn xanh, và những video bị sót sẽ vĩnh viễn không tìm ra được.

Xem thêm mục "Chia việc cho nhiều người" trong [RUNBOOK.md](../RUNBOOK.md).

## Sự cố hay gặp

| Triệu chứng | Xử lý |
|---|---|
| `pip install` chết ở ô 2 | chưa bật Internet trong Settings |
| Chỉ thấy 1 GPU | Accelerator chưa đặt GPU T4 ×2 |
| `Không tìm thấy video nào trong /kaggle/input` | chưa gắn dataset vào Input |
| Hết dung lượng `/kaggle/working` | giới hạn 20 GB — chạy ít video hơn mỗi phiên, hoặc thêm `--skip-ocr` rồi chạy OCR ở phiên riêng |
| `moov atom not found` | video trong dataset bị hỏng — xem mục A.1 trong RUNBOOK |
| `temp.zip: No such file or directory` | hai tiến trình đua nhau tải model — đã sửa, `git pull` lấy bản mới |
| Output không có `manifest.csv` | **đúng như thiết kế** — Kaggle không chạy `--build-manifest`. Chạy nó ở bước 4 trên máy gộp |
| `05_thumbnails.py` báo thiếu manifest | đã sửa: giờ nó đọc `keyframes.json` khi chưa có manifest. `git pull` |

## Vì sao phải tải weights trước

Trước khi chạy song song, script tải sẵn CLIP, SigLIP2 và EasyOCR **ở tiến trình cha**. Không làm vậy thì hai tiến trình con cùng thấy cache trống và cùng tải về một chỗ.

EasyOCR là ca hỏng nặng nhất: nó tải về một file tên **cố định** `temp.zip`, giải nén xong thì `os.remove()`. Tiến trình chậm hơn gọi `remove` trên file vừa bị xoá và chết với `FileNotFoundError`.

Bước tải sẵn mất vài phút ở lần chạy đầu, sau đó cache nằm trong `/kaggle/working/.cache` nên các lần sau (và cả hai GPU) đều dùng lại.
