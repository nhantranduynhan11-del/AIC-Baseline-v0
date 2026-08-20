"""A.4 OCR -> SQLite FTS5. Engine: EasyOCR (da chot).

Chay tren ANH KEYFRAME da co tu A.2, khong dong toi video nua.

Moi vung chu EasyOCR doc duoc la MOT DONG trong bang `ocr`, khong gop cac vung
lai thanh mot chuoi. Hai ly do:
  - Moi vung co confidence rieng; gop lai thi mat thong tin do.
  - Phrase query se sai neu gop: cum chu ghep tu hai vung khong lien quan tren
    man hinh van khop, tao ket qua sai ma khong ai thay.

Khong dung sliding window (da chot): moi keyframe chay OCR dung mot lan.

`text_norm` KHONG sinh o day - sqlite_store.insert_ocr lo, de moi dong vao DB
deu chac chan da chuan hoa.

Ghi chu: banner chu chay duoi man hinh (breaking-news ticker) trong video thoi
su la nhieu, khong lien quan toi truy van. Khong xu ly dac biet, de qua cung duoc.

easyocr/cv2 import LAZY -> module import duoc tren may khong cai chung.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

ENGINE = "easyocr"
DEFAULT_LANGUAGES = ("vi",)


def build_reader(languages: tuple[str, ...] = DEFAULT_LANGUAGES, gpu: bool = True):
    """Tao EasyOCR Reader. Lan dau chay se tai model ve ~/.EasyOCR.

    Tren vast.ai nen tro EASYOCR_MODULE_PATH vao volume de khong tai lai moi lan
    dung instance moi.
    """
    import easyocr

    return easyocr.Reader(list(languages), gpu=gpu)


def read_image(reader, image_bgr, min_confidence: float = 0.1) -> list[tuple[str, float]]:
    """Doc chu tren mot anh. Tra ve [(text, confidence), ...].

    min_confidence o day chi de vut rac ro rang. Nguong loc that su nen dat luc
    TRUY VAN (search_text(min_confidence=...)), vi doi nguong o day dong nghia
    phai chay lai OCR toan bo - cung ly do A.1 luu raw predictions.
    """
    results = reader.readtext(image_bgr)
    out: list[tuple[str, float]] = []
    for item in results:
        # EasyOCR tra ve (bbox, text, confidence)
        text, confidence = item[1], float(item[2])
        text = text.strip()
        if text and confidence >= min_confidence:
            out.append((text, confidence))
    return out


def iter_keyframes(keyframes_dir: str | Path, video_id: str) -> Iterator[tuple[int, Path]]:
    """Yield (frame_idx, duong dan anh) theo dung thu tu trong keyframes.json cua A.2."""
    from aic.preprocess.keyframe import KEYFRAME_META, read_keyframe_meta

    keyframes_dir = Path(keyframes_dir)
    meta = read_keyframe_meta(keyframes_dir / video_id / KEYFRAME_META)
    for kf in meta["keyframes"]:
        yield int(kf["frame_idx"]), keyframes_dir / kf["path"]


def run_video(
    reader,
    conn,
    keyframes_dir: str | Path,
    video_id: str,
    *,
    min_confidence: float = 0.1,
) -> dict[str, Any]:
    """Chay OCR cho toan bo keyframe cua mot video va ghi vao SQLite.

    Xoa ket qua cu cua video truoc khi ghi -> chay lai khong sinh dong trung lap.
    """
    import cv2

    from aic.store import sqlite_store as store

    store.delete_video(conn, video_id)

    n_rows = 0
    n_frames_with_text = 0
    n_frames = 0
    batch: list[tuple[str, int, str, float]] = []

    for frame_idx, img_path in iter_keyframes(keyframes_dir, video_id):
        n_frames += 1
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Khong doc duoc keyframe: {img_path}")
        found = read_image(reader, image, min_confidence)
        if found:
            n_frames_with_text += 1
        batch.extend((video_id, frame_idx, text, conf) for text, conf in found)

    if batch:
        n_rows = store.insert_ocr(conn, batch)
    store.mark_done(conn, video_id, n_rows, ENGINE)
    conn.commit()

    return {
        "video_id": video_id,
        "n_frames": n_frames,
        "n_frames_with_text": n_frames_with_text,
        "n_rows": n_rows,
    }


def video_ids_with_keyframes(keyframes_dir: str | Path) -> list[str]:
    from aic.preprocess.keyframe import KEYFRAME_META

    return sorted(p.parent.name for p in Path(keyframes_dir).glob(f"*/{KEYFRAME_META}"))
