"""Xuat ket qua dung dinh dang nop bai cua ban to chuc.

Ba loai task:
    KIS   : <ten file video>, <frame idx>
    QA    : <ten file video>, <frame idx>, <answer>
    TRAKE : <ten file video>, <frame id 1>, <frame id 2>, ..., <frame id N>

Quy chuan CSV cua ban to chuc:
  - UTF-8, KHONG BOM
  - delimiter la dau phay
  - KHONG co header row
  - dau ngoac kep chi bat buoc khi truong co ky tu dac biet -> QUOTE_MINIMAL cua
    module `csv` lam dung viec nay, va tu escape dau nhay kep bang cach nhan doi
  - khoang trang dau/cuoi ĐƯỢC GIU NGUYEN, khong tu dong trim

⚠️ Quyet dinh: ghi KHONG co khoang trang sau dau phay ("L01_V028,3450"), du vi du
trong tai lieu hien thi co khoang trang. Ly do: ban to chuc noi ro khoang trang
khong bi trim, nen "L01_V028, 3450, Mau do" se lam answer thanh " Mau do" voi mot
dau cach thua o dau. Dau phay tran la dinh dang CSV chuan, moi parser deu doc duoc.

Toi da 100 dong moi file - gioi han nop bai. Vuot thi RAISE chu khong tu cat, de
khong bao gio am tham vut mat ket qua.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

MAX_ROWS = 100
MAX_ANSWER_CHARS = 100
TASKS = ("kis", "qa", "trake")


def _check_row_count(rows: Sequence[Any]) -> None:
    if len(rows) > MAX_ROWS:
        raise ValueError(
            f"{len(rows)} dong, vuot gioi han {MAX_ROWS} cua ban to chuc. "
            "Cat bot o phia goi ham de chu dong chon giu dong nao."
        )


def _writer(handle):
    # lineterminator mac dinh cua csv.writer la CRLF - ban to chuc chap nhan.
    return csv.writer(handle, delimiter=",", quoting=csv.QUOTE_MINIMAL)


def _open(path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="" theo yeu cau cua module csv; encoding utf-8 KHONG BOM.
    return open(path, "w", encoding="utf-8", newline="")


def write_kis(path: str | Path, rows: Iterable[tuple[str, int]]) -> int:
    """rows: (video_id, frame_idx). Tra ve so dong da ghi."""
    rows = [(str(video_id), int(frame_idx)) for video_id, frame_idx in rows]
    _check_row_count(rows)
    with _open(path) as f:
        _writer(f).writerows(rows)
    return len(rows)


def write_qa(path: str | Path, rows: Iterable[tuple[str, int, str]]) -> int:
    """rows: (video_id, frame_idx, answer). Answer toi da 100 ky tu."""
    prepared: list[tuple[str, int, str]] = []
    for video_id, frame_idx, answer in rows:
        answer = str(answer)
        if len(answer) > MAX_ANSWER_CHARS:
            raise ValueError(
                f"Answer dai {len(answer)} ky tu, vuot {MAX_ANSWER_CHARS}: {answer[:60]}..."
            )
        if not answer:
            raise ValueError(f"Answer rong cho {video_id} frame {frame_idx}")
        prepared.append((str(video_id), int(frame_idx), answer))

    _check_row_count(prepared)
    with _open(path) as f:
        _writer(f).writerows(prepared)
    return len(prepared)


def write_trake(
    path: str | Path,
    rows: Iterable[tuple[str, Sequence[int]]],
    n_events: int | None = None,
) -> int:
    """rows: (video_id, [frame_id_1, ..., frame_id_N]).

    n_events: so su kien ma cau truy van yeu cau. Truyen vao thi kiem tra tung
    dong co dung so frame - sai so luong la bai nop hong, phai biet truoc khi nop.

    Thu tu frame phai theo thu tu thoi gian cua cac su kien, tuc khong giam dan.
    """
    prepared: list[tuple[str, list[int]]] = []
    for video_id, frames in rows:
        frames = [int(f) for f in frames]
        if not frames:
            raise ValueError(f"{video_id}: TRAKE khong co frame nao")
        if n_events is not None and len(frames) != n_events:
            raise ValueError(
                f"{video_id}: co {len(frames)} frame nhung truy van yeu cau {n_events} su kien"
            )
        if any(b < a for a, b in zip(frames, frames[1:])):
            raise ValueError(
                f"{video_id}: frame khong theo thu tu thoi gian: {frames}"
            )
        prepared.append((str(video_id), frames))

    _check_row_count(prepared)
    with _open(path) as f:
        writer = _writer(f)
        for video_id, frames in prepared:
            writer.writerow([video_id, *frames])
    return len(prepared)


def hits_to_kis(rows: Sequence[dict[str, Any]]) -> list[tuple[str, int]]:
    """Doi ket qua da hydrate cua pipeline thanh dong KIS."""
    return [(row["video_id"], row["frame_idx"]) for row in rows]


def hits_to_qa(rows: Sequence[dict[str, Any]], answer: str) -> list[tuple[str, int, str]]:
    """Doi ket qua da hydrate thanh dong QA, dung chung mot answer cho moi dong."""
    return [(row["video_id"], row["frame_idx"], answer) for row in rows]


def validate_file(path: str | Path, task: str, n_events: int | None = None) -> dict[str, Any]:
    """Doc lai file da ghi va kiem tra dinh dang. Dung truoc khi nop cho chac.

    Doc bang chinh module csv nen phat hien duoc loi quote/escape - thu ma nhin
    bang mat rat de bo qua.
    """
    task = task.lower()
    if task not in TASKS:
        raise ValueError(f"task phai la mot trong {TASKS}, nhan '{task}'")

    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f, delimiter=","))

    if not rows:
        raise ValueError(f"{path}: file rong")
    if len(rows) > MAX_ROWS:
        raise ValueError(f"{path}: {len(rows)} dong, vuot {MAX_ROWS}")

    for line_no, row in enumerate(rows, start=1):
        if task == "kis" and len(row) != 2:
            raise ValueError(f"{path}:{line_no}: KIS can dung 2 cot, co {len(row)}")
        if task == "qa":
            if len(row) != 3:
                raise ValueError(f"{path}:{line_no}: QA can dung 3 cot, co {len(row)}")
            if len(row[2]) > MAX_ANSWER_CHARS:
                raise ValueError(f"{path}:{line_no}: answer {len(row[2])} ky tu")
        if task == "trake":
            if len(row) < 2:
                raise ValueError(f"{path}:{line_no}: TRAKE can it nhat 1 frame")
            if n_events is not None and len(row) - 1 != n_events:
                raise ValueError(
                    f"{path}:{line_no}: {len(row) - 1} frame, can {n_events}"
                )
        if task in ("kis", "qa"):
            int(row[1])          # raise neu frame idx khong phai so
        else:
            frames = [int(x) for x in row[1:]]
            if any(b < a for a, b in zip(frames, frames[1:])):
                raise ValueError(f"{path}:{line_no}: frame khong theo thu tu thoi gian")

    return {"task": task, "rows": len(rows), "path": str(path)}
