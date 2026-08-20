"""B.1 buoc 5 - hard filter (TANG B: fusion giua cac modality).

Hai dieu quan trong nhat, deu la loi hệ thong AIC25 cu:

  1. Day la POST-FILTER: chay SAU khi da co ket qua RRF, KHONG dung de search
     truc tiep. Semantic search quyet dinh thu hang; OCR chi thu hep lai.
  2. KHONG gop modality bang RRF. Cac modality tra loi nhung cau hoi KHAC NHAU
     ("giong ngu nghia?" vs "co chu nay khong?"), nen khong the gop theo thu hang
     nhu tang A.

Optional theo tung query: chi ap khi cau truy van thuc su co yeu cau tuong ung,
khong ap mac dinh.

Thu tu bat buoc trong pipeline:
    RRF tren TOAN BO ung vien  ->  loc  ->  cat top-100
Cat top-100 truoc roi moi loc se cho ra it hon 100 ket qua mot cach vo ly.

Nang cap sau (N.1): thay hard filter bang harmonic mean tren diem da min-max,
dung bm25 cua FTS5 lam diem cho modality OCR. Khi do keyframe khong thoa dieu
kien chi TUT HANG chu khong bi vut bo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from aic.retrieval.fusion import FusedHit


@dataclass(frozen=True)
class FilterResult:
    """Tap idx duoc phep di tiep, kem so lieu de biet filter co qua chat khong."""

    allowed: set[int]
    n_rows: int      # so dong OCR khop
    n_frames: int    # so keyframe rieng biet khop
    n_unknown: int   # cap (video_id, frame_idx) khong co trong manifest

    def __bool__(self) -> bool:
        return bool(self.allowed)


def ocr_allowed_idxs(
    conn,
    bundle,
    query: str,
    *,
    phrase: bool = True,
    min_confidence: float = 0.3,
) -> FilterResult:
    """Tim cac keyframe co chua cum chu, tra ve tap idx cua manifest.

    SQLite tra ve (video_id, frame_idx); FAISS/RRF lam viec tren idx. Buoc anh xa
    di qua bundle.frame_lookup.

    n_unknown > 0 nghia la DB OCR va manifest lech nhau - thuong do build lai
    manifest sau khi da chay OCR. Khong raise (van loc duoc bang phan con lai)
    nhung phai dem va bao ra de con biet.
    """
    from aic.store import sqlite_store as store

    rows = store.search_text(conn, query, phrase=phrase, min_confidence=min_confidence)

    allowed: set[int] = set()
    frames: set[tuple[str, int]] = set()
    unknown: set[tuple[str, int]] = set()
    for row in rows:
        key = (row["video_id"], int(row["frame_idx"]))
        frames.add(key)
        idx = bundle.idx_of(*key)
        if idx is None:
            unknown.add(key)
        else:
            allowed.add(idx)

    return FilterResult(
        allowed=allowed, n_rows=len(rows), n_frames=len(frames), n_unknown=len(unknown)
    )


def apply_filter(hits: Sequence[FusedHit], allowed: set[int] | None) -> list[FusedHit]:
    """Giu lai cac hit co idx trong `allowed`, DANH SO LAI rank tu 1.

    Danh so lai la bat buoc: rank cua FusedHit la vi tri trong danh sach, de
    nguyen sau khi loc thi se thung lo (1, 5, 9...) va sai o moi noi dung toi no.

    allowed = None -> khong loc gi (query khong co dieu kien OCR).
    """
    if allowed is None:
        return list(hits)
    kept = [hit for hit in hits if hit.idx in allowed]
    return [
        FusedHit(idx=hit.idx, score=hit.score, rank=position, ranks=hit.ranks)
        for position, hit in enumerate(kept, start=1)
    ]


def describe(result: FilterResult) -> str:
    """Mot dong tom tat de in ra CLI / log."""
    text = f"{result.n_rows} dong OCR khop tren {result.n_frames} keyframe"
    if result.n_unknown:
        text += f" ({result.n_unknown} keyframe khong co trong manifest - DB va manifest lech)"
    return text
