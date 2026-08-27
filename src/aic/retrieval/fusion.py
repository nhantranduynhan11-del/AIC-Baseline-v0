"""B.1 buoc 4 - RRF (TANG A: fusion giua cac model embedding).

    RRF_Score(d) = sum_i  1 / (k + rank_i(d))        voi k = 60

Vi sao gop theo THU HANG chu khong theo diem: CLIP va SigLIP2 cung tra loi mot
cau hoi ("anh nao giong query nhat") nhung thang diem cosine cua chung khong so
sanh truc tiep duoc voi nhau. Gop theo vi tri xep hang moi dung ban chat.

Bon rang buoc cua baseline, deu co test:
  - TONG QUAT THEO N: nhan mot dict cac ranked list. Them model thu 3 chi la
    them mot key, khong phai sua ham.
  - Keyframe khong co trong top-K cua mot model thi BO QUA dong gop do (coi nhu 0),
    KHONG gan rank gia.
  - KHONG co trong so cho tung model. Cong thuc RRF chuan khong co weight.
  - KHONG dung RRF de gop MODALITY (OCR, ASR, object). Do la tang B, dung hard
    filter - xem filters.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedHit:
    """Mot ket qua sau khi gop. `ranks` giu lai vi tri o tung model de debug."""

    idx: int
    score: float
    rank: int
    ranks: dict[str, int] = field(default_factory=dict)
    sequence_idxs: list[int] = field(default_factory=list)


def _rank_of(item: Any, position: int, source: str) -> tuple[int, int]:
    """Tra ve (idx, rank) cua mot phan tu trong ranked list.

    Chap nhan ca `Hit` lan so nguyen thuan. Neu phan tu tu khai bao `rank` ma
    lech voi vi tri trong list thi bao loi: dau hieu list da bi cat/loc truoc khi
    dua vao RRF, luc do thu hang khong con dung nua.
    """
    idx = getattr(item, "idx", item)
    declared = getattr(item, "rank", None)
    if declared is not None and declared != position:
        raise ValueError(
            f"'{source}': phan tu thu {position} khai bao rank={declared}. "
            "Ranked list phai duoc truyen nguyen ven, dung thu tu - dung cat/loc truoc RRF."
        )
    return int(idx), position


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[Any]],
    k: int = DEFAULT_RRF_K,
    top_n: int | None = None,
) -> list[FusedHit]:
    """Gop N ranked list bang RRF. Tra ve danh sach da xep hang giam dan.

    ranked_lists: {ten_nguon: ranked list}. N = so key -> tong quat theo N.
    k: hang so RRF, chuan la 60.
    top_n: cat lay bao nhieu ket qua dau (None = giu het).
    """
    if k <= 0:
        raise ValueError(f"k phai > 0, nhan {k}")
    if not ranked_lists:
        raise ValueError("Khong co ranked list nao de gop")

    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}

    for source, hits in ranked_lists.items():
        for position, item in enumerate(hits, start=1):
            idx, rank = _rank_of(item, position, source)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
            ranks.setdefault(idx, {})[source] = rank

    # Sap xep giam dan theo diem; hoa diem thi uu tien idx nho hon de ket qua on
    # dinh giua cac lan chay (quan trong khi so sanh cau hinh voi nhau).
    order = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_n is not None:
        order = order[:top_n]

    return [
        FusedHit(idx=idx, score=score, rank=position, ranks=ranks[idx])
        for position, (idx, score) in enumerate(order, start=1)
    ]
