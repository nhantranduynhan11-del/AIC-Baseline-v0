"""B.1 - ghep toan bo luong search, xuat top-100.

    query -> encode 2 model -> search FAISS x2 -> RRF -> [hard filter] -> top-100

THU TU QUAN TRONG: RRF chay tren TOAN BO ung vien (khong cat), roi moi loc, roi
moi cat top-100. Neu cat 100 truoc rồi loc thi ket qua tra ve se it hon 100 mot
cach vo ly - trong khi van con thua ung vien hop le o hang 101 tro di.

Giai doan 2 se chen temporal re-rank vao NGAY TRUOC buoc cat top-100.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aic.retrieval.filters import FilterResult, apply_filter, ocr_allowed_idxs
from aic.retrieval.fusion import DEFAULT_RRF_K, FusedHit, reciprocal_rank_fusion
from aic.retrieval.search import IndexBundle


def search(
    bundle: IndexBundle,
    encoder,
    *,
    text: str | None = None,
    image_rgb: np.ndarray | None = None,
    top_k_per_model: int = 1000,
    rrf_k: int = DEFAULT_RRF_K,
    top_n: int = 100,
    allowed_idxs: set[int] | None = None,
) -> list[FusedHit]:
    """Chay het luong B.1 cho mot query, tra ve top-N.

    allowed_idxs: tap idx duoc phep di tiep (ket qua hard filter). None = khong loc.
    top_n mac dinh 100 - gioi han nop bai cua ban to chuc.
    """
    vectors = encoder.encode(text=text, image_rgb=image_rgb)
    ranked_lists = bundle.search(vectors, top_k_per_model)

    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=None)
    fused = apply_filter(fused, allowed_idxs)
    return fused[:top_n]


def search_with_ocr(
    bundle: IndexBundle,
    encoder,
    conn,
    *,
    ocr_query: str,
    ocr_phrase: bool = True,
    ocr_min_confidence: float = 0.3,
    **kwargs: Any,
) -> tuple[list[FusedHit], FilterResult]:
    """Nhu `search` nhung co them dieu kien OCR. Tra ve ca so lieu cua filter.

    Tra kem FilterResult de cho goi biet vi sao ket qua it - do filter qua chat
    hay do that su khong co gi khop.
    """
    result = ocr_allowed_idxs(
        conn, bundle, ocr_query, phrase=ocr_phrase, min_confidence=ocr_min_confidence
    )
    hits = search(bundle, encoder, allowed_idxs=result.allowed, **kwargs)
    return hits, result


def search_from_config(bundle: IndexBundle, encoder, cfg: Any, **kwargs) -> list[FusedHit]:
    """Nhu `search` nhung lay tham so mac dinh tu config."""
    kwargs.setdefault("top_k_per_model", cfg.retrieval.top_k_per_model)
    kwargs.setdefault("rrf_k", cfg.retrieval.rrf_k)
    kwargs.setdefault("top_n", cfg.retrieval.final_top_n)
    return search(bundle, encoder, **kwargs)


def hydrate(bundle: IndexBundle, hits: list[FusedHit]) -> list[dict[str, Any]]:
    """Gan metadata manifest + thu hang o tung model vao ket qua da gop."""
    rows = []
    for hit in hits:
        entry = bundle.entry(hit.idx)
        rows.append(
            {
                "idx": hit.idx,
                "video_id": entry.video_id,
                "frame_idx": entry.frame_idx,
                "pts_time": entry.pts_time,
                "path": entry.path,
                "score": hit.score,
                "rank": hit.rank,
                "ranks": hit.ranks,
            }
        )
    return rows
