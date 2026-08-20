"""B.1 buoc 3 - search FAISS tung model, tra ve danh sach CO THU HANG.

Moi model search rieng trong index cua no -> hai danh sach top-K doc lap. Buoc
gop (RRF) la viec cua fusion.py; o day chi tra ve ranked list dung dinh dang.

K = 500-1000 (config). IndexFlatIP quet toan bo bat ke K, nen K lon gan nhu
mien phi - chi ton them o buoc sort.

Diem so tra ve la cosine, VI CA index lan query deu da normalize L2. Hai thang
diem cua hai model KHONG so sanh truc tiep duoc voi nhau - do chinh la ly do
tang A gop theo THU HANG chu khong theo diem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aic.manifest import KeyframeEntry, assert_alignment, load_manifest
from aic.retrieval.encode_query import MODEL_KEYS
from aic.store import faiss_store


@dataclass(frozen=True)
class Hit:
    """Mot ket qua trong ranked list cua MOT model."""

    idx: int      # row cua FAISS == idx trong manifest
    score: float  # cosine
    rank: int     # bat dau tu 1 - dau vao cua RRF


def search_index(index, query_vector: np.ndarray, k: int) -> list[Hit]:
    """Search mot index, tra ve list[Hit] da sap xep giam dan theo diem.

    FAISS tra ve -1 cho o trong khi k > ntotal; nhung o do bi loai bo chu khong
    duoc gan rank gia.
    """
    if k < 1:
        raise ValueError(f"k phai >= 1, nhan {k}")

    scores, ids = faiss_store.search(index, query_vector, k)
    hits: list[Hit] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        hits.append(Hit(idx=int(idx), score=float(score), rank=len(hits) + 1))
    return hits


class IndexBundle:
    """Hai FAISS index + manifest, da kiem tra khop nhau.

    Kiem tra bat bien ID (A.6) NGAY LUC LOAD chu khong doi den luc tra ket qua sai:
    ntotal cua ca hai index phai bang so dong manifest.
    """

    def __init__(self, indexes: dict[str, Any], entries: list[KeyframeEntry]):
        missing = set(MODEL_KEYS) - set(indexes)
        if missing:
            raise ValueError(f"Thieu index cho model: {sorted(missing)}")

        self.indexes = indexes
        self.entries = entries
        self._frame_lookup: dict[tuple[str, int], int] | None = None
        assert_alignment(
            len(entries), {key: int(index.ntotal) for key, index in indexes.items()}
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "IndexBundle":
        entries = load_manifest(cfg.paths.manifest)
        indexes = {
            "clip": faiss_store.load_index(cfg.paths.faiss_clip),
            "siglip2": faiss_store.load_index(cfg.paths.faiss_siglip),
        }
        return cls(indexes, entries)

    @property
    def ntotal(self) -> int:
        return len(self.entries)

    def dims(self) -> dict[str, int]:
        return {key: faiss_store.index_dim(index) for key, index in self.indexes.items()}

    def search(self, query_vectors: dict[str, np.ndarray], k: int) -> dict[str, list[Hit]]:
        """Search ca hai model. Tra ve {model_key: ranked list}.

        Vector cua model nao phai vao dung index cua model do - so chieu khac nhau
        nen nham se loi ngay, nhung van kiem tra tuong minh cho ro rang.
        """
        results: dict[str, list[Hit]] = {}
        for key in MODEL_KEYS:
            if key not in query_vectors:
                raise ValueError(f"Thieu vector query cho model '{key}'")
            index = self.indexes[key]
            vector = np.asarray(query_vectors[key], dtype=np.float32)
            expected = faiss_store.index_dim(index)
            if vector.reshape(-1).shape[0] != expected:
                raise ValueError(
                    f"'{key}': query {vector.reshape(-1).shape[0]} chieu nhung index "
                    f"{expected} chieu. Co the da encode bang model khac luc index."
                )
            results[key] = search_index(index, vector, k)
        return results

    def entry(self, idx: int) -> KeyframeEntry:
        return self.entries[idx]

    @property
    def frame_lookup(self) -> dict[tuple[str, int], int]:
        """(video_id, frame_idx) -> idx cua manifest. Dung cho hard filter (tang B).

        SQLite chi biet (video_id, frame_idx); FAISS chi biet idx. Day la cau noi
        giua hai ben. Dung lazy vi chi query nao co dieu kien OCR moi can toi.
        """
        if self._frame_lookup is None:
            self._frame_lookup = {
                (e.video_id, e.frame_idx): e.idx for e in self.entries
            }
        return self._frame_lookup

    def idx_of(self, video_id: str, frame_idx: int) -> int | None:
        return self.frame_lookup.get((video_id, int(frame_idx)))

    def hydrate(self, hits: Sequence[Hit]) -> list[dict[str, Any]]:
        """Gan metadata manifest vao ket qua de hien thi / xuat file."""
        out = []
        for hit in hits:
            entry = self.entries[hit.idx]
            out.append(
                {
                    "idx": hit.idx,
                    "video_id": entry.video_id,
                    "frame_idx": entry.frame_idx,
                    "pts_time": entry.pts_time,
                    "path": entry.path,
                    "score": hit.score,
                    "rank": hit.rank,
                }
            )
        return out


def index_files_ready(cfg: Any) -> bool:
    return all(
        Path(p).exists()
        for p in (cfg.paths.manifest, cfg.paths.faiss_clip, cfg.paths.faiss_siglip)
    )
