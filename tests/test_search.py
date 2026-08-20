"""Test B.1 buoc 3 - search FAISS tra ve ranked list.

Chay duoc voi faiss-cpu, khong can torch (khong dung encoder that).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.manifest import KeyframeEntry
from aic.retrieval.search import Hit, IndexBundle, search_index
from aic.store import faiss_store


def normalized(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.rand(rows, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def entries(n: int) -> list[KeyframeEntry]:
    return [
        KeyframeEntry(i, f"L01_V{i // 3:03d}", i * 8, i * 0.32, f"L01_V{i // 3:03d}/{i * 8}.jpg")
        for i in range(n)
    ]


@pytest.fixture
def bundle():
    clip = faiss_store.build_flat_ip(normalized(10, 8, 1))
    siglip = faiss_store.build_flat_ip(normalized(10, 16, 2))
    return IndexBundle({"clip": clip, "siglip2": siglip}, entries(10))


class TestSearchIndex:
    def test_rank_bat_dau_tu_1_va_lien_tuc(self):
        emb = normalized(10, 8, 3)
        index = faiss_store.build_flat_ip(emb)
        hits = search_index(index, emb[0], k=5)
        assert [h.rank for h in hits] == [1, 2, 3, 4, 5]

    def test_diem_giam_dan(self):
        emb = normalized(20, 8, 4)
        index = faiss_store.build_flat_ip(emb)
        scores = [h.score for h in search_index(index, emb[7], k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_query_trung_vector_da_index_thi_dung_dau(self):
        emb = normalized(20, 8, 5)
        index = faiss_store.build_flat_ip(emb)
        top = search_index(index, emb[13], k=1)[0]
        assert top.idx == 13
        assert top.score == pytest.approx(1.0, abs=1e-4)

    def test_k_lon_hon_ntotal_khong_sinh_rank_gia(self):
        emb = normalized(4, 8, 6)
        index = faiss_store.build_flat_ip(emb)
        hits = search_index(index, emb[0], k=100)
        assert len(hits) == 4          # khong padding, khong idx = -1
        assert all(h.idx >= 0 for h in hits)

    def test_k_khong_hop_le(self):
        index = faiss_store.build_flat_ip(normalized(4, 8, 7))
        with pytest.raises(ValueError, match="k phai"):
            search_index(index, normalized(1, 8, 8)[0], k=0)

    def test_query_chua_normalize_bi_chan(self):
        index = faiss_store.build_flat_ip(normalized(4, 8, 9))
        with pytest.raises(ValueError, match="normalize"):
            search_index(index, normalized(1, 8, 10)[0] * 5.0, k=2)


class TestIndexBundle:
    def test_search_ca_hai_model_tra_ve_hai_ranked_list(self, bundle):
        result = bundle.search({"clip": normalized(1, 8, 11)[0],
                                "siglip2": normalized(1, 16, 12)[0]}, k=5)
        assert set(result) == {"clip", "siglip2"}
        assert all(len(hits) == 5 for hits in result.values())

    def test_chan_lech_so_chieu_giua_query_va_index(self, bundle):
        with pytest.raises(ValueError, match="chieu"):
            bundle.search({"clip": normalized(1, 16, 13)[0],      # nham vector SigLIP2
                           "siglip2": normalized(1, 16, 14)[0]}, k=5)

    def test_thieu_vector_mot_model(self, bundle):
        with pytest.raises(ValueError, match="siglip2"):
            bundle.search({"clip": normalized(1, 8, 15)[0]}, k=5)

    def test_bat_lech_ntotal_voi_manifest_ngay_luc_load(self):
        clip = faiss_store.build_flat_ip(normalized(10, 8, 16))
        siglip = faiss_store.build_flat_ip(normalized(9, 16, 17))   # thieu 1 vector
        with pytest.raises(AssertionError, match="siglip2"):
            IndexBundle({"clip": clip, "siglip2": siglip}, entries(10))

    def test_thieu_han_mot_index(self):
        clip = faiss_store.build_flat_ip(normalized(10, 8, 18))
        with pytest.raises(ValueError, match="siglip2"):
            IndexBundle({"clip": clip}, entries(10))

    def test_hydrate_gan_dung_metadata_manifest(self, bundle):
        hits = [Hit(idx=4, score=0.9, rank=1), Hit(idx=0, score=0.8, rank=2)]
        rows = bundle.hydrate(hits)
        assert rows[0]["video_id"] == "L01_V001" and rows[0]["frame_idx"] == 32
        assert rows[1]["video_id"] == "L01_V000" and rows[1]["rank"] == 2

    def test_dims(self, bundle):
        assert bundle.dims() == {"clip": 8, "siglip2": 16}
