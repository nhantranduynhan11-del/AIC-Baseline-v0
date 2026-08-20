"""Test B.1 end-to-end (khong can torch): encoder gia + FAISS that."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.manifest import KeyframeEntry
from aic.retrieval import pipeline
from aic.retrieval.search import IndexBundle
from aic.store import faiss_store

N = 300


def normalized(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.rand(rows, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeEncoder:
    """Tra ve dung vector cua mot row da index -> biet truoc ket qua dung."""

    def __init__(self, clip_emb, siglip_emb, target):
        self.clip_emb, self.siglip_emb, self.target = clip_emb, siglip_emb, target

    def encode(self, text=None, image_rgb=None):
        return {"clip": self.clip_emb[self.target], "siglip2": self.siglip_emb[self.target]}


@pytest.fixture
def setup():
    clip_emb = normalized(N, 8, 1)
    siglip_emb = normalized(N, 16, 2)
    bundle = IndexBundle(
        {
            "clip": faiss_store.build_flat_ip(clip_emb),
            "siglip2": faiss_store.build_flat_ip(siglip_emb),
        },
        [KeyframeEntry(i, f"V{i // 50:03d}", i * 8, i * 0.32, f"V{i // 50:03d}/{i * 8}.jpg")
         for i in range(N)],
    )
    return bundle, clip_emb, siglip_emb


def test_ket_qua_dung_nam_dau_khi_ca_hai_model_dong_thuan(setup):
    bundle, clip_emb, siglip_emb = setup
    hits = pipeline.search(
        bundle, FakeEncoder(clip_emb, siglip_emb, target=137),
        text="x", top_k_per_model=100, top_n=100,
    )
    assert hits[0].idx == 137
    assert hits[0].ranks == {"clip": 1, "siglip2": 1}


def test_cat_dung_top_100(setup):
    bundle, clip_emb, siglip_emb = setup
    hits = pipeline.search(
        bundle, FakeEncoder(clip_emb, siglip_emb, target=5),
        text="x", top_k_per_model=250, top_n=100,
    )
    assert len(hits) == 100
    assert [h.rank for h in hits] == list(range(1, 101))


def test_K_lon_hon_100_van_cat_ve_100(setup):
    bundle, clip_emb, siglip_emb = setup
    hits = pipeline.search(
        bundle, FakeEncoder(clip_emb, siglip_emb, target=5),
        text="x", top_k_per_model=N, top_n=100,
    )
    assert len(hits) == 100


def test_hydrate_gan_metadata_va_rank_tung_model(setup):
    bundle, clip_emb, siglip_emb = setup
    hits = pipeline.search(
        bundle, FakeEncoder(clip_emb, siglip_emb, target=137),
        text="x", top_k_per_model=50, top_n=5,
    )
    rows = pipeline.hydrate(bundle, hits)
    assert rows[0]["video_id"] == "V002" and rows[0]["frame_idx"] == 137 * 8
    assert rows[0]["ranks"] == {"clip": 1, "siglip2": 1}
    assert [r["rank"] for r in rows] == [1, 2, 3, 4, 5]


def test_search_from_config_lay_tham_so_tu_config(setup):
    bundle, clip_emb, siglip_emb = setup

    class Cfg:
        retrieval = type("R", (), {"top_k_per_model": 40, "rrf_k": 60, "final_top_n": 7})()

    hits = pipeline.search_from_config(
        bundle, FakeEncoder(clip_emb, siglip_emb, target=1), Cfg(), text="x"
    )
    assert len(hits) == 7
