"""Test B.1 buoc 5 - hard filter tang B. Dung FAISS + SQLite that, khong can torch."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.manifest import KeyframeEntry
from aic.retrieval import pipeline
from aic.retrieval.filters import apply_filter, describe, ocr_allowed_idxs
from aic.retrieval.fusion import FusedHit
from aic.retrieval.search import IndexBundle
from aic.store import faiss_store
from aic.store import sqlite_store as store

N = 12


def normalized(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.rand(rows, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def bundle():
    """12 keyframe: video V000 (idx 0-5, frame 0,8,...,40), V001 (idx 6-11)."""
    entries = [
        KeyframeEntry(i, f"V{i // 6:03d}", (i % 6) * 8, (i % 6) * 0.32,
                      f"V{i // 6:03d}/{(i % 6) * 8}.jpg")
        for i in range(N)
    ]
    return IndexBundle(
        {
            "clip": faiss_store.build_flat_ip(normalized(N, 8, 1)),
            "siglip2": faiss_store.build_flat_ip(normalized(N, 16, 2)),
        },
        entries,
    )


@pytest.fixture
def conn(tmp_path):
    c = store.open_db(tmp_path / "m.db")
    store.insert_ocr(c, [
        ("V000", 8, "Đường Điện Biên Phủ", 0.91),   # -> idx 1
        ("V000", 8, "TIN NÓNG", 0.75),              # -> idx 1
        ("V001", 24, "Điện Biên Phủ hôm nay", 0.55),  # -> idx 9
        ("V001", 40, "dien bien", 0.20),            # -> idx 11, confidence thap
        ("V999", 0, "Điện Biên Phủ", 0.90),         # khong co trong manifest
    ])
    c.commit()
    return c


def hits(*idxs: int) -> list[FusedHit]:
    return [FusedHit(idx=i, score=1.0 - n * 0.01, rank=n + 1, ranks={"clip": n + 1})
            for n, i in enumerate(idxs)]


class TestOcrAllowedIdxs:
    def test_anh_xa_video_frame_sang_idx_manifest(self, conn, bundle):
        result = ocr_allowed_idxs(conn, bundle, "dien bien phu", min_confidence=0.3)
        assert result.allowed == {1, 9}

    def test_dem_keyframe_khong_co_trong_manifest(self, conn, bundle):
        result = ocr_allowed_idxs(conn, bundle, "dien bien phu", min_confidence=0.3)
        assert result.n_unknown == 1        # V999 frame 0
        assert "lech" in describe(result)

    def test_loc_theo_confidence(self, conn, bundle):
        assert ocr_allowed_idxs(conn, bundle, "dien bien", min_confidence=0.3).allowed == {1, 9}
        assert ocr_allowed_idxs(conn, bundle, "dien bien", min_confidence=0.1).allowed == {1, 9, 11}

    def test_phrase_va_khong_phrase(self, conn, bundle):
        assert ocr_allowed_idxs(conn, bundle, "phu duong").allowed == set()
        assert ocr_allowed_idxs(conn, bundle, "phu duong", phrase=False).allowed == {1}

    def test_khong_khop_gi_thi_rong(self, conn, bundle):
        result = ocr_allowed_idxs(conn, bundle, "khong ton tai dau")
        assert result.allowed == set() and not result

    def test_nhieu_dong_tren_cung_keyframe_chi_ra_mot_idx(self, conn, bundle):
        result = ocr_allowed_idxs(conn, bundle, "tin nong", min_confidence=0.3)
        assert result.allowed == {1} and result.n_rows == 1


class TestApplyFilter:
    def test_giu_thu_tu_va_danh_so_lai_rank(self):
        out = apply_filter(hits(5, 3, 9, 1), {3, 1})
        assert [h.idx for h in out] == [3, 1]
        assert [h.rank for h in out] == [1, 2]      # khong phai [2, 4]

    def test_giu_nguyen_diem_va_rank_tung_model(self):
        out = apply_filter(hits(5, 3), {3})
        assert out[0].score == pytest.approx(0.99)
        assert out[0].ranks == {"clip": 2}

    def test_allowed_none_thi_khong_loc(self):
        original = hits(5, 3, 9)
        assert apply_filter(original, None) == original

    def test_allowed_rong_thi_khong_con_gi(self):
        assert apply_filter(hits(5, 3), set()) == []


class TestThuTuLocTruocKhiCat:
    """Loi de mac nhat: cat top-N truoc roi moi loc."""

    def test_loc_truoc_cat_sau_van_du_so_luong(self, conn, bundle):
        class FakeEncoder:
            def encode(self, text=None, image_rgb=None):
                return {"clip": normalized(1, 8, 3)[0], "siglip2": normalized(1, 16, 4)[0]}

        # Chi 2 keyframe thoa dieu kien OCR. Neu cat top_n=2 TRUOC khi loc thi gan
        # nhu chac chan ra 0-1 ket qua; loc truoc thi luon ra dung 2.
        out = pipeline.search(
            bundle, FakeEncoder(), text="x",
            top_k_per_model=N, top_n=2, allowed_idxs={1, 9},
        )
        assert len(out) == 2
        assert {h.idx for h in out} == {1, 9}
        assert [h.rank for h in out] == [1, 2]

    def test_search_with_ocr_tra_ve_ca_so_lieu_filter(self, conn, bundle):
        class FakeEncoder:
            def encode(self, text=None, image_rgb=None):
                return {"clip": normalized(1, 8, 5)[0], "siglip2": normalized(1, 16, 6)[0]}

        out, result = pipeline.search_with_ocr(
            bundle, FakeEncoder(), conn,
            ocr_query="dien bien phu", ocr_min_confidence=0.3,
            text="x", top_k_per_model=N, top_n=100,
        )
        assert {h.idx for h in out} == {1, 9}
        assert result.n_frames == 3 and result.n_unknown == 1

    def test_khong_co_dieu_kien_ocr_thi_tra_ve_binh_thuong(self, bundle):
        class FakeEncoder:
            def encode(self, text=None, image_rgb=None):
                return {"clip": normalized(1, 8, 7)[0], "siglip2": normalized(1, 16, 8)[0]}

        out = pipeline.search(bundle, FakeEncoder(), text="x", top_k_per_model=N, top_n=100)
        assert len(out) == N
