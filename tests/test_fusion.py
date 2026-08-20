"""Test B.1 buoc 4 - RRF.

Khong can torch lan faiss: RRF chi lam viec tren ranked list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from aic.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from aic.retrieval.search import Hit


def ranked(*idxs: int) -> list[Hit]:
    """Ranked list gia: rank 1..n theo dung thu tu truyen vao."""
    return [Hit(idx=idx, score=1.0 - i * 0.01, rank=i + 1) for i, idx in enumerate(idxs)]


class TestCongThuc:
    def test_diem_dung_cong_thuc_rrf(self):
        # idx 7: rank 1 o clip, rank 3 o siglip2
        out = reciprocal_rank_fusion({"clip": ranked(7, 1), "siglip2": ranked(1, 2, 7)}, k=60)
        top = {h.idx: h.score for h in out}
        assert top[7] == pytest.approx(1 / 61 + 1 / 63)
        assert top[1] == pytest.approx(1 / 62 + 1 / 61)

    def test_k_mac_dinh_la_60(self):
        assert DEFAULT_RRF_K == 60

    def test_khong_co_trong_so_cho_tung_model(self):
        """Hai model doi xung: doi cho hai danh sach khong duoc lam doi ket qua."""
        a, b = ranked(1, 2, 3), ranked(3, 2, 1)
        out1 = reciprocal_rank_fusion({"clip": a, "siglip2": b})
        out2 = reciprocal_rank_fusion({"clip": b, "siglip2": a})
        assert {h.idx: h.score for h in out1} == {h.idx: h.score for h in out2}


class TestKhongGanRankGia:
    def test_chi_xuat_hien_o_mot_model_thi_chi_cong_mot_lan(self):
        out = reciprocal_rank_fusion({"clip": ranked(5), "siglip2": ranked(9)}, k=60)
        scores = {h.idx: h.score for h in out}
        assert scores[5] == pytest.approx(1 / 61)
        assert scores[9] == pytest.approx(1 / 61)

    def test_ranks_chi_ghi_nguon_that_su_co_ket_qua(self):
        out = reciprocal_rank_fusion({"clip": ranked(5, 9), "siglip2": ranked(9)})
        by_idx = {h.idx: h.ranks for h in out}
        assert by_idx[5] == {"clip": 1}
        assert by_idx[9] == {"clip": 2, "siglip2": 1}

    def test_xuat_hien_o_ca_hai_thi_thang_ket_qua_chi_o_mot_ben(self):
        """Ban chat RRF: dong thuan giua cac model quan trong hon thu hang cao le loi."""
        out = reciprocal_rank_fusion(
            {"clip": ranked(1, 2), "siglip2": ranked(3, 2)}, k=60
        )
        assert out[0].idx == 2   # rank 2 + rank 2 > rank 1 don le


class TestTongQuatTheoN:
    def test_mot_danh_sach(self):
        out = reciprocal_rank_fusion({"clip": ranked(4, 5, 6)})
        assert [h.idx for h in out] == [4, 5, 6]

    def test_ba_danh_sach_khong_phai_sua_ham(self):
        out = reciprocal_rank_fusion({
            "clip": ranked(1, 2),
            "siglip2": ranked(2, 1),
            "beit3": ranked(2, 3),
        }, k=60)
        assert out[0].idx == 2
        assert out[0].score == pytest.approx(1 / 62 + 1 / 61 + 1 / 61)
        assert set(out[0].ranks) == {"clip", "siglip2", "beit3"}

    def test_khong_co_danh_sach_nao(self):
        with pytest.raises(ValueError, match="ranked list"):
            reciprocal_rank_fusion({})


class TestXepHangVaCat:
    def test_rank_dau_ra_lien_tuc_tu_1(self):
        out = reciprocal_rank_fusion({"clip": ranked(9, 8, 7), "siglip2": ranked(7, 8, 9)})
        assert [h.rank for h in out] == [1, 2, 3]

    def test_top_n_cat_dung_so_luong(self):
        out = reciprocal_rank_fusion({"clip": ranked(*range(500))}, top_n=100)
        assert len(out) == 100
        assert [h.rank for h in out] == list(range(1, 101))

    def test_top_n_lon_hon_so_ket_qua_thi_giu_het(self):
        assert len(reciprocal_rank_fusion({"clip": ranked(1, 2)}, top_n=100)) == 2

    def test_hoa_diem_thi_uu_tien_idx_nho_de_ket_qua_on_dinh(self):
        out = reciprocal_rank_fusion({"clip": ranked(9, 3), "siglip2": ranked(3, 9)})
        assert [h.idx for h in out] == [3, 9]   # cung diem -> idx nho truoc

    def test_diem_giam_dan(self):
        out = reciprocal_rank_fusion({"clip": ranked(*range(20)), "siglip2": ranked(*range(19, -1, -1))})
        scores = [h.score for h in out]
        assert scores == sorted(scores, reverse=True)


class TestDauVao:
    def test_chap_nhan_so_nguyen_thuan(self):
        out = reciprocal_rank_fusion({"a": [10, 20], "b": [20, 10]}, k=60)
        assert {h.idx for h in out} == {10, 20}

    def test_chan_ranked_list_da_bi_cat_truoc_khi_gop(self):
        full = ranked(1, 2, 3, 4, 5)
        with pytest.raises(ValueError, match="nguyen ven"):
            reciprocal_rank_fusion({"clip": full[2:]})

    def test_k_khong_hop_le(self):
        with pytest.raises(ValueError, match="k phai"):
            reciprocal_rank_fusion({"clip": ranked(1)}, k=0)
