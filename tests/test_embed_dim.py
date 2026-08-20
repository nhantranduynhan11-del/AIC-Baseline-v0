"""Test xac dinh so chieu embedding.

Nguon goc: SigLIP2 dung vision tower cua timm (`TimmModel`), lop do khong co
`visual.output_dim` nhu `VisionTransformer` cua open_clip -> AttributeError sau
khi da nap model xong (11 phut tren vast.ai).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from aic.models import embed_dim as ed


class FakeVisual:
    def __init__(self, output_dim=None):
        if output_dim is not None:
            self.output_dim = output_dim


class FakeModel:
    def __init__(self, output_dim=None):
        self.visual = FakeVisual(output_dim)


class TimmLikeModel:
    """Giong TimmModel: truy cap output_dim nem AttributeError."""

    class _Visual:
        def __getattr__(self, item):
            raise AttributeError(f"'TimmModel' object has no attribute '{item}'")

    visual = _Visual()


class TestTuThuocTinh:
    def test_lay_duoc_khi_co_output_dim(self):
        assert ed.dim_from_attribute(FakeModel(768)) == 768

    def test_none_khi_khong_co(self):
        assert ed.dim_from_attribute(FakeModel()) is None

    def test_none_khi_thuoc_tinh_nem_loi(self):
        """Dung ca timm: getattr nem AttributeError chu khong tra None."""
        assert ed.dim_from_attribute(TimmLikeModel()) is None

    def test_none_khi_model_khong_co_visual(self):
        assert ed.dim_from_attribute(object()) is None

    def test_bo_qua_gia_tri_vo_ly(self):
        assert ed.dim_from_attribute(FakeModel(0)) is None
        assert ed.dim_from_attribute(FakeModel(-5)) is None


class TestTuConfig:
    def test_ten_khong_ton_tai_thi_none(self):
        assert ed.dim_from_config("khong-co-model-nao-ten-nay") is None

    def test_khong_cai_open_clip_van_khong_no(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "open_clip", None)
        assert ed.dim_from_config("ViT-L-14-quickgelu") is None


class TestThuTuUuTien:
    def test_thuoc_tinh_duoc_uu_tien(self):
        assert ed.resolve_embed_dim(FakeModel(768), "ViT-L-14-quickgelu") == 768

    def test_roi_ve_probe_khi_hai_cach_dau_that_bai(self):
        called = []

        def probe():
            called.append(1)
            return 1024

        assert ed.resolve_embed_dim(TimmLikeModel(), "ten-la", probe=probe) == 1024
        assert called == [1]

    def test_khong_goi_probe_neu_da_co_ket_qua(self):
        def probe():
            raise AssertionError("khong duoc goi")

        assert ed.resolve_embed_dim(FakeModel(768), "x", probe=probe) == 768

    def test_raise_khi_khong_cach_nao_ra(self):
        with pytest.raises(RuntimeError, match="Khong xac dinh duoc so chieu"):
            ed.resolve_embed_dim(TimmLikeModel(), "ten-la")

    def test_probe_tra_ve_0_cung_bi_coi_la_that_bai(self):
        with pytest.raises(RuntimeError):
            ed.resolve_embed_dim(TimmLikeModel(), "ten-la", probe=lambda: 0)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("open_clip") is None,
    reason="chua cai open_clip tren may nay",
)
class TestVoiOpenClipThat:
    """Xac minh voi registry that - chay duoc khi da cai open_clip (khong can weights)."""

    def test_clip_768(self):
        assert ed.dim_from_config("ViT-L-14-quickgelu") == 768

    def test_siglip2_1024(self):
        assert ed.dim_from_config("ViT-L-16-SigLIP2-256") == 1024
