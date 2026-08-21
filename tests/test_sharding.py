"""Test chia phần cho nhiều GPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from aic.sharding import parse_shard, select_shard


class TestParse:
    @pytest.mark.parametrize("text,expected", [
        ("0/2", (0, 2)), ("1/2", (1, 2)), ("3/4", (3, 4)), (" 0/1 ", (0, 1)),
    ])
    def test_hop_le(self, text, expected):
        assert parse_shard(text) == expected

    @pytest.mark.parametrize("text", ["2", "abc", "a/b", ""])
    def test_sai_cu_phap(self, text):
        with pytest.raises(ValueError, match="I/N"):
            parse_shard(text)

    def test_chi_so_ngoai_pham_vi(self):
        with pytest.raises(ValueError, match="0..1"):
            parse_shard("2/2")

    def test_so_phan_khong_hop_le(self):
        with pytest.raises(ValueError, match="N phải"):
            parse_shard("0/0")


class TestSelect:
    ITEMS = list("abcdefghij")     # 10 phần tử

    def test_khong_chia_thi_giu_nguyen(self):
        assert select_shard(self.ITEMS, None) == self.ITEMS

    def test_hai_phan_roi_nhau_va_phu_kin(self):
        a = select_shard(self.ITEMS, "0/2")
        b = select_shard(self.ITEMS, "1/2")
        assert set(a) & set(b) == set()          # không chồng nhau
        assert sorted(a + b) == self.ITEMS       # không sót phần tử nào

    def test_chia_xen_ke_chu_khong_cat_khoi(self):
        """Xen kẽ để hai GPU không bị lệch tải khi video dài ngắn khác nhau."""
        assert select_shard(self.ITEMS, "0/2") == list("acegi")
        assert select_shard(self.ITEMS, "1/2") == list("bdfhj")

    def test_ba_phan(self):
        parts = [select_shard(self.ITEMS, f"{i}/3") for i in range(3)]
        assert sorted(sum(parts, [])) == self.ITEMS
        assert [len(p) for p in parts] == [4, 3, 3]      # lệch nhiều nhất 1

    def test_mot_phan_duy_nhat(self):
        assert select_shard(self.ITEMS, "0/1") == self.ITEMS

    def test_so_phan_nhieu_hon_so_phan_tu(self):
        assert select_shard(["x", "y"], "0/5") == ["x"]
        assert select_shard(["x", "y"], "2/5") == []

    def test_danh_sach_rong(self):
        assert select_shard([], "0/2") == []

    def test_giu_nguyen_thu_tu_trong_tung_phan(self):
        assert select_shard(list(range(20)), "1/4") == [1, 5, 9, 13, 17]
