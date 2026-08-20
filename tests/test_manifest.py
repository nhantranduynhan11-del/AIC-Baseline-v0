"""Test bat bien ID (A.6) - phan de vo nhat neu khong co test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from aic.manifest import (
    KeyframeEntry, assert_alignment, load_manifest, write_manifest,
)
from aic.text_norm import normalize_vi


def _entries(n=5):
    return [
        KeyframeEntry(idx=-1, video_id=f"L01_V00{i // 3}", frame_idx=i * 8,
                      pts_time=i * 0.32, path=f"L01_V00{i // 3}/{i * 8}.jpg")
        for i in range(n)
    ]


def test_write_gan_lai_idx_lien_tuc(tmp_path):
    p = tmp_path / "manifest.csv"
    n = write_manifest(p, _entries(5))
    assert n == 5
    assert [e.idx for e in load_manifest(p)] == [0, 1, 2, 3, 4]


def test_roundtrip_giu_nguyen_thu_tu(tmp_path):
    p = tmp_path / "manifest.csv"
    write_manifest(p, _entries(4))
    got = load_manifest(p)
    assert [e.frame_idx for e in got] == [0, 8, 16, 24]
    assert got[2].video_id == "L01_V000"


def test_bat_loi_idx_lech(tmp_path):
    p = tmp_path / "manifest.csv"
    write_manifest(p, _entries(3))
    lines = p.read_text(encoding="utf-8").splitlines()
    lines[2] = "7" + lines[2][1:]          # pha idx dong thu 2
    p.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="lech ID"):
        load_manifest(p)


def test_assert_alignment():
    assert_alignment(100, {"clip": 100, "siglip2": 100})
    with pytest.raises(AssertionError, match="siglip2"):
        assert_alignment(100, {"clip": 100, "siglip2": 99})


@pytest.mark.parametrize("raw,expected", [
    ("Đường Điện Biên Phủ", "duong dien bien phu"),
    ("ĐỎ", "do"),
    ("Bình Thạnh", "binh thanh"),
    ("", ""),
])
def test_normalize_vi_xu_ly_chu_d(raw, expected):
    assert normalize_vi(raw) == expected
