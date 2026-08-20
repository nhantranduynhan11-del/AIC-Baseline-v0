"""Test nhan dang MP4 tai cat cut - dung dung truong hop gap tren vast.ai.

Xay dung file MP4 toi gian bang tay: MP4 la chuoi box, moi box mo dau bang
4 byte kich thuoc (big-endian) + 4 byte loai.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from aic.preprocess import shot_detect as sd


def box(box_type: bytes, payload: bytes = b"") -> bytes:
    size = 8 + len(payload)
    return size.to_bytes(4, "big") + box_type + payload


def box64(box_type: bytes, payload: bytes = b"") -> bytes:
    """Box dung kich thuoc 64-bit: truong 32-bit bang 1, kich thuoc that nam sau."""
    size = 16 + len(payload)
    return (1).to_bytes(4, "big") + box_type + size.to_bytes(8, "big") + payload


FTYP = box(b"ftyp", b"isom" + b"\x00" * 8)


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_bytes(data)
    return p


class TestFileDayDu:
    def test_ftyp_mdat_moov(self, tmp_path):
        p = write(tmp_path, "ok.mp4", FTYP + box(b"mdat", b"x" * 4000) + box(b"moov", b"y" * 500))
        sd.check_video_readable(p)

    def test_moov_nam_truoc_mdat_faststart(self, tmp_path):
        p = write(tmp_path, "fs.mp4", FTYP + box(b"moov", b"y" * 500) + box(b"mdat", b"x" * 4000))
        sd.check_video_readable(p)

    def test_box_kich_thuoc_64_bit(self, tmp_path):
        p = write(tmp_path, "big.mp4", FTYP + box64(b"mdat", b"x" * 4000) + box(b"moov", b"y" * 500))
        sd.check_video_readable(p)

    def test_box_kich_thuoc_0_keo_den_het_file(self, tmp_path):
        """size = 0 nghia la box keo den cuoi file - hop le voi mdat."""
        data = FTYP + box(b"moov", b"y" * 500) + (0).to_bytes(4, "big") + b"mdat" + b"x" * 4000
        sd.check_video_readable(write(tmp_path, "z.mp4", data))


class TestFileCatCut:
    def test_thieu_moov(self, tmp_path):
        """Dung trieu chung tren vast.ai: ftyp con nguyen, moov o cuoi da mat."""
        p = write(tmp_path, "cut.mp4", FTYP + box(b"mdat", b"x" * 24_000))
        with pytest.raises(ValueError, match="khong co box 'moov'"):
            sd.check_video_readable(p)

    def test_thong_bao_neu_ro_cach_xu_ly(self, tmp_path):
        p = write(tmp_path, "cut.mp4", FTYP + box(b"mdat", b"x" * 24_000))
        with pytest.raises(ValueError) as exc:
            sd.check_video_readable(p)
        assert "cat cut" in str(exc.value)
        assert "wget -c" in str(exc.value)
        assert "ftyp" in str(exc.value) and "mdat" in str(exc.value)   # liet ke box da thay

    def test_box_khai_bao_dai_hon_file(self, tmp_path):
        """mdat khai 100k byte nhung file chi con 5k - cat cut giua box."""
        header = (100_000).to_bytes(4, "big") + b"mdat"
        p = write(tmp_path, "t.mp4", FTYP + header + b"x" * 5_000)
        with pytest.raises(ValueError, match="CAT CUT"):
            sd.check_video_readable(p)


class TestKhongApNhamDinhDangKhac:
    def test_mkv_khong_bi_kiem_tra_moov(self, tmp_path):
        p = write(tmp_path, "a.mkv", b"\x1a\x45\xdf\xa3" + b"x" * 5000)
        sd.check_video_readable(p)

    def test_avi_khong_bi_kiem_tra_moov(self, tmp_path):
        p = write(tmp_path, "b.avi", b"RIFF" + b"\x00" * 4 + b"AVI " + b"x" * 5000)
        sd.check_video_readable(p)


class TestDuyetBox:
    def test_liet_ke_dung_thu_tu_va_kich_thuoc(self, tmp_path):
        p = write(tmp_path, "x.mp4", FTYP + box(b"free", b"z" * 100) + box(b"moov", b"y" * 200))
        boxes = list(sd.iter_mp4_boxes(p, p.stat().st_size))
        assert [t for t, _, _ in boxes] == [b"ftyp", b"free", b"moov"]
        assert [s for _, _, s in boxes] == [20, 108, 208]
        assert [o for _, o, _ in boxes] == [0, 20, 128]

    def test_header_hong_thi_dung_lai_khong_treo(self, tmp_path):
        """box_size < 8 la vo nghia; phai dung chu khong lap vo han."""
        p = write(tmp_path, "b.mp4", FTYP + (3).to_bytes(4, "big") + b"junk" + b"x" * 100)
        assert [t for t, _, _ in sd.iter_mp4_boxes(p, p.stat().st_size)] == [b"ftyp"]
