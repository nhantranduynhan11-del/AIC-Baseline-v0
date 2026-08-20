"""Test A.1 - phan chay duoc ma khong can torch/GPU.

Phan can torch (build_model, predict_raw_frames, predictions_to_shots) chi
verify duoc tren vast.ai.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.preprocess import shot_detect as sd


def mp4_bytes(payload: int = 4096, with_moov: bool = True) -> bytes:
    """MP4 toi thieu hop le: ftyp + mdat + moov."""
    def box(box_type: bytes, data: bytes = b"") -> bytes:
        return (8 + len(data)).to_bytes(4, "big") + box_type + data

    out = box(b"ftyp", b"isom" + bytes(8)) + box(b"mdat", b"x" * payload)
    return out + box(b"moov", b"y" * 200) if with_moov else out


@pytest.fixture
def preds():
    return np.array([0.1, 0.2, 0.9, 0.1, 0.05, 0.1, 0.8, 0.2, 0.1], dtype=np.float32)


def test_scenes_to_shots_tinh_dung_thoi_gian_va_probability(preds):
    shots = sd._scenes_to_shots([[0, 1], [3, 5], [7, 8]], preds, fps=25.0)
    assert [s["shot_id"] for s in shots] == [1, 2, 3]
    assert shots[1] == {
        "shot_id": 2,
        "start_frame": 3,
        "end_frame": 5,
        "start_time": 0.12,
        "end_time": 0.2,
        "probability": pytest.approx(0.1, abs=1e-6),
    }


def test_scenes_to_shots_fps_le_bi_chan(preds):
    with pytest.raises(ValueError, match="fps"):
        sd._scenes_to_shots([[0, 1]], preds, fps=0.0)


def test_shot_json_roundtrip(tmp_path, preds):
    shots = sd._scenes_to_shots([[0, 8]], preds, fps=30.0)
    record = sd.make_record(Path("videos/L01_V001.mp4"), preds, 30.0, 0.5, shots)
    assert record["video_id"] == "L01_V001"
    assert record["n_frames"] == 9 and record["n_shots"] == 1

    p = tmp_path / "L01_V001.json"
    sd.write_shots(p, record)
    assert sd.read_shots(p) == record


def test_read_shots_chan_version_cu(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"version": 0, "shots": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        sd.read_shots(p)


def test_save_raw_giu_float32(tmp_path, preds):
    p = tmp_path / "L01_V001.npy"
    sd.save_raw(p, preds.astype(np.float64))
    loaded = np.load(p)
    assert loaded.dtype == np.float32
    np.testing.assert_allclose(loaded, preds, rtol=1e-6)


def test_find_videos_loc_dung_duoi_va_sap_xep(tmp_path):
    for name in ["b.mp4", "a.MP4", "c.txt", "sub/d.mkv", "e.jpg"]:
        f = tmp_path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    got = sd.find_videos(tmp_path, [".mp4", ".mkv"])
    assert [p.name for p in got] == ["a.MP4", "b.mp4", "d.mkv"]


class TestChanDoanLoiFfmpeg:
    """Loi ffmpeg phai noi ro no hong o dau, khong phai 'ffmpeg error'."""

    def test_file_khong_ton_tai(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Khong co file"):
            sd.check_video_readable(tmp_path / "khong-co.mp4")

    def test_file_rong(self, tmp_path):
        p = tmp_path / "e.mp4"
        p.write_bytes(b"")
        with pytest.raises(ValueError, match="file rong"):
            sd.check_video_readable(p)

    def test_file_qua_nho_gan_nhu_chac_chan_tai_loi(self, tmp_path):
        p = tmp_path / "t.mp4"
        p.write_bytes(b"x" * 200)
        with pytest.raises(ValueError, match="200 byte"):
            sd.check_video_readable(p)

    def test_file_hop_le_thi_khong_bao_gi(self, tmp_path):
        p = tmp_path / "ok.mp4"
        p.write_bytes(mp4_bytes())
        sd.check_video_readable(p)

    def test_thong_bao_loi_boc_duoc_stderr_cua_ffmpeg(self):
        class FakeFfmpegError(Exception):
            stderr = (b"[mov,mp4] moov atom not found\n"
                      b"data/videos/L21_V001.mp4: Invalid data found when processing input\n")

        msg = sd._ffmpeg_message(FakeFfmpegError(), "data/videos/L21_V001.mp4")
        assert "L21_V001.mp4" in msg
        assert "moov atom not found" in msg          # dong that su huu ich
        assert "Invalid data found" in msg

    def test_khong_co_stderr_thi_van_ra_thong_bao_doc_duoc(self):
        msg = sd._ffmpeg_message(Exception(), "a.mp4", tool="ffprobe")
        assert "ffprobe that bai voi a.mp4" in msg


class TestNhanDangFileKhongPhaiVideo:
    """Tai hong thuong tra ve trang HTML hoac con tro LFS thay vi video."""

    def _write(self, tmp_path, name, data):
        p = tmp_path / name
        p.write_bytes(data)
        return p

    def test_trang_html_bao_loi(self, tmp_path):
        p = self._write(tmp_path, "a.mp4", b"<!DOCTYPE html><html>quota exceeded</html>" + b" " * 2000)
        with pytest.raises(ValueError, match="khong phai file video"):
            sd.check_video_readable(p)

    def test_con_tro_git_lfs(self, tmp_path):
        p = self._write(tmp_path, "b.mp4",
                        b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n" + b" " * 2000)
        with pytest.raises(ValueError, match="con tro LFS"):
            sd.check_video_readable(p)

    def test_mp4_thieu_box_ftyp(self, tmp_path):
        p = self._write(tmp_path, "c.mp4", b"\x00\x00\x00\x20MOOV" + b"x" * 5000)
        with pytest.raises(ValueError, match="ftyp"):
            sd.check_video_readable(p)

    def test_mp4_hop_le_di_qua(self, tmp_path):
        p = self._write(tmp_path, "d.mp4", mp4_bytes(payload=8000))
        sd.check_video_readable(p)

    def test_mp4_co_ftyp_nhung_thieu_moov_bi_chan(self, tmp_path):
        """Đúng trường hợp gặp trên vast.ai - xem thêm tests/test_mp4_check.py."""
        p = self._write(tmp_path, "f.mp4", mp4_bytes(with_moov=False))
        with pytest.raises(ValueError, match="moov"):
            sd.check_video_readable(p)

    def test_mkv_khong_bi_ap_luat_ftyp(self, tmp_path):
        """Matroska dung magic khac; chi .mp4/.m4v/.mov moi kiem tra ftyp."""
        p = self._write(tmp_path, "e.mkv", b"\x1a\x45\xdf\xa3" + b"x" * 5000)
        sd.check_video_readable(p)
