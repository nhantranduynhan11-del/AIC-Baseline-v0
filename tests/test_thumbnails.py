"""Test sinh thumbnail — đặc biệt là chạy được KHI CHƯA CÓ manifest.

Kaggle cố tình không chạy --build-manifest (manifest phải sinh một lần trên máy
đã gom đủ mọi phần), nên thumbnail không được phụ thuộc vào nó.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.manifest import KeyframeEntry, write_manifest
from aic.preprocess import keyframe as kf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def thumbs_mod():
    spec = importlib.util.spec_from_file_location(
        "thumbs_script", ROOT / "scripts" / "05_thumbnails.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_video(root: Path, video_id: str, frames: list[int]) -> None:
    d = root / video_id
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / kf.KEYFRAME_EMB, np.zeros((len(frames), 4), dtype=np.float32))
    kf.write_keyframe_meta(d / kf.KEYFRAME_META, {
        "version": kf.KEYFRAME_META_VERSION, "video_id": video_id, "fps": 25.0,
        "sample_every": 8, "l2_threshold": 0.4, "clip_model": "x", "dim": 4,
        "n_sampled": len(frames) * 3, "n_keyframes": len(frames),
        "keyframes": [
            {"frame_idx": f, "shot_id": 1, "pts_time": f / 25.0, "path": f"{video_id}/{f}.jpg"}
            for f in frames
        ],
    })


class TestKhongCanManifest:
    def test_doc_tu_keyframes_json_khi_chua_co_manifest(self, thumbs_mod, tmp_path):
        keyframes = tmp_path / "keyframes"
        make_video(keyframes, "L21_V001", [0, 8, 16])
        make_video(keyframes, "L21_V002", [0, 24])

        paths, source = thumbs_mod.list_keyframe_paths(tmp_path / "khong-co.csv", keyframes)

        assert paths == ["L21_V001/0.jpg", "L21_V001/8.jpg", "L21_V001/16.jpg",
                         "L21_V002/0.jpg", "L21_V002/24.jpg"]
        assert "chua co manifest" in source

    def test_uu_tien_manifest_khi_da_co(self, thumbs_mod, tmp_path):
        keyframes = tmp_path / "keyframes"
        make_video(keyframes, "L21_V001", [0, 8])

        manifest = tmp_path / "manifest.csv"
        write_manifest(manifest, [
            KeyframeEntry(-1, "L21_V001", 0, 0.0, "L21_V001/0.jpg"),
            KeyframeEntry(-1, "L21_V001", 8, 0.32, "L21_V001/8.jpg"),
        ])

        paths, source = thumbs_mod.list_keyframe_paths(manifest, keyframes)
        assert paths == ["L21_V001/0.jpg", "L21_V001/8.jpg"]
        assert "manifest" in source and "chua co" not in source

    def test_khong_co_gi_thi_tra_ve_rong(self, thumbs_mod, tmp_path):
        paths, _ = thumbs_mod.list_keyframe_paths(tmp_path / "x.csv", tmp_path / "trong")
        assert paths == []

    def test_hai_nguon_cho_ra_cung_tap_duong_dan(self, thumbs_mod, tmp_path):
        """Đường vòng và đường chính phải trùng nhau, nếu không thumbnail sẽ lệch ảnh."""
        keyframes = tmp_path / "keyframes"
        make_video(keyframes, "L21_V001", [0, 8, 16])
        make_video(keyframes, "L21_V002", [0, 24])

        via_glob, _ = thumbs_mod.list_keyframe_paths(tmp_path / "khong-co.csv", keyframes)

        n, _ = kf.build_manifest(keyframes, tmp_path / "m.csv", tmp_path / "e.npy")
        via_manifest, _ = thumbs_mod.list_keyframe_paths(tmp_path / "m.csv", keyframes)

        assert n == len(via_glob)
        assert via_glob == via_manifest


class TestSinhAnh:
    def test_thumbnail_nho_hon_va_giu_ti_le(self, thumbs_mod, tmp_path):
        from PIL import Image

        src = tmp_path / "big.jpg"
        Image.new("RGB", (1920, 1080), (200, 30, 30)).save(src, "JPEG", quality=95)
        dst = tmp_path / "out" / "small.jpg"

        thumbs_mod.make_thumb(src, dst, size=320, quality=80)

        with Image.open(dst) as img:
            assert img.size == (320, 180)          # giữ đúng tỉ lệ 16:9
        assert dst.stat().st_size < src.stat().st_size
