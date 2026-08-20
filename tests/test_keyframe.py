"""Test A.2 - phan chay duoc ma khong can torch/cv2/GPU.

Phan can model (extract_video, iter_planned_frames) chi verify duoc tren vast.ai.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.manifest import load_manifest
from aic.preprocess import keyframe as kf


def unit(*values) -> np.ndarray:
    """Vector don vi 3 chieu de tinh khoang cach L2 bang tay."""
    v = np.array(values, dtype=np.float32)
    return v / np.linalg.norm(v)


class TestKeyframeSelector:
    def test_frame_dau_tien_cua_shot_luon_duoc_giu(self):
        s = kf.KeyframeSelector(0.4)
        s.start_shot()
        assert s.consider(unit(1, 0, 0)) is True

    def test_qua_giong_thi_loai(self):
        s = kf.KeyframeSelector(0.4)
        s.start_shot()
        s.consider(unit(1, 0, 0))
        # lech rat nho -> khoang cach ~0.1 < 0.4
        assert s.consider(unit(1, 0.1, 0)) is False

    def test_khac_du_nhieu_thi_giu(self):
        s = kf.KeyframeSelector(0.4)
        s.start_shot()
        s.consider(unit(1, 0, 0))
        assert s.consider(unit(0, 1, 0)) is True  # vuong goc -> d = sqrt(2)

    def test_so_voi_keyframe_gan_nhat_da_chon_khong_phai_frame_truoc(self):
        """Chuoi frame troi tu tu: moi buoc lech 0.3 (<0.4) nen KHONG duoc giu,
        nhung tich luy den buoc thu 2 thi vuot 0.4 so voi anchor -> phai giu."""
        s = kf.KeyframeSelector(0.4)
        s.start_shot()
        a = unit(1, 0, 0)
        s.consider(a)
        b = a + np.array([0, 0.3, 0], dtype=np.float32)
        c = a + np.array([0, 0.45, 0], dtype=np.float32)
        assert s.consider(b) is False   # d = 0.30 <= 0.4
        assert s.consider(c) is True    # van so voi `a` -> d = 0.45 > 0.4

    def test_start_shot_reset_moc_so_sanh(self):
        s = kf.KeyframeSelector(0.4)
        s.start_shot()
        v = unit(1, 0, 0)
        s.consider(v)
        assert s.consider(v) is False   # y het -> loai
        s.start_shot()
        assert s.consider(v) is True    # sang shot moi -> lai la anchor

    def test_threshold_khong_hop_le(self):
        with pytest.raises(ValueError, match="l2_threshold"):
            kf.KeyframeSelector(0)


class TestSamplePlan:
    def test_lay_mau_tu_start_frame_moi_8_frame(self):
        shots = [{"shot_id": 1, "start_frame": 0, "end_frame": 20}]
        assert kf.build_sample_plan(shots, 8) == [(0, 1), (8, 1), (16, 1)]

    def test_moi_shot_bat_dau_lai_tu_start_frame_cua_no(self):
        shots = [
            {"shot_id": 1, "start_frame": 0, "end_frame": 20},
            {"shot_id": 2, "start_frame": 21, "end_frame": 30},
        ]
        assert kf.build_sample_plan(shots, 8) == [(0, 1), (8, 1), (16, 1), (21, 2), (29, 2)]

    def test_shot_ngan_hon_sample_every_van_co_dung_1_mau(self):
        shots = [{"shot_id": 7, "start_frame": 100, "end_frame": 103}]
        assert kf.build_sample_plan(shots, 8) == [(100, 7)]

    def test_sample_every_khong_hop_le(self):
        with pytest.raises(ValueError, match="sample_every"):
            kf.build_sample_plan([], 0)


def _fake_video(root: Path, video_id: str, frames: list[int], dim: int = 4) -> np.ndarray:
    """Tao san ket qua A.2 cua mot video (khong can model)."""
    d = root / video_id
    d.mkdir(parents=True, exist_ok=True)
    emb = np.random.RandomState(abs(hash(video_id)) % 2**31).rand(len(frames), dim).astype(np.float32)
    np.save(d / kf.KEYFRAME_EMB, emb)
    kf.write_keyframe_meta(d / kf.KEYFRAME_META, {
        "version": kf.KEYFRAME_META_VERSION,
        "video_id": video_id,
        "fps": 25.0,
        "sample_every": 8,
        "l2_threshold": 0.4,
        "clip_model": "ViT-L-14-quickgelu/dfn2b",
        "dim": dim,
        "n_sampled": len(frames) * 2,
        "n_keyframes": len(frames),
        "keyframes": [
            {"frame_idx": f, "shot_id": 1, "pts_time": round(f / 25.0, 4),
             "path": f"{video_id}/{f}.jpg"}
            for f in frames
        ],
    })
    return emb


class TestBuildManifest:
    def test_manifest_khop_tung_hang_voi_embedding(self, tmp_path):
        root = tmp_path / "keyframes"
        e_b = _fake_video(root, "L01_V002", [0, 16])
        e_a = _fake_video(root, "L01_V001", [0, 8, 24])

        manifest = tmp_path / "manifest.csv"
        emb_path = tmp_path / "clip.npy"
        meta_path = tmp_path / "meta.json"
        n, dim = kf.build_manifest(root, manifest, emb_path, meta_path)

        assert (n, dim) == (5, 4)
        entries = load_manifest(manifest)
        # duyet theo THU TU TEN video, khong phai thu tu tao file
        assert [e.video_id for e in entries] == ["L01_V001"] * 3 + ["L01_V002"] * 2
        assert [e.frame_idx for e in entries] == [0, 8, 24, 0, 16]
        assert [e.idx for e in entries] == [0, 1, 2, 3, 4]

        all_emb = np.load(emb_path)
        np.testing.assert_array_equal(all_emb[:3], e_a)
        np.testing.assert_array_equal(all_emb[3:], e_b)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["n_manifest"] == 5 and meta["ntotal"]["clip_embeddings"] == 5

    def test_pts_time_tinh_tu_fps(self, tmp_path):
        root = tmp_path / "keyframes"
        _fake_video(root, "L01_V001", [0, 25, 50])
        n, _ = kf.build_manifest(root, tmp_path / "m.csv", tmp_path / "e.npy")
        assert n == 3
        assert [e.pts_time for e in load_manifest(tmp_path / "m.csv")] == [0.0, 1.0, 2.0]

    def test_bat_loi_embedding_lech_so_keyframe(self, tmp_path):
        root = tmp_path / "keyframes"
        _fake_video(root, "L01_V001", [0, 8])
        np.save(root / "L01_V001" / kf.KEYFRAME_EMB, np.zeros((5, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="clip.npy"):
            kf.build_manifest(root, tmp_path / "m.csv", tmp_path / "e.npy")

    def test_bat_loi_lech_so_chieu_giua_hai_video(self, tmp_path):
        root = tmp_path / "keyframes"
        _fake_video(root, "L01_V001", [0], dim=4)
        _fake_video(root, "L01_V002", [0], dim=8)
        with pytest.raises(ValueError, match="dim"):
            kf.build_manifest(root, tmp_path / "m.csv", tmp_path / "e.npy")


def test_read_keyframe_meta_chan_version_cu(tmp_path):
    p = tmp_path / "keyframes.json"
    p.write_text('{"version": 0}', encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        kf.read_keyframe_meta(p)
