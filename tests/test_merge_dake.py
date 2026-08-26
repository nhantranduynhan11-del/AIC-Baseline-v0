"""Test gộp keyframe DAKE vào baseline.

Rủi ro lớn nhất: thứ tự keyframes.json sau khi gộp. Manifest lấy thứ tự từ đây,
và mọi vector .npy phải khớp từng dòng — lệch một hàng là sai bất biến ID.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.preprocess.indexing import SIGLIP_EMB
from aic.preprocess.keyframe import KEYFRAME_EMB, KEYFRAME_META, KEYFRAME_META_VERSION

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "09_merge_dake.py"


def write_meta(d: Path, video_id: str, frames: list[int], *, method: str = "baseline"):
    d.mkdir(parents=True, exist_ok=True)
    for f in frames:
        (d / f"{f}.jpg").write_bytes(b"jpeg" + str(f).encode())
    meta = {
        "version": KEYFRAME_META_VERSION, "video_id": video_id, "fps": 25.0,
        "sample_every": 8, "l2_threshold": 0.4, "clip_model": "x", "dim": 4,
        "n_sampled": len(frames) * 3, "n_keyframes": len(frames), "method": method,
        "keyframes": [
            {"frame_idx": f, "shot_id": 1, "pts_time": round(f / 25.0, 4),
             "path": f"{video_id}/{f}.jpg", "trigger": "LOCAL_TEXT_STATIC"}
            for f in frames
        ],
    }
    (d / KEYFRAME_META).write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")


def make_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"""
paths:
  data_root: {tmp_path.as_posix()}
  videos: {(tmp_path / 'videos').as_posix()}
  shots: {(tmp_path / 'shots').as_posix()}
  shots_raw: {(tmp_path / 'shots/raw').as_posix()}
  keyframes: {(tmp_path / 'keyframes').as_posix()}
  thumbs: {(tmp_path / 'thumbs').as_posix()}
  index_dir: {(tmp_path / 'index').as_posix()}
  clip_embeddings: {(tmp_path / 'index/clip.npy').as_posix()}
  manifest: {(tmp_path / 'index/manifest.csv').as_posix()}
  faiss_clip: {(tmp_path / 'index/c.faiss').as_posix()}
  faiss_siglip: {(tmp_path / 'index/s.faiss').as_posix()}
  index_meta: {(tmp_path / 'index/meta.json').as_posix()}
  metadata_db: {(tmp_path / 'm.db').as_posix()}
""", encoding="utf-8")
    return cfg


def run(cfg: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg), *args],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
    )


@pytest.fixture
def scene(tmp_path):
    """baseline {0,8,16,24}, dake {8,10,20,24,30} -> hợp {0,8,10,16,20,24,30}."""
    base = tmp_path / "keyframes" / "L25_V001"
    dake = tmp_path / "keyframes_dake" / "L25_V001"
    write_meta(base, "L25_V001", [0, 8, 16, 24])
    write_meta(dake, "L25_V001", [8, 10, 20, 24, 30], method="dake")
    np.save(base / KEYFRAME_EMB, np.zeros((4, 4), dtype=np.float32))
    np.save(base / SIGLIP_EMB, np.zeros((4, 4), dtype=np.float32))
    return tmp_path, base, dake


def read_meta(base: Path) -> dict:
    return json.loads((base / KEYFRAME_META).read_text(encoding="utf-8"))


class TestGop:
    def test_hop_hai_tap_va_sap_theo_frame_idx(self, scene):
        tmp_path, base, _ = scene
        assert run(make_config(tmp_path)).returncode == 0

        meta = read_meta(base)
        assert [k["frame_idx"] for k in meta["keyframes"]] == [0, 8, 10, 16, 20, 24, 30]
        assert meta["n_keyframes"] == 7
        assert meta["n_baseline"] == 4 and meta["n_dake_added"] == 3

    def test_danh_dau_nguon_tung_keyframe(self, scene):
        tmp_path, base, _ = scene
        run(make_config(tmp_path))
        source = {k["frame_idx"]: k["source"] for k in read_meta(base)["keyframes"]}
        assert source == {0: "baseline", 8: "both", 10: "dake", 16: "baseline",
                          20: "dake", 24: "both", 30: "dake"}

    def test_chi_chep_anh_chua_co_khong_de_len(self, scene):
        tmp_path, base, _ = scene
        goc = (base / "8.jpg").read_bytes()          # frame 8 có ở cả hai
        run(make_config(tmp_path))
        assert (base / "8.jpg").read_bytes() == goc  # giữ bản baseline
        assert (base / "10.jpg").exists()            # ảnh DAKE mới được chép
        assert (base / "30.jpg").exists()

    def test_vector_cu_bi_danh_dau_stale(self, scene):
        """Vector cũ có 4 hàng, tập mới 7 hàng — để nguyên là dùng nhầm."""
        tmp_path, base, _ = scene
        run(make_config(tmp_path))
        assert not (base / KEYFRAME_EMB).exists()
        assert not (base / SIGLIP_EMB).exists()
        assert (base / f"{KEYFRAME_EMB}.stale").exists()
        assert (base / f"{SIGLIP_EMB}.stale").exists()

    def test_pts_time_cua_keyframe_moi_lay_tu_dake(self, scene):
        tmp_path, base, _ = scene
        run(make_config(tmp_path))
        by_idx = {k["frame_idx"]: k["pts_time"] for k in read_meta(base)["keyframes"]}
        assert by_idx[10] == pytest.approx(10 / 25.0)
        assert by_idx[30] == pytest.approx(30 / 25.0)

    def test_dry_run_khong_doi_gi(self, scene):
        tmp_path, base, _ = scene
        before = (base / KEYFRAME_META).read_bytes()
        result = run(make_config(tmp_path), "--dry-run")
        assert result.returncode == 0
        assert (base / KEYFRAME_META).read_bytes() == before
        assert not (base / "10.jpg").exists()
        assert (base / KEYFRAME_EMB).exists()

    def test_khong_gop_hai_lan(self, scene):
        tmp_path, _, _ = scene
        cfg = make_config(tmp_path)
        assert run(cfg).returncode == 0
        second = run(cfg)
        assert second.returncode == 1
        assert "đã gộp rồi" in second.stderr

    def test_bao_loi_khi_dake_co_video_khong_co_baseline(self, tmp_path):
        write_meta(tmp_path / "keyframes" / "L25_V001", "L25_V001", [0, 8])
        write_meta(tmp_path / "keyframes_dake" / "L25_V999", "L25_V999", [0], method="dake")
        result = run(make_config(tmp_path))
        assert result.returncode == 1
        assert "L25_V999" in result.stderr


class TestLuiLai:
    def test_restore_tra_ve_nguyen_trang(self, scene):
        tmp_path, base, _ = scene
        cfg = make_config(tmp_path)
        before = json.loads((base / KEYFRAME_META).read_text(encoding="utf-8"))
        run(cfg)
        assert run(cfg, "--restore").returncode == 0

        assert json.loads((base / KEYFRAME_META).read_text(encoding="utf-8")) == before
        assert (base / KEYFRAME_EMB).exists()            # vector được khôi phục
        assert (base / SIGLIP_EMB).exists()
        assert not (base / f"{KEYFRAME_EMB}.stale").exists()

    def test_sau_khi_restore_gop_lai_duoc(self, scene):
        tmp_path, base, _ = scene
        cfg = make_config(tmp_path)
        run(cfg)
        run(cfg, "--restore")
        assert run(cfg).returncode == 0
        assert read_meta(base)["n_keyframes"] == 7
