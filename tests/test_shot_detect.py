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
