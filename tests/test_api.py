"""Test tang API. Dung FAISS + SQLite that, encoder gia -> khong can torch/GPU."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aic.api import app as api
from aic.api.state import AppState
from aic.manifest import KeyframeEntry
from aic.retrieval.search import IndexBundle
from aic.store import faiss_store
from aic.store import sqlite_store as store

N = 12


def normalized(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.rand(rows, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class FakeEncoder:
    def encode(self, text=None, image_rgb=None):
        return {"clip": normalized(1, 8, 99)[0], "siglip2": normalized(1, 16, 98)[0]}


class FakeCfg:
    """Cfg toi thieu ma endpoint cham toi."""

    def __init__(self, tmp_path):
        self.paths = type("P", (), {
            "keyframes": str(tmp_path / "keyframes"),
            "thumbs": str(tmp_path / "thumbs"),
            "metadata_db": str(tmp_path / "m.db"),
        })()
        self.retrieval = type("R", (), {"top_k_per_model": 50, "rrf_k": 60, "final_top_n": 100})()
        self.ocr = type("O", (), {"query_min_confidence": 0.3})()


@pytest.fixture
def client(tmp_path):
    """3 video x 4 keyframe = 12, frame_idx 0/8/16/24."""
    entries = [
        KeyframeEntry(i, f"V{i // 4:03d}", (i % 4) * 8, (i % 4) * 0.32,
                      f"V{i // 4:03d}/{(i % 4) * 8}.jpg")
        for i in range(N)
    ]
    bundle = IndexBundle(
        {
            "clip": faiss_store.build_flat_ip(normalized(N, 8, 1)),
            "siglip2": faiss_store.build_flat_ip(normalized(N, 16, 2)),
        },
        entries,
    )

    # anh keyframe gia: file JPEG that de FileResponse tra ve duoc
    from PIL import Image

    for e in entries:
        p = tmp_path / "keyframes" / e.path
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 18), (e.idx * 10 % 255, 0, 0)).save(p, "JPEG")

    conn = store.open_db(tmp_path / "m.db", check_same_thread=False)
    store.insert_ocr(conn, [
        ("V000", 8, "Đường Điện Biên Phủ", 0.91),      # idx 1
        ("V001", 16, "Điện Biên Phủ hôm nay", 0.55),   # idx 6
    ])
    conn.commit()

    state = AppState(cfg=FakeCfg(tmp_path), bundle=bundle, encoder=FakeEncoder(), conn=conn)
    api.app.dependency_overrides[api.get_state] = lambda: state
    with TestClient(api.app) as c:
        yield c
    api.app.dependency_overrides.clear()


class TestHealth:
    def test_bao_chua_san_sang_khi_chua_nap_state(self, tmp_path):
        with TestClient(api.app) as c:
            body = c.get("/health").json()
        assert body["ready"] is False and "error" in body

    def test_khong_co_state_thi_search_tra_503(self, tmp_path):
        with TestClient(api.app) as c:
            assert c.post("/search", json={"query": "x"}).status_code == 503


class TestSearch:
    def test_tra_ve_top_n_da_xep_hang(self, client):
        body = client.post("/search", json={"query": "x", "top_n": 5}).json()
        assert body["n_hits"] == 5
        assert [h["rank"] for h in body["hits"]] == [1, 2, 3, 4, 5]
        assert set(body["hits"][0]) == {
            "idx", "video_id", "frame_idx", "pts_time", "score", "rank", "ranks"
        }

    def test_top_n_khong_duoc_vuot_100(self, client):
        assert client.post("/search", json={"query": "x", "top_n": 101}).status_code == 422

    def test_phai_truyen_dung_mot_trong_query_hoac_image_idx(self, client):
        assert client.post("/search", json={}).status_code == 400
        assert client.post("/search", json={"query": "x", "image_idx": 1}).status_code == 400

    def test_query_bang_anh(self, client):
        body = client.post("/search", json={"image_idx": 3, "top_n": 3}).json()
        assert body["n_hits"] == 3

    def test_ocr_filter_thu_hep_ket_qua(self, client):
        body = client.post("/search", json={"query": "x", "ocr": "dien bien phu"}).json()
        assert {h["idx"] for h in body["hits"]} == {1, 6}
        assert body["ocr_filter"]["n_frames"] == 2

    def test_ocr_filter_loc_theo_confidence(self, client):
        body = client.post(
            "/search",
            json={"query": "x", "ocr": "dien bien phu", "ocr_min_confidence": 0.7},
        ).json()
        assert {h["idx"] for h in body["hits"]} == {1}

    def test_khong_ocr_thi_tra_het(self, client):
        body = client.post("/search", json={"query": "x"}).json()
        assert body["n_hits"] == N and body["ocr_filter"] is None


class TestAnh:
    def test_keyframe_tra_ve_jpeg(self, client):
        r = client.get("/keyframe/5")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"

    def test_thumb_roi_ve_anh_goc_khi_chua_sinh_thumbnail(self, client):
        assert client.get("/thumb/5").status_code == 200

    def test_idx_ngoai_pham_vi(self, client):
        assert client.get("/keyframe/999").status_code == 404
        assert client.get("/keyframe/-1").status_code == 404


class TestNeighbors:
    def test_chi_lay_trong_cung_video(self, client):
        body = client.get("/neighbors/5?w=5").json()      # idx 5 thuoc V001 (idx 4..7)
        assert body["video_id"] == "V001"
        assert [i["idx"] for i in body["items"]] == [4, 5, 6, 7]

    def test_danh_dau_keyframe_hien_tai(self, client):
        items = client.get("/neighbors/5?w=1").json()["items"]
        assert [i["idx"] for i in items] == [4, 5, 6]
        assert [i["is_current"] for i in items] == [False, True, False]

    def test_o_bien_dau_video(self, client):
        body = client.get("/neighbors/0?w=5").json()
        assert [i["idx"] for i in body["items"]] == [0, 1, 2, 3]

    def test_o_bien_cuoi_manifest(self, client):
        body = client.get(f"/neighbors/{N - 1}?w=5").json()
        assert [i["idx"] for i in body["items"]] == [8, 9, 10, 11]


class TestOcrEndpoint:
    def test_tra_ve_chu_tren_keyframe(self, client):
        body = client.get("/ocr/1").json()
        assert body["available"] is True
        assert body["texts"][0]["text"] == "Đường Điện Biên Phủ"

    def test_keyframe_khong_co_chu(self, client):
        assert client.get("/ocr/0").json()["texts"] == []


class TestExport:
    def test_kis(self, client):
        r = client.post("/export", json={
            "task": "kis",
            "items": [{"video_id": "L01_V028", "frame_idx": 3450}],
        })
        assert r.status_code == 200
        assert r.text == "L01_V028,3450\r\n"
        assert "attachment" in r.headers["content-disposition"]

    def test_qa_escape_dau_phay(self, client):
        r = client.post("/export", json={
            "task": "qa",
            "items": [{"video_id": "L01_V028", "frame_idx": 3450}],
            "answer": "Có 3 người, gồm nam và nữ",
        })
        assert r.text == 'L01_V028,3450,"Có 3 người, gồm nam và nữ"\r\n'

    def test_qa_thieu_answer(self, client):
        r = client.post("/export", json={
            "task": "qa", "items": [{"video_id": "V", "frame_idx": 1}],
        })
        assert r.status_code == 400

    def test_answer_qua_100_ky_tu_bi_chan_o_schema(self, client):
        r = client.post("/export", json={
            "task": "qa", "items": [{"video_id": "V", "frame_idx": 1}], "answer": "x" * 101,
        })
        assert r.status_code == 422

    def test_trake(self, client):
        r = client.post("/export", json={
            "task": "trake",
            "items": [{"video_id": "L01_V028", "frames": [100, 250, 400]}],
            "n_events": 3,
        })
        assert r.text == "L01_V028,100,250,400\r\n"

    def test_trake_sai_so_su_kien(self, client):
        r = client.post("/export", json={
            "task": "trake",
            "items": [{"video_id": "V", "frames": [10, 20]}],
            "n_events": 3,
        })
        assert r.status_code == 400 and "3 su kien" in r.json()["detail"]

    def test_danh_sach_rong(self, client):
        assert client.post("/export", json={"task": "kis", "items": []}).status_code == 400

    def test_vuot_100_dong(self, client):
        r = client.post("/export", json={
            "task": "kis",
            "items": [{"video_id": "V", "frame_idx": i} for i in range(101)],
        })
        assert r.status_code == 400 and "100" in r.json()["detail"]
