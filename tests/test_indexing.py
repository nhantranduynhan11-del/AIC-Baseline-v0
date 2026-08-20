"""Test A.3 - tap trung vao bat bien ID va viec normalize.

`faiss` chua chac co tren may dev nen cac test dung FAISS that duoc skip neu thieu.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.manifest import KeyframeEntry, write_manifest
from aic.preprocess import indexing
from aic.preprocess import keyframe as kf
from aic.store import faiss_store


def normalized(rows: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.rand(rows, dim).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


@pytest.fixture
def dataset(tmp_path):
    """Hai video, tong 5 keyframe. Tra ve (keyframes_dir, manifest_path, siglip_emb)."""
    root = tmp_path / "keyframes"
    frames = {"L01_V001": [0, 8, 24], "L01_V002": [0, 16]}
    entries, blocks = [], []
    for seed, (video_id, idxs) in enumerate(frames.items()):
        d = root / video_id
        d.mkdir(parents=True)
        emb = normalized(len(idxs), 4, seed)
        np.save(d / indexing.SIGLIP_EMB, emb)
        blocks.append(emb)
        kf.write_keyframe_meta(d / kf.KEYFRAME_META, {
            "version": kf.KEYFRAME_META_VERSION, "video_id": video_id, "fps": 25.0,
            "sample_every": 8, "l2_threshold": 0.4, "clip_model": "x", "dim": 4,
            "n_sampled": 10, "n_keyframes": len(idxs),
            "keyframes": [
                {"frame_idx": f, "shot_id": 1, "pts_time": f / 25.0, "path": f"{video_id}/{f}.jpg"}
                for f in idxs
            ],
        })
        entries += [
            KeyframeEntry(-1, video_id, f, f / 25.0, f"{video_id}/{f}.jpg") for f in idxs
        ]

    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, entries)
    return root, manifest, np.concatenate(blocks)


class TestGatherEmbeddings:
    def test_thu_tu_theo_manifest_khong_theo_glob(self, dataset):
        root, manifest, expected = dataset
        got = indexing.gather_embeddings(manifest, root, indexing.SIGLIP_EMB, 5)
        np.testing.assert_array_equal(got, expected)

    def test_thieu_file_bao_loi_ro_rang(self, dataset):
        root, manifest, _ = dataset
        (root / "L01_V002" / indexing.SIGLIP_EMB).unlink()
        with pytest.raises(FileNotFoundError, match="siglip2.npy"):
            indexing.gather_embeddings(manifest, root, indexing.SIGLIP_EMB, 5)

    def test_bat_loi_so_hang_lech_trong_mot_video(self, dataset):
        root, manifest, _ = dataset
        np.save(root / "L01_V001" / indexing.SIGLIP_EMB, normalized(7, 4, 9))
        with pytest.raises(AssertionError, match="L01_V001"):
            indexing.gather_embeddings(manifest, root, indexing.SIGLIP_EMB, 5)

    def test_bat_loi_lech_so_chieu_giua_hai_video(self, dataset):
        root, manifest, _ = dataset
        np.save(root / "L01_V002" / indexing.SIGLIP_EMB, normalized(2, 8, 3))
        with pytest.raises(ValueError, match="dim"):
            indexing.gather_embeddings(manifest, root, indexing.SIGLIP_EMB, 5)


class TestVideoIdsInManifest:
    def test_giu_thu_tu_xuat_hien(self, dataset):
        _, manifest, _ = dataset
        assert indexing.video_ids_in_manifest(manifest) == ["L01_V001", "L01_V002"]

    def test_bat_manifest_bi_xen_ke_video(self, tmp_path):
        manifest = tmp_path / "m.csv"
        write_manifest(manifest, [
            KeyframeEntry(-1, "A", 0, 0.0, "A/0.jpg"),
            KeyframeEntry(-1, "B", 0, 0.0, "B/0.jpg"),
            KeyframeEntry(-1, "A", 8, 0.3, "A/8.jpg"),
        ])
        with pytest.raises(ValueError, match="lien tuc"):
            indexing.video_ids_in_manifest(manifest)


class TestLoadClipEmbeddings:
    def test_bat_loi_lech_so_dong_manifest(self, tmp_path):
        p = tmp_path / "clip.npy"
        np.save(p, normalized(4, 4, 0))
        with pytest.raises(AssertionError, match="clip_embeddings"):
            indexing.load_clip_embeddings(p, 5)

    def test_thieu_file_chi_ro_buoc_can_chay(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="build-manifest"):
            indexing.load_clip_embeddings(tmp_path / "khong-co.npy", 5)


class TestAssertNormalized:
    def test_chap_nhan_vector_don_vi(self):
        faiss_store.assert_normalized(normalized(10, 8, 1))

    def test_chan_vector_chua_normalize(self):
        with pytest.raises(ValueError, match="normalize"):
            faiss_store.assert_normalized(normalized(10, 8, 1) * 3.0, "clip")

    def test_mang_rong_khong_bao_loi(self):
        faiss_store.assert_normalized(np.zeros((0, 4), dtype=np.float32))


faiss_required = pytest.mark.skipif(
    __import__("importlib").util.find_spec("faiss") is None,
    reason="chua cai faiss tren may nay",
)


@faiss_required
class TestFaissIndex:
    def test_build_va_search_tra_ve_dung_row(self, tmp_path):
        emb = normalized(20, 8, 5)
        index = faiss_store.build_flat_ip(emb, name="clip")
        assert index.ntotal == 20
        scores, ids = faiss_store.search(index, emb[3], k=1)
        assert ids[0][0] == 3
        assert scores[0][0] == pytest.approx(1.0, abs=1e-4)

    def test_tu_choi_embedding_chua_normalize(self):
        with pytest.raises(ValueError, match="normalize"):
            faiss_store.build_flat_ip(normalized(5, 4, 2) * 2.0)

    def test_roundtrip_qua_dia(self, tmp_path):
        emb = normalized(10, 4, 7)
        p = tmp_path / "x.faiss"
        faiss_store.save_index(faiss_store.build_flat_ip(emb), p)
        assert faiss_store.load_index(p).ntotal == 10

    def test_build_indexes_khop_manifest(self, dataset, tmp_path):
        root, manifest, siglip = dataset
        clip_path = tmp_path / "clip_embeddings.npy"
        np.save(clip_path, normalized(5, 4, 42))
        result = indexing.build_indexes(
            manifest, root, clip_path,
            tmp_path / "clip.faiss", tmp_path / "siglip.faiss", tmp_path / "meta.json",
        )
        assert result["n_manifest"] == 5
        assert result["ntotal"] == {"clip": 5, "siglip2": 5}
