"""Test gộp kết quả từ nhiều máy (chia việc cho nhóm)."""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from aic.preprocess.indexing import SIGLIP_EMB
from aic.preprocess.keyframe import KEYFRAME_EMB, KEYFRAME_META
from aic.store import sqlite_store as store

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "06_merge.py"


def make_member_db(path: Path, videos: dict[str, list[str]]):
    """videos: {video_id: [text, ...]} - mô phỏng DB OCR của một thành viên."""
    conn = store.open_db(path)
    for video_id, texts in videos.items():
        rows = [(video_id, i * 8, t, 0.8) for i, t in enumerate(texts)]
        store.insert_ocr(conn, rows)
        store.mark_done(conn, video_id, len(rows), "easyocr")
    conn.commit()
    conn.close()


def write_config(tmp_path: Path) -> Path:
    """Config trỏ mọi đường dẫn vào tmp_path."""
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
  faiss_clip: {(tmp_path / 'index/clip.faiss').as_posix()}
  faiss_siglip: {(tmp_path / 'index/siglip.faiss').as_posix()}
  index_meta: {(tmp_path / 'index/meta.json').as_posix()}
  metadata_db: {(tmp_path / 'metadata.db').as_posix()}
""", encoding="utf-8")
    return cfg


def run(cfg: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--config", str(cfg), *args],
        capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT),
    )


class TestGopDB:
    def test_gop_hai_db_thanh_vien(self, tmp_path):
        cfg = write_config(tmp_path)
        a, b = tmp_path / "an.db", tmp_path / "binh.db"
        make_member_db(a, {"L21_V001": ["TIN NÓNG", "Thời sự"], "L21_V002": ["Hà Nội"]})
        make_member_db(b, {"L21_V003": ["Đường Điện Biên Phủ"]})

        result = run(cfg, "--merge-db", str(a), str(b))
        assert result.returncode == 0, result.stderr

        conn = store.open_db(tmp_path / "metadata.db")
        assert store.stats(conn)["rows"] == 4
        assert store.done_videos(conn) == {"L21_V001", "L21_V002", "L21_V003"}

    def test_chi_muc_fts_hoat_dong_sau_khi_gop(self, tmp_path):
        """Quan trọng: gộp bằng INSERT..SELECT nên trigger FTS5 phải chạy."""
        cfg = write_config(tmp_path)
        a = tmp_path / "an.db"
        make_member_db(a, {"L21_V001": ["Đường Điện Biên Phủ"]})
        assert run(cfg, "--merge-db", str(a)).returncode == 0

        conn = store.open_db(tmp_path / "metadata.db")
        found = store.search_text(conn, "dien bien phu")     # gõ không dấu
        assert [r["text"] for r in found] == ["Đường Điện Biên Phủ"]

    def test_chay_lai_khong_sinh_dong_trung(self, tmp_path):
        cfg = write_config(tmp_path)
        a = tmp_path / "an.db"
        make_member_db(a, {"L21_V001": ["x", "y"]})
        run(cfg, "--merge-db", str(a))
        run(cfg, "--merge-db", str(a))
        assert store.stats(store.open_db(tmp_path / "metadata.db"))["rows"] == 2

    def test_overwrite_thay_the_ban_cu(self, tmp_path):
        cfg = write_config(tmp_path)
        a, b = tmp_path / "a.db", tmp_path / "b.db"
        make_member_db(a, {"L21_V001": ["cu"]})
        make_member_db(b, {"L21_V001": ["moi", "them"]})
        run(cfg, "--merge-db", str(a))
        run(cfg, "--merge-db", str(b), "--overwrite")

        conn = store.open_db(tmp_path / "metadata.db")
        texts = {r[0] for r in conn.execute("SELECT text FROM ocr")}
        assert texts == {"moi", "them"}

    def test_db_khong_ton_tai_bao_loi(self, tmp_path):
        cfg = write_config(tmp_path)
        result = run(cfg, "--merge-db", str(tmp_path / "khong-co.db"))
        assert result.returncode == 1


def make_artifacts(tmp_path: Path, video_id: str, *, shot=True, kf=True,
                   clip=True, siglip=True):
    if shot:
        (tmp_path / "shots").mkdir(parents=True, exist_ok=True)
        (tmp_path / "shots" / f"{video_id}.json").write_text("{}", encoding="utf-8")
    d = tmp_path / "keyframes" / video_id
    d.mkdir(parents=True, exist_ok=True)
    if kf:
        (d / KEYFRAME_META).write_text("{}", encoding="utf-8")
    if clip:
        np.save(d / KEYFRAME_EMB, np.zeros((2, 4), dtype=np.float32))
    if siglip:
        np.save(d / SIGLIP_EMB, np.zeros((2, 4), dtype=np.float32))


class TestKiemTraDayDu:
    def test_bao_sach_khi_du_het(self, tmp_path):
        cfg = write_config(tmp_path)
        for v in ["L21_V001", "L21_V002"]:
            make_artifacts(tmp_path, v)
        db = tmp_path / "src.db"
        make_member_db(db, {"L21_V001": ["a"], "L21_V002": ["b"]})
        run(cfg, "--merge-db", str(db))

        result = run(cfg, "--check")
        assert result.returncode == 0
        assert "Đủ hết" in result.stdout

    def test_bat_thieu_siglip2(self, tmp_path):
        cfg = write_config(tmp_path)
        make_artifacts(tmp_path, "L21_V001")
        make_artifacts(tmp_path, "L21_V002", siglip=False)

        result = run(cfg, "--check")
        assert result.returncode == 1
        assert "L21_V002: thiếu" in result.stdout
        assert "A.3 siglip2" in result.stdout

    def test_canh_bao_khong_duoc_build_manifest(self, tmp_path):
        cfg = write_config(tmp_path)
        make_artifacts(tmp_path, "L21_V001", kf=False)
        result = run(cfg, "--check")
        assert "CHƯA được chạy --build-manifest" in result.stdout

    def test_thieu_ocr_cung_bi_bat(self, tmp_path):
        cfg = write_config(tmp_path)
        make_artifacts(tmp_path, "L21_V001")
        result = run(cfg, "--check")
        assert result.returncode == 1
        assert "A.4 OCR" in result.stdout

    def test_khong_co_du_lieu_gi(self, tmp_path):
        cfg = write_config(tmp_path)
        assert run(cfg, "--check").returncode == 1

    def test_can_it_nhat_mot_co(self, tmp_path):
        cfg = write_config(tmp_path)
        assert run(cfg).returncode == 2
