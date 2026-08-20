"""Test A.4 / A.6 - SQLite FTS5. Khong can torch lan easyocr."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from aic.store import sqlite_store as store

ROWS = [
    ("L01_V001", 24, "Đường Điện Biên Phủ", 0.91),
    ("L01_V001", 24, "TIN NÓNG", 0.75),
    ("L01_V002", 8, "Quận Bình Thạnh", 0.62),
    ("L01_V003", 0, "duong dien bien", 0.20),
]


@pytest.fixture
def conn(tmp_path):
    c = store.open_db(tmp_path / "metadata.db")
    store.insert_ocr(c, ROWS)
    store.mark_done(c, "L01_V001", 2, "easyocr")
    c.commit()
    return c


class TestSchema:
    def test_fts5_co_san(self, conn):
        store.check_fts5(conn)

    def test_bang_ocr_dung_5_cot(self, conn):
        cols = [r[1] for r in conn.execute("PRAGMA table_info(ocr)")]
        assert cols == ["video_id", "frame_idx", "text", "text_norm", "confidence"]

    def test_frame_idx_giu_kieu_so(self, conn):
        """Ly do khong dung bang FTS5 lam bang chinh: FTS5 bien moi thu thanh TEXT."""
        row = conn.execute("SELECT frame_idx, confidence FROM ocr LIMIT 1").fetchone()
        assert isinstance(row["frame_idx"], int)
        assert isinstance(row["confidence"], float)

    def test_mo_lai_db_khong_hong_du_lieu(self, tmp_path):
        p = tmp_path / "x.db"
        c1 = store.open_db(p)
        store.insert_ocr(c1, ROWS[:1])
        c1.commit()
        c1.close()
        assert store.stats(store.open_db(p))["rows"] == 1


class TestChuanHoa:
    def test_text_norm_duoc_sinh_tu_dong(self, conn):
        row = conn.execute(
            "SELECT text, text_norm FROM ocr WHERE video_id='L01_V001' AND confidence=0.91"
        ).fetchone()
        assert row["text"] == "Đường Điện Biên Phủ"       # nguyen van de hien thi
        assert row["text_norm"] == "duong dien bien phu"  # da bo dau + D->d

    def test_go_khong_dau_van_tim_thay_chu_co_dau(self, conn):
        found = [r["text"] for r in store.search_text(conn, "dien bien phu")]
        assert "Đường Điện Biên Phủ" in found

    def test_chu_D_gach_ngang_tim_duoc(self, conn):
        """Cho FTS5 tokenizer khong xu ly duoc - phai chuan hoa phia Python."""
        assert [r["text"] for r in store.search_text(conn, "Duong")] != []

    def test_go_co_dau_cung_tim_thay(self, conn):
        assert [r["text"] for r in store.search_text(conn, "Điện Biên Phủ")] != []


class TestTruyVan:
    def test_phrase_query_doi_hoi_lien_nhau(self, conn):
        assert store.search_text(conn, "phu duong", phrase=True) == []

    def test_khong_phrase_thi_chi_can_du_tu(self, conn):
        assert len(store.search_text(conn, "phu duong", phrase=False)) == 1

    def test_loc_theo_confidence(self, conn):
        assert len(store.search_text(conn, "dien bien")) == 2
        assert len(store.search_text(conn, "dien bien", min_confidence=0.5)) == 1

    def test_bm25_cang_cao_cang_khop(self, conn):
        rows = store.search_text(conn, "dien bien")
        scores = [r["bm25"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_limit(self, conn):
        assert len(store.search_text(conn, "dien bien", limit=1)) == 1

    def test_matching_frames_tra_ve_cap_video_frame(self, conn):
        assert store.matching_frames(conn, "TIN NONG") == {("L01_V001", 24)}

    def test_frame_texts_sap_theo_confidence(self, conn):
        texts = store.frame_texts(conn, "L01_V001", 24)
        assert [t["text"] for t in texts] == ["Đường Điện Biên Phủ", "TIN NÓNG"]

    def test_query_rong_bi_chan(self, conn):
        with pytest.raises(ValueError, match="rong"):
            store.search_text(conn, "   ")

    def test_dau_nhay_kep_khong_pha_cu_phap(self, conn):
        store.search_text(conn, 'tin "nong"')      # khong duoc raise loi cu phap FTS5

    def test_to_fts_query_thoat_dau_nhay(self):
        assert store.to_fts_query('a"b') == '"a""b"'


class TestResume:
    def test_done_videos(self, conn):
        assert store.done_videos(conn) == {"L01_V001"}

    def test_video_khong_co_chu_van_duoc_danh_dau_xong(self, conn):
        store.mark_done(conn, "L01_V009", 0, "easyocr")
        conn.commit()
        assert "L01_V009" in store.done_videos(conn)

    def test_delete_video_don_luon_chi_muc_fts(self, conn):
        assert store.search_text(conn, "TIN NONG") != []
        store.delete_video(conn, "L01_V001")
        conn.commit()
        assert store.search_text(conn, "TIN NONG") == []
        assert store.done_videos(conn) == set()

    def test_chay_lai_khong_sinh_dong_trung(self, conn):
        store.delete_video(conn, "L01_V001")
        store.insert_ocr(conn, ROWS[:2])
        conn.commit()
        assert store.stats(conn)["rows"] == 4

    def test_stats(self, conn):
        assert store.stats(conn) == {"rows": 4, "frames": 3, "videos": 1}
