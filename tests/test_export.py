"""Test dinh dang nop bai. Doc lai bang byte tho de kiem chung tung ky tu."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from aic.submit import export


def raw(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


class TestKIS:
    def test_dinh_dang_dung_2_cot_khong_header(self, tmp_path):
        p = tmp_path / "kis.csv"
        n = export.write_kis(p, [("L01_V028", 3450), ("L02_V011", 1200)])
        assert n == 2
        assert raw(p) == "L01_V028,3450\r\nL02_V011,1200\r\n"

    def test_khong_co_khoang_trang_sau_dau_phay(self, tmp_path):
        p = tmp_path / "kis.csv"
        export.write_kis(p, [("L01_V028", 3450)])
        assert ", " not in raw(p)

    def test_utf8_khong_bom(self, tmp_path):
        p = tmp_path / "kis.csv"
        export.write_kis(p, [("L01_V028", 1)])
        assert not p.read_bytes().startswith(b"\xef\xbb\xbf")

    def test_dung_100_dong_thi_duoc(self, tmp_path):
        p = tmp_path / "kis.csv"
        assert export.write_kis(p, [("V", i) for i in range(100)]) == 100

    def test_vuot_100_dong_thi_raise_chu_khong_tu_cat(self, tmp_path):
        with pytest.raises(ValueError, match="vuot gioi han 100"):
            export.write_kis(tmp_path / "kis.csv", [("V", i) for i in range(101)])


class TestQA:
    def test_answer_don_gian_khong_can_ngoac_kep(self, tmp_path):
        p = tmp_path / "qa.csv"
        export.write_qa(p, [("L01_V028", 3450, "5"), ("L02_V011", 1200, "Năm người")])
        assert raw(p) == "L01_V028,3450,5\r\nL02_V011,1200,Năm người\r\n"

    def test_dau_phay_trong_answer_duoc_bao_ngoac_kep(self, tmp_path):
        p = tmp_path / "qa.csv"
        export.write_qa(p, [("L01_V028", 3450, "Có 3 người, bao gồm nam và nữ")])
        assert raw(p) == 'L01_V028,3450,"Có 3 người, bao gồm nam và nữ"\r\n'

    def test_dau_ngoac_kep_duoc_escape_bang_double_quotes(self, tmp_path):
        p = tmp_path / "qa.csv"
        export.write_qa(p, [("L01_V028", 3450, 'Anh ấy nói "Xin chào"')])
        assert raw(p) == 'L01_V028,3450,"Anh ấy nói ""Xin chào"""\r\n'

    def test_xuong_dong_trong_answer_duoc_bao_ngoac_kep(self, tmp_path):
        p = tmp_path / "qa.csv"
        export.write_qa(p, [("L01_V028", 3450, "Dòng 1\nDòng 2")])
        assert raw(p) == 'L01_V028,3450,"Dòng 1\nDòng 2"\r\n'
        export.validate_file(p, "qa")      # doc lai van ra dung 3 cot

    def test_answer_dung_100_ky_tu_thi_duoc(self, tmp_path):
        p = tmp_path / "qa.csv"
        assert export.write_qa(p, [("V", 1, "x" * 100)]) == 1

    def test_answer_101_ky_tu_bi_chan(self, tmp_path):
        with pytest.raises(ValueError, match="101 ky tu"):
            export.write_qa(tmp_path / "qa.csv", [("V", 1, "x" * 101)])

    def test_answer_rong_bi_chan(self, tmp_path):
        with pytest.raises(ValueError, match="rong"):
            export.write_qa(tmp_path / "qa.csv", [("V", 1, "")])


class TestTRAKE:
    def test_so_frame_thay_doi_theo_tung_dong(self, tmp_path):
        p = tmp_path / "trake.csv"
        n = export.write_trake(p, [("L01_V028", [100, 250, 400]), ("L02_V011", [10, 20])])
        assert n == 2
        assert raw(p) == "L01_V028,100,250,400\r\nL02_V011,10,20\r\n"

    def test_kiem_tra_so_su_kien_khop_voi_truy_van(self, tmp_path):
        with pytest.raises(ValueError, match="yeu cau 3 su kien"):
            export.write_trake(tmp_path / "t.csv", [("V", [10, 20])], n_events=3)

    def test_frame_khong_theo_thu_tu_thoi_gian_bi_chan(self, tmp_path):
        with pytest.raises(ValueError, match="thu tu thoi gian"):
            export.write_trake(tmp_path / "t.csv", [("V", [100, 50, 200])])

    def test_frame_trung_nhau_van_chap_nhan(self, tmp_path):
        assert export.write_trake(tmp_path / "t.csv", [("V", [100, 100, 200])]) == 1

    def test_khong_co_frame_nao_bi_chan(self, tmp_path):
        with pytest.raises(ValueError, match="khong co frame"):
            export.write_trake(tmp_path / "t.csv", [("V", [])])


class TestValidateFile:
    def test_doc_lai_file_hop_le(self, tmp_path):
        p = tmp_path / "kis.csv"
        export.write_kis(p, [("L01_V028", 3450)])
        assert export.validate_file(p, "kis") == {"task": "kis", "rows": 1, "path": str(p)}

    def test_bat_sai_so_cot(self, tmp_path):
        p = tmp_path / "x.csv"
        export.write_kis(p, [("L01_V028", 3450)])
        with pytest.raises(ValueError, match="QA can dung 3 cot"):
            export.validate_file(p, "qa")

    def test_bat_frame_idx_khong_phai_so(self, tmp_path):
        p = tmp_path / "x.csv"
        p.write_text("L01_V028,abc\r\n", encoding="utf-8")
        with pytest.raises(ValueError):
            export.validate_file(p, "kis")

    def test_bat_trake_sai_so_su_kien(self, tmp_path):
        p = tmp_path / "t.csv"
        export.write_trake(p, [("V", [10, 20])])
        with pytest.raises(ValueError, match="2 frame, can 3"):
            export.validate_file(p, "trake", n_events=3)

    def test_file_rong_bi_chan(self, tmp_path):
        p = tmp_path / "e.csv"
        p.write_text("", encoding="utf-8")
        with pytest.raises(ValueError, match="rong"):
            export.validate_file(p, "kis")

    def test_task_khong_hop_le(self, tmp_path):
        p = tmp_path / "x.csv"
        export.write_kis(p, [("V", 1)])
        with pytest.raises(ValueError, match="task phai"):
            export.validate_file(p, "vqa")


class TestChuyenDoiTuPipeline:
    def test_hits_to_kis(self):
        rows = [{"video_id": "L01_V001", "frame_idx": 24, "score": 0.9},
                {"video_id": "L01_V002", "frame_idx": 8, "score": 0.8}]
        assert export.hits_to_kis(rows) == [("L01_V001", 24), ("L01_V002", 8)]

    def test_hits_to_qa(self):
        rows = [{"video_id": "L01_V001", "frame_idx": 24}]
        assert export.hits_to_qa(rows, "Màu đỏ") == [("L01_V001", 24, "Màu đỏ")]
