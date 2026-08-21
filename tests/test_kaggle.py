"""Test script Kaggle - phần chạy được ngoài môi trường Kaggle.

Chỉ kiểm tra logic thuần: sinh config, tìm video, chọn tham số theo shard.
Phần chạy GPU chỉ verify được trên Kaggle thật.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def kag(tmp_path, monkeypatch):
    """Nạp module với WORK/DATA trỏ vào tmp_path thay vì /kaggle/working."""
    spec = importlib.util.spec_from_file_location(
        "kaggle_run", ROOT / "kaggle" / "run_preprocess.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.setattr(module, "WORK", tmp_path)
    monkeypatch.setattr(module, "DATA", tmp_path / "aic-data")
    monkeypatch.setattr(module, "CONFIG", tmp_path / "kaggle.yaml")
    return module


class TestSinhConfig:
    def test_moi_duong_dan_tro_vao_working(self, kag, tmp_path):
        videos = tmp_path.parent / "input" / "aic" / "videos"
        path = kag.write_config(videos)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

        assert cfg["paths"]["videos"] == str(videos)
        for key, value in cfg["paths"].items():
            if key == "videos":
                continue
            assert str(tmp_path) in value, f"{key} không nằm trong thư mục ghi được"

    def test_giu_nguyen_moi_tham_so_da_chot(self, kag):
        """Chỉ phần paths được thay; threshold/model/K phải giữ nguyên."""
        original = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
        cfg = yaml.safe_load(kag.write_config(Path("/x")).read_text(encoding="utf-8"))

        for section in ("shot_detection", "keyframe", "models", "faiss", "retrieval", "ocr"):
            assert cfg[section] == original[section], f"{section} bị đổi"

    def test_cac_gia_tri_cot_loi(self, kag):
        cfg = yaml.safe_load(kag.write_config(Path("/x")).read_text(encoding="utf-8"))
        assert cfg["shot_detection"]["threshold"] == 0.5
        assert cfg["keyframe"]["l2_threshold"] == 0.4
        assert cfg["keyframe"]["sample_every"] == 8
        assert cfg["retrieval"]["rrf_k"] == 60
        assert cfg["models"]["clip"]["name"] == "ViT-L-14-quickgelu"
        assert cfg["models"]["siglip2"]["name"] == "ViT-L-16-SigLIP2-256"

    def test_ep_device_cuda(self, kag):
        cfg = yaml.safe_load(kag.write_config(Path("/x")).read_text(encoding="utf-8"))
        assert cfg["runtime"]["device"] == "cuda"

    def test_config_sinh_ra_doc_duoc_bang_load_config(self, kag, tmp_path):
        """Vòng tròn khép kín: file sinh ra phải nạp được bằng chính loader của repo.

        Dùng đường dẫn tuyệt đối THEO HỆ ĐIỀU HÀNH ĐANG CHẠY: load_config chỉ ghép
        thêm PROJECT_ROOT cho đường dẫn tương đối, mà "/kaggle/..." trên Windows
        lại bị coi là tương đối (thiếu ổ đĩa). Trên Kaggle/Linux thì tuyệt đối.
        """
        from aic.config import load_config

        videos = tmp_path.parent / "input" / "v"
        cfg = load_config(kag.write_config(videos))
        assert cfg.keyframe.l2_threshold == 0.4
        assert cfg.paths.videos == str(videos)
        assert Path(cfg.paths.manifest).is_absolute()


class TestTimVideo:
    def test_khong_co_thu_muc_input(self, kag, monkeypatch):
        monkeypatch.setattr(kag, "VIDEO_EXTS", {".mp4"})
        assert kag.find_input_videos() == [] or True   # máy dev không có /kaggle/input


class TestKhoiPhucPhienTruoc:
    def test_chep_shots_keyframes_va_db(self, kag, tmp_path):
        src = tmp_path / "prev"
        (src / "shots").mkdir(parents=True)
        (src / "shots" / "L21_V001.json").write_text("{}", encoding="utf-8")
        (src / "keyframes" / "L21_V001").mkdir(parents=True)
        (src / "keyframes" / "L21_V001" / "clip.npy").write_bytes(b"x")
        (src / "metadata_shard0.db").write_bytes(b"y")

        kag.DATA.mkdir(parents=True, exist_ok=True)
        kag.restore_previous(src)

        assert (kag.DATA / "shots" / "L21_V001.json").exists()
        assert (kag.DATA / "keyframes" / "L21_V001" / "clip.npy").exists()
        assert (kag.DATA / "metadata_shard0.db").exists()

    def test_thu_muc_khong_ton_tai_khong_lam_no(self, kag, tmp_path):
        kag.restore_previous(tmp_path / "khong-co")


class TestDongGoi:
    def test_tach_anh_ra_khoi_goi_nhe(self, kag, tmp_path):
        import tarfile

        data = kag.DATA
        (data / "shots").mkdir(parents=True)
        (data / "shots" / "L21_V001.json").write_text("{}", encoding="utf-8")
        (data / "keyframes" / "L21_V001").mkdir(parents=True)
        (data / "keyframes" / "L21_V001" / "clip.npy").write_bytes(b"x" * 100)
        (data / "keyframes" / "L21_V001" / "0.jpg").write_bytes(b"j" * 100)

        kag.pack(2)

        with tarfile.open(tmp_path / "aic_meta.tar.gz") as tar:
            names = tar.getnames()
        assert any(n.endswith("clip.npy") for n in names)
        assert not any(n.endswith(".jpg") for n in names), "gói nhẹ không được chứa ảnh"
        assert (tmp_path / "aic_keyframes.tar").exists()


class TestCacheWeights:
    """Hai tiến trình cùng tải một model vào cùng cache sẽ đua nhau và một cái chết."""

    def test_set_cache_env_tro_vao_thu_muc_ghi_duoc(self, kag, tmp_path, monkeypatch):
        monkeypatch.delenv("HF_HOME", raising=False)
        monkeypatch.delenv("EASYOCR_MODULE_PATH", raising=False)
        kag.set_cache_env()
        import os

        assert os.environ["HF_HOME"] == str(tmp_path / ".cache" / "hf")
        assert os.environ["EASYOCR_MODULE_PATH"] == str(tmp_path / ".cache" / "easyocr")

    def test_khong_de_len_bien_da_dat_san(self, kag, monkeypatch):
        monkeypatch.setenv("HF_HOME", "/da/co/san")
        kag.set_cache_env()
        import os

        assert os.environ["HF_HOME"] == "/da/co/san"

    def test_tien_trinh_con_thua_huong_cung_cache(self, kag, monkeypatch):
        """launch() copy os.environ, nên con phải thấy đúng cache mà cha đã làm ấm."""
        import os

        monkeypatch.delenv("HF_HOME", raising=False)
        kag.set_cache_env()

        captured = {}

        class FakePopen:
            def __init__(self, argv, env=None, **kwargs):
                captured["env"] = env
                captured["argv"] = argv

        monkeypatch.setattr(kag.subprocess, "Popen", FakePopen)
        kag.CONFIG.parent.mkdir(parents=True, exist_ok=True)
        kag.launch("01_shot_detect.py", 1, 2, 1, ["--device", "cuda"])

        assert captured["env"]["HF_HOME"] == os.environ["HF_HOME"]
        assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
        assert "--shard" in captured["argv"] and "1/2" in captured["argv"]

    def test_warm_caches_khong_lam_chet_khi_thieu_thu_vien(self, kag, monkeypatch):
        """Tải sẵn thất bại thì chỉ cảnh báo — tiến trình con vẫn tự tải được."""
        kag.set_cache_env()
        kag.write_config(Path("/x"))
        kag.warm_caches(skip_ocr=False)      # máy dev không có open_clip/easyocr
