"""A.6 FAISS store - IndexFlatIP + normalize L2.

Quy tac khong duoc pha:
  - Normalize L2 CA vector index LAN vector query -> tich vo huong = cosine.
    Normalize mot ben ma quen ben kia thi diem so sai lang le, khong bao gio bao loi.
  - Khong IVF/HNSW o v0: vong so loai uu tien do chinh xac, Flat cho 100% recall
    khong co sai so xap xi.
  - Thu tu them vector = thu tu manifest. add() mot lan, mot mach, khong sort lai.

Ngan sach RAM: Flat giu toan bo vector trong bo nho, N x D x 4 byte moi index.
500k keyframe: 500k x 768 x 4 = 1.5GB (CLIP) + 500k x 1024 x 4 = 2.0GB (SigLIP2).

faiss import LAZY de module import duoc tren may chua cai faiss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NORM_TOLERANCE = 1e-3


def assert_normalized(vectors: np.ndarray, name: str = "vectors") -> None:
    """Chan som truong hop quen normalize - loi nay khong tu bao."""
    if vectors.size == 0:
        return
    norms = np.linalg.norm(vectors, axis=1)
    worst = float(np.max(np.abs(norms - 1.0)))
    if worst > NORM_TOLERANCE:
        raise ValueError(
            f"{name}: chua normalize L2 (lech toi da {worst:.4f} so voi 1.0). "
            "IndexFlatIP chi bang cosine khi vector da normalize."
        )


def build_flat_ip(embeddings: np.ndarray, *, name: str = "index"):
    """Tao IndexFlatIP va add toan bo embedding theo dung thu tu dua vao."""
    import faiss

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"{name}: can mang 2 chieu, nhan shape {embeddings.shape}")
    assert_normalized(embeddings, name)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    if index.ntotal != len(embeddings):
        raise AssertionError(f"{name}: ntotal {index.ntotal} != {len(embeddings)} vector dua vao")
    return index


def save_index(index, path: str | Path) -> None:
    import faiss

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: str | Path):
    import faiss

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chua co FAISS index: {path}. Chay A.3 truoc.")
    return faiss.read_index(str(path))


def search(index, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Tra ve (scores, ids), moi cai shape (n_query, k).

    ids la row index cua FAISS == idx trong manifest. FAISS tra ve -1 cho o trong
    khi k > ntotal.
    """
    queries = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
    assert_normalized(queries, "query")
    return index.search(queries, min(k, index.ntotal))


def index_dim(index) -> int:
    return int(index.d)
