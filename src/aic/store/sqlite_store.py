"""A.6 SQLite FTS5 - MOT file .db duy nhat, thay the Elasticsearch cho tang B.

Chua OCR (A.4) truoc; sau nay them ASR (B.2) va object detection (A.5) vao cung
file nay, moi thu mot bang rieng.

Cau truc: bang thuong `ocr` (dung 5 cot theo baseline) + mot FTS5 external-content
index `ocr_fts` chi danh chi muc cot `text_norm`.

Vi sao tach lam hai thay vi de `ocr` la bang FTS5 luon:
  - Bang FTS5 luu moi thu duoi dang TEXT -> frame_idx va confidence mat kieu,
    khong so sanh so hoc duoc trong SQL (hard filter can loc theo confidence).
  - External content index khong nhan doi du lieu: no chi giu chi muc, doc noi
    dung tu bang goc qua rowid.
  - `bm25(ocr_fts)` van dung binh thuong - do la nguon diem cho duong nang cap
    harmonic mean (N.1).

Tokenizer dung `unicode61 remove_diacritics 2` lam lop bao ve thu hai. Lop chinh
van la chuan hoa phia Python (`aic.text_norm.normalize_vi`), vi tokenizer nay bo
duoc dau nguyen am nhung KHONG xu ly `d`/`D`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Sequence

from aic.text_norm import normalize_vi

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ocr (
    video_id   TEXT    NOT NULL,
    frame_idx  INTEGER NOT NULL,
    text       TEXT    NOT NULL,
    text_norm  TEXT    NOT NULL,
    confidence REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ocr_frame ON ocr(video_id, frame_idx);

CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
    text_norm,
    content='ocr',
    content_rowid='rowid',
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS ocr_ai AFTER INSERT ON ocr BEGIN
    INSERT INTO ocr_fts(rowid, text_norm) VALUES (new.rowid, new.text_norm);
END;

CREATE TRIGGER IF NOT EXISTS ocr_ad AFTER DELETE ON ocr BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, text_norm) VALUES('delete', old.rowid, old.text_norm);
END;

CREATE TRIGGER IF NOT EXISTS ocr_au AFTER UPDATE ON ocr BEGIN
    INSERT INTO ocr_fts(ocr_fts, rowid, text_norm) VALUES('delete', old.rowid, old.text_norm);
    INSERT INTO ocr_fts(rowid, text_norm) VALUES (new.rowid, new.text_norm);
END;

-- Danh dau video da chay xong, ke ca video khong co chu nao -> resume dung.
CREATE TABLE IF NOT EXISTS ocr_done (
    video_id   TEXT PRIMARY KEY,
    n_rows     INTEGER NOT NULL,
    engine     TEXT    NOT NULL,
    created_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def check_fts5(conn: sqlite3.Connection) -> None:
    """FTS5 la bat buoc. Ban Python thieu no thi phai biet ngay tu dau."""
    options = {row[0] for row in conn.execute("pragma compile_options")}
    if "ENABLE_FTS5" not in options:
        raise RuntimeError(
            "Ban SQLite cua Python nay khong bat FTS5. Tang B khong chay duoc."
        )


def connect(path: str | Path, check_same_thread: bool = True) -> sqlite3.Connection:
    """check_same_thread=False khi dung trong FastAPI: endpoint chay trong
    threadpool nen connection bi cham tu nhieu thread khac nhau. An toan o day
    vi phia API chi DOC."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    check_fts5(conn)
    conn.executescript(_SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def open_db(path: str | Path, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = connect(path, check_same_thread=check_same_thread)
    ensure_schema(conn)
    return conn


# --- ghi -------------------------------------------------------------------


def insert_ocr(conn: sqlite3.Connection, rows: Iterable[tuple[str, int, str, float]]) -> int:
    """rows: (video_id, frame_idx, text, confidence). `text_norm` duoc sinh o day.

    Sinh text_norm tap trung tai mot cho de khong bao gio co dong nao lot vao DB
    ma chua chuan hoa - do la kieu loi lam query khong tim thay gi ma khong bao loi.
    """
    payload = [
        (video_id, int(frame_idx), text, normalize_vi(text), float(confidence))
        for video_id, frame_idx, text, confidence in rows
    ]
    conn.executemany(
        "INSERT INTO ocr(video_id, frame_idx, text, text_norm, confidence) VALUES (?,?,?,?,?)",
        payload,
    )
    return len(payload)


def mark_done(conn: sqlite3.Connection, video_id: str, n_rows: int, engine: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ocr_done(video_id, n_rows, engine) VALUES (?,?,?)",
        (video_id, int(n_rows), engine),
    )


def delete_video(conn: sqlite3.Connection, video_id: str) -> int:
    """Xoa ket qua cu cua mot video truoc khi chay lai (trigger tu don FTS)."""
    cur = conn.execute("DELETE FROM ocr WHERE video_id = ?", (video_id,))
    conn.execute("DELETE FROM ocr_done WHERE video_id = ?", (video_id,))
    return cur.rowcount


def done_videos(conn: sqlite3.Connection) -> set[str]:
    return {row[0] for row in conn.execute("SELECT video_id FROM ocr_done")}


# --- doc -------------------------------------------------------------------


def to_fts_query(text: str, phrase: bool = True) -> str:
    """Bien input nguoi dung thanh bieu thuc MATCH an toan.

    phrase=True -> tim dung cum chu lien nhau ("cum chu"), dung voi hanh vi thuc
    te la nguoi dung go lai dong chu vua nhin thay tren man hinh.
    phrase=False -> moi tu la mot dieu kien AND, khong bat buoc lien nhau.

    Dau nhay kep trong input duoc nhan doi de khong pha cu phap FTS5.
    """
    normalized = normalize_vi(text).strip()
    if not normalized:
        raise ValueError("Query OCR rong")
    if phrase:
        return '"' + normalized.replace('"', '""') + '"'
    return " ".join('"' + tok.replace('"', '""') + '"' for tok in normalized.split())


def search_text(
    conn: sqlite3.Connection,
    query: str,
    *,
    phrase: bool = True,
    min_confidence: float = 0.0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Full-text search tren text_norm. Tra ve kem diem BM25.

    bm25() cua SQLite tra ve so AM, cang am cang khop. Doi dau tai day de moi noi
    khac trong he thong deu theo quy uoc "diem cao = tot hon".
    """
    sql = """
        SELECT o.video_id, o.frame_idx, o.text, o.confidence,
               -bm25(ocr_fts) AS bm25
        FROM ocr_fts
        JOIN ocr o ON o.rowid = ocr_fts.rowid
        WHERE ocr_fts MATCH ? AND o.confidence >= ?
        ORDER BY bm25 DESC
    """
    params: list[Any] = [to_fts_query(query, phrase), float(min_confidence)]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return [dict(row) for row in conn.execute(sql, params)]


def matching_frames(
    conn: sqlite3.Connection,
    query: str,
    *,
    phrase: bool = True,
    min_confidence: float = 0.0,
) -> set[tuple[str, int]]:
    """Tap (video_id, frame_idx) co chua cum chu - dau vao cua hard filter (Task 8)."""
    return {
        (row["video_id"], row["frame_idx"])
        for row in search_text(conn, query, phrase=phrase, min_confidence=min_confidence)
    }


def frame_texts(conn: sqlite3.Connection, video_id: str, frame_idx: int) -> list[dict[str, Any]]:
    """Toan bo chu doc duoc tren mot keyframe - de hien thi cho nguoi dung."""
    rows = conn.execute(
        "SELECT text, confidence FROM ocr WHERE video_id = ? AND frame_idx = ? "
        "ORDER BY confidence DESC",
        (video_id, int(frame_idx)),
    )
    return [dict(row) for row in rows]


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    return {
        "rows": conn.execute("SELECT COUNT(*) FROM ocr").fetchone()[0],
        "frames": conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT video_id, frame_idx FROM ocr)"
        ).fetchone()[0],
        "videos": conn.execute("SELECT COUNT(*) FROM ocr_done").fetchone()[0],
    }
