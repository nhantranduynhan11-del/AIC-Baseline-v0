"""FastAPI app - tang API cho UI (B.1, giai doan 1).

Chay:
    uvicorn aic.api.app:app --app-dir src --host 0.0.0.0 --port 8000

Tang nay co tinh MONG: moi logic tim kiem nam o aic.retrieval.pipeline, moi logic
dinh dang nop bai nam o aic.submit.export. O day chi lam ba viec: nhan HTTP, goi
xuong, tra ve JSON.

Endpoint:
    GET  /health            trang thai + cau hinh dang chay
    POST /search            query -> top-100 (co the kem OCR filter)
    GET  /keyframe/{idx}    anh keyframe do phan giai goc
    GET  /thumb/{idx}       thumbnail cho grid (roi ve anh goc neu chua sinh)
    GET  /neighbors/{idx}   keyframe lan can trong CUNG video, theo thoi gian
    GET  /ocr/{idx}         chu doc duoc tren keyframe
    POST /export            danh sach da chon -> CSV dung dinh dang nop bai
    GET  /                  UI (web/index.html)
"""

from __future__ import annotations

import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from aic.api.state import AppState, can_load
from aic.config import load_config

cfg = load_config()
_state: AppState | None = None
_load_error: str = "Chua nap"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Nap model + index mot lan luc startup. Thieu file thi khong crash, chi ghi ly do."""
    global _state, _load_error

    ok, reason = can_load(cfg)
    if not ok:
        _load_error = reason
        print(f"[API] Khong nap duoc state: {reason}")
    else:
        try:
            print("[API] Dang nap index va 2 model...")
            _state = AppState.load(cfg)
            print(
                f"[API] San sang: {_state.bundle.ntotal} keyframe, "
                f"dim={_state.bundle.dims()}, OCR={'co' if _state.has_ocr else 'chua co'}"
            )
        except Exception as exc:  # noqa: BLE001 - muon API van len de bao loi ro rang
            _load_error = f"{type(exc).__name__}: {exc}"
            print(f"[API] Nap that bai: {_load_error}")
    yield


app = FastAPI(title="AIC 2026 - HCMUT Technologia Retrieval", version="0.1.0", lifespan=lifespan)


def get_state() -> AppState:
    if _state is None:
        raise HTTPException(status_code=503, detail=_load_error)
    return _state


# --- schema ---------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str | None = Field(default=None, description="Cau truy van text (Textual KIS)")
    image_idx: int | None = Field(
        default=None, description="Dung keyframe nay lam query (Video KIS)"
    )
    top_k_per_model: int | None = Field(default=None, ge=1)
    top_n: int = Field(default=100, ge=1, le=100, description="Gioi han nop bai la 100")
    ocr: str | None = Field(default=None, description="Cum chu phai co tren man hinh")
    ocr_phrase: bool = True
    ocr_min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class SearchHit(BaseModel):
    idx: int
    video_id: str
    frame_idx: int
    pts_time: float
    score: float
    rank: int
    ranks: dict[str, int]


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    n_hits: int
    ocr_filter: dict[str, Any] | None = None
    translated_query: str | None = None


class ExportItem(BaseModel):
    video_id: str
    frame_idx: int | None = None
    frames: list[int] | None = None       # chi dung cho TRAKE


class ExportRequest(BaseModel):
    task: Literal["kis", "qa", "trake"]
    items: list[ExportItem]
    answer: str | None = Field(default=None, max_length=100)
    n_events: int | None = None           # so su kien TRAKE yeu cau, de kiem tra


# --- endpoint -------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    ready = _state is not None
    out: dict[str, Any] = {
        "ready": ready,
        "clip": cfg.models.clip.name,
        "siglip2": cfg.models.siglip2.name,
        "top_k_per_model": cfg.retrieval.top_k_per_model,
        "rrf_k": cfg.retrieval.rrf_k,
        "final_top_n": cfg.retrieval.final_top_n,
    }
    if ready:
        out |= {
            "n_keyframes": _state.bundle.ntotal,
            "dims": _state.bundle.dims(),
            "ocr": _state.has_ocr,
        }
    else:
        out["error"] = _load_error
    return out


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, state: AppState = Depends(get_state)) -> SearchResponse:
    from aic.retrieval import pipeline
    from aic.retrieval.filters import ocr_allowed_idxs

    if (req.query is None) == (req.image_idx is None):
        raise HTTPException(400, "Truyen dung mot trong hai: query hoac image_idx")

    allowed = None
    filter_info = None
    if req.ocr:
        if not state.has_ocr:
            raise HTTPException(400, "Chua co DB OCR. Chay scripts/04_ocr.py truoc.")
        min_conf = (
            req.ocr_min_confidence
            if req.ocr_min_confidence is not None
            else state.cfg.ocr.query_min_confidence
        )
        result = ocr_allowed_idxs(
            state.conn, state.bundle, req.ocr, phrase=req.ocr_phrase, min_confidence=min_conf
        )
        allowed = result.allowed
        filter_info = {
            "query": req.ocr,
            "n_rows": result.n_rows,
            "n_frames": result.n_frames,
            "n_unknown": result.n_unknown,
        }

    image_rgb = None
    if req.image_idx is not None:
        image_rgb = _read_keyframe_rgb(state, req.image_idx)

    with state.lock:
        hits = pipeline.search(
            state.bundle,
            state.encoder,
            text=req.query,
            image_rgb=image_rgb,
            top_k_per_model=req.top_k_per_model or state.cfg.retrieval.top_k_per_model,
            rrf_k=state.cfg.retrieval.rrf_k,
            top_n=req.top_n,
            allowed_idxs=allowed,
        )

    rows = pipeline.hydrate(state.bundle, hits)
    return SearchResponse(
        hits=[SearchHit(**{k: v for k, v in row.items() if k != "path"}) for row in rows],
        n_hits=len(rows),
        ocr_filter=filter_info,
        translated_query=getattr(state.encoder, "last_translated_text", None),
    )


@app.get("/keyframe/{idx}")
def keyframe(idx: int, state: AppState = Depends(get_state)) -> FileResponse:
    return FileResponse(_keyframe_path(state, idx), media_type="image/jpeg")


@app.get("/thumb/{idx}")
def thumb(idx: int, state: AppState = Depends(get_state)) -> FileResponse:
    """Thumbnail cho grid. Chua sinh thumbnail thi roi ve anh goc.

    Roi ve nhu vay de UI chay duoc ngay ca khi chua chay scripts/05_thumbnails.py,
    chi la cuon grid se nang.
    """
    entry = _entry(state, idx)
    thumb_path = Path(state.cfg.paths.thumbs) / entry.path
    if thumb_path.exists():
        return FileResponse(thumb_path, media_type="image/jpeg")
    return FileResponse(_keyframe_path(state, idx), media_type="image/jpeg")


@app.get("/neighbors/{idx}")
def neighbors(idx: int, w: int = 5, state: AppState = Depends(get_state)) -> dict[str, Any]:
    """Keyframe lan can trong CUNG video, theo thu tu thoi gian.

    Manifest xep lien tuc theo video va tang dan theo frame_idx, nen chi can di
    sang trai/phai chung nao con cung video_id.
    """
    entry = _entry(state, idx)
    entries = state.bundle.entries

    start = idx
    while start > 0 and entries[start - 1].video_id == entry.video_id and idx - start < w:
        start -= 1
    end = idx
    last = len(entries) - 1
    while end < last and entries[end + 1].video_id == entry.video_id and end - idx < w:
        end += 1

    return {
        "video_id": entry.video_id,
        "current": idx,
        "items": [
            {
                "idx": e.idx,
                "frame_idx": e.frame_idx,
                "pts_time": e.pts_time,
                "is_current": e.idx == idx,
            }
            for e in entries[start : end + 1]
        ],
    }


@app.get("/ocr/{idx}")
def ocr(idx: int, state: AppState = Depends(get_state)) -> dict[str, Any]:
    entry = _entry(state, idx)
    if not state.has_ocr:
        return {"idx": idx, "available": False, "texts": []}

    from aic.store import sqlite_store as store

    texts = store.frame_texts(state.conn, entry.video_id, entry.frame_idx)
    return {"idx": idx, "available": True, "texts": texts}


@app.post("/export", response_class=PlainTextResponse)
def export_csv(req: ExportRequest, state: AppState = Depends(get_state)) -> PlainTextResponse:
    """Ghi ra file tam roi doc lai - de dung chung duong code (va validate) voi CLI."""
    from aic.submit import export as ex

    if not req.items:
        raise HTTPException(400, "Danh sach rong")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"submission_{req.task}.csv"
        try:
            if req.task == "kis":
                ex.write_kis(path, [(i.video_id, _need_frame(i)) for i in req.items])
            elif req.task == "qa":
                if not req.answer:
                    raise HTTPException(400, "Task qa can `answer`")
                ex.write_qa(
                    path, [(i.video_id, _need_frame(i), req.answer) for i in req.items]
                )
            else:
                ex.write_trake(
                    path,
                    [(i.video_id, i.frames or []) for i in req.items],
                    n_events=req.n_events,
                )
            ex.validate_file(path, req.task, n_events=req.n_events)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        # Doc BYTE roi tu decode: read_text() bat universal newlines va se
        # bien CRLF thanh LF, lam hong dinh dang file nop bai.
        content = path.read_bytes().decode("utf-8")

    return PlainTextResponse(
        content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="submission_{req.task}.csv"'},
    )


# --- helper ---------------------------------------------------------------


def _entry(state: AppState, idx: int):
    if idx < 0 or idx >= state.bundle.ntotal:
        raise HTTPException(404, f"idx {idx} ngoai pham vi 0..{state.bundle.ntotal - 1}")
    return state.bundle.entry(idx)


def _keyframe_path(state: AppState, idx: int) -> Path:
    path = Path(state.cfg.paths.keyframes) / _entry(state, idx).path
    if not path.exists():
        raise HTTPException(404, f"Khong tim thay anh: {path}")
    return path


def _read_keyframe_rgb(state: AppState, idx: int):
    import cv2

    image = cv2.imread(str(_keyframe_path(state, idx)))
    if image is None:
        raise HTTPException(404, f"Khong doc duoc anh cua idx {idx}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _need_frame(item: ExportItem) -> int:
    if item.frame_idx is None:
        raise HTTPException(400, f"{item.video_id}: thieu frame_idx")
    return item.frame_idx


# --- UI -------------------------------------------------------------------
# Mount SAU CUNG: StaticFiles o "/" bat moi duong dan con lai, nen phai dang ky
# sau tat ca endpoint API thi cac route tren moi duoc uu tien.
_WEB_DIR = Path(__file__).resolve().parents[3] / "web"
if _WEB_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
