"""B.1 - ghep toan bo luong search, xuat top-100.

    query -> encode 2 model -> search FAISS x2 -> RRF -> [hard filter] -> top-100

THU TU QUAN TRONG: RRF chay tren TOAN BO ung vien (khong cat), roi moi loc, roi
moi cat top-100. Neu cat 100 truoc rồi loc thi ket qua tra ve se it hon 100 mot
cach vo ly - trong khi van con thua ung vien hop le o hang 101 tro di.

Giai doan 2 se chen temporal re-rank vao NGAY TRUOC buoc cat top-100.
"""

from __future__ import annotations

from typing import Any

from pathlib import Path
import numpy as np

from aic.retrieval.filters import FilterResult, apply_filter, ocr_allowed_idxs
from aic.retrieval.fusion import DEFAULT_RRF_K, FusedHit, reciprocal_rank_fusion
from aic.retrieval.search import IndexBundle


from collections import defaultdict
import cv2

def check_color(img_bgr, box_xyxy, color_name: str) -> bool:
    if not color_name or color_name.lower() == "none":
        return True
    
    x1, y1, x2, y2 = map(int, box_xyxy)
    h, w = img_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    bw, bh = x2 - x1, y2 - y1
    if bw <= 4 or bh <= 4:
        return False
        
    # Crop central 50%
    cx1 = x1 + bw // 4
    cx2 = x2 - bw // 4
    cy1 = y1 + bh // 4
    cy2 = y2 - bh // 4
    
    crop = img_bgr[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return False
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    color_name = color_name.lower()
    
    masks = []
    if color_name == "red":
        masks.append(cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])))
        masks.append(cv2.inRange(hsv, np.array([160, 50, 50]), np.array([179, 255, 255])))
    elif color_name == "orange":
        masks.append(cv2.inRange(hsv, np.array([10, 50, 50]), np.array([25, 255, 255])))
    elif color_name == "yellow":
        masks.append(cv2.inRange(hsv, np.array([25, 50, 50]), np.array([35, 255, 255])))
    elif color_name == "green":
        masks.append(cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255])))
    elif color_name == "blue":
        masks.append(cv2.inRange(hsv, np.array([90, 50, 50]), np.array([130, 255, 255])))
    elif color_name == "purple":
        masks.append(cv2.inRange(hsv, np.array([130, 50, 50]), np.array([160, 255, 255])))
    elif color_name == "black":
        masks.append(cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 255, 50])))
    elif color_name == "white":
        masks.append(cv2.inRange(hsv, np.array([0, 0, 200]), np.array([179, 30, 255])))
    elif color_name == "gray":
        masks.append(cv2.inRange(hsv, np.array([0, 0, 50]), np.array([179, 40, 200])))
    else:
        return True
        
    final_mask = masks[0]
    for m in masks[1:]:
        final_mask = cv2.bitwise_or(final_mask, m)
        
    total_pixels = final_mask.size
    matched = cv2.countNonZero(final_mask)
    
    return (matched / total_pixels) >= 0.20

def temporal_search(
    bundle: IndexBundle,
    encoder,
    sub_queries: list[str],
    top_k_per_model: int,
    rrf_k: int,
    top_n: int,
    allowed_idxs: set[int] | None,
    max_gap_frames: int = 1500,
    yolo_model: Any = None,
    keyframes_dir: str | None = None,
    od_filters: list[dict[str, Any]] | None = None,
) -> list[FusedHit]:
    all_hits = []
    # Search each sub-query deeply
    for q in sub_queries:
        vectors = encoder.encode(text=q, image_rgb=None)
        ranked_lists = bundle.search(vectors, top_k_per_model)
        fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=None)
        fused = apply_filter(fused, allowed_idxs)
        # Keep top 500 per sub-query for matching to avoid losing items
        all_hits.append(fused[:500])

    video_hits = defaultdict(lambda: [[] for _ in range(len(sub_queries))])
    for q_idx, hits in enumerate(all_hits):
        for hit in hits:
            entry = bundle.entry(hit.idx)
            video_hits[entry.video_id][q_idx].append((hit, entry.frame_idx))

    valid_sequences = []
    for video_id, q_hits in video_hits.items():
        if any(not lst for lst in q_hits):
            continue
        
        for lst in q_hits:
            lst.sort(key=lambda x: x[1])

        def dfs(current_q, current_path):
            if current_q == len(sub_queries):
                valid_sequences.append(list(current_path))
                return
            
            prev_frame = current_path[-1][1] if current_path else -1
            first_frame = current_path[0][1] if current_path else -1
            
            for hit, frame_idx in q_hits[current_q]:
                if frame_idx > prev_frame:
                    if first_frame == -1 or (frame_idx - first_frame <= max_gap_frames):
                        current_path.append((hit, frame_idx))
                        dfs(current_q + 1, current_path)
                        current_path.pop()

        dfs(0, [])

    scored_seqs = []
    for seq in valid_sequences:
        total_score = sum(item[0].score for item in seq)
        scored_seqs.append((total_score, seq))

    scored_seqs.sort(key=lambda x: x[0], reverse=True)

    final_hits = []
    for score, seq in scored_seqs:
        first_hit = seq[0][0]
        merged_ranks = {}
        for i, (hit, _) in enumerate(seq):
            for model, r in hit.ranks.items():
                merged_ranks[f"Q{i+1}_{model}"] = r
        
        seq_idxs = [hit.idx for hit, _ in seq]
        final_hits.append(FusedHit(idx=first_hit.idx, score=score, rank=0, ranks=merged_ranks, sequence_idxs=seq_idxs))
        if not od_filters and len(final_hits) >= top_n:
            break
    
    if od_filters and yolo_model and keyframes_dir:
        valid_hits = []
        labels = [f["label"] for f in od_filters]
        
        for hit in final_hits:
            seq_idxs = hit.sequence_idxs if hit.sequence_idxs else [hit.idx]
            seq_valid = False
            for idx in seq_idxs:
                entry = bundle.entry(idx)
                img_path = str(Path(keyframes_dir) / entry.path)
                results = yolo_model(img_path, conf=0.15, iou=0.5, verbose=False)
                img_bgr = results[0].orig_img
                
                filter_counts = [0] * len(od_filters)
                
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    label = results[0].names[cls_id]
                    
                    for i, f in enumerate(od_filters):
                        if label == f["label"]:
                            req_color = f.get("color")
                            if check_color(img_bgr, box.xyxy[0].cpu().numpy(), req_color):
                                filter_counts[i] += 1
                
                frame_valid = True
                for i, f in enumerate(od_filters):
                    req_count = int(f["count"])
                    actual = filter_counts[i]
                    lower = max(1, req_count - 1)
                    upper = req_count + 1
                    if not (lower <= actual <= upper):
                        frame_valid = False
                        break
                
                if frame_valid:
                    seq_valid = True
                    break  # Chi can it nhat 1 frame trong chuoi thoa man
            
            if seq_valid:
                valid_hits.append(hit)
                if len(valid_hits) >= top_n:
                    break
        
        final_hits = valid_hits
    else:
        final_hits = final_hits[:top_n]

    return [FusedHit(idx=h.idx, score=h.score, rank=i, ranks=h.ranks, sequence_idxs=h.sequence_idxs) for i, h in enumerate(final_hits, 1)]


def search(
    bundle: IndexBundle,
    encoder,
    *,
    text: str | None = None,
    image_rgb: np.ndarray | None = None,
    top_k_per_model: int = 1000,
    rrf_k: int = DEFAULT_RRF_K,
    top_n: int = 100,
    allowed_idxs: set[int] | None = None,
    yolo_model: Any = None,
    keyframes_dir: str | None = None,
    od_filters: list[dict[str, Any]] | None = None,
) -> list[FusedHit]:
    """Chay het luong B.1 cho mot query, tra ve top-N.

    allowed_idxs: tap idx duoc phep di tiep (ket qua hard filter). None = khong loc.
    top_n mac dinh 100 - gioi han nop bai cua ban to chuc.
    """
    if text and '\n' in text.strip():
        sub_queries = [q.strip() for q in text.split('\n') if q.strip()]
        if len(sub_queries) > 1:
            return temporal_search(
                bundle, encoder, sub_queries, top_k_per_model, rrf_k, top_n, allowed_idxs,
                yolo_model=yolo_model,
                keyframes_dir=keyframes_dir,
                od_filters=od_filters
            )

    vectors = encoder.encode(text=text, image_rgb=image_rgb)
    ranked_lists = bundle.search(vectors, top_k_per_model)

    fused = reciprocal_rank_fusion(ranked_lists, k=rrf_k, top_n=None)
    fused = apply_filter(fused, allowed_idxs)
    return fused[:top_n]


def search_with_ocr(
    bundle: IndexBundle,
    encoder,
    conn,
    *,
    ocr_query: str,
    ocr_phrase: bool = True,
    ocr_min_confidence: float = 0.3,
    **kwargs: Any,
) -> tuple[list[FusedHit], FilterResult]:
    """Nhu `search` nhung co them dieu kien OCR. Tra ve ca so lieu cua filter.

    Tra kem FilterResult de cho goi biet vi sao ket qua it - do filter qua chat
    hay do that su khong co gi khop.
    """
    result = ocr_allowed_idxs(
        conn, bundle, ocr_query, phrase=ocr_phrase, min_confidence=ocr_min_confidence
    )
    hits = search(bundle, encoder, allowed_idxs=result.allowed, **kwargs)
    return hits, result


def search_from_config(bundle: IndexBundle, encoder, cfg: Any, **kwargs) -> list[FusedHit]:
    """Nhu `search` nhung lay tham so mac dinh tu config."""
    kwargs.setdefault("top_k_per_model", cfg.retrieval.top_k_per_model)
    kwargs.setdefault("rrf_k", cfg.retrieval.rrf_k)
    kwargs.setdefault("top_n", cfg.retrieval.final_top_n)
    return search(bundle, encoder, **kwargs)


def hydrate(bundle: IndexBundle, hits: list[FusedHit]) -> list[dict[str, Any]]:
    """Gan metadata manifest + thu hang o tung model vao ket qua da gop."""
    rows = []
    for hit in hits:
        entry = bundle.entry(hit.idx)
        d = {
            "idx": hit.idx,
            "video_id": entry.video_id,
            "frame_idx": entry.frame_idx,
            "pts_time": entry.pts_time,
            "path": entry.path,
            "score": hit.score,
            "rank": hit.rank,
            "ranks": hit.ranks,
        }
        if hit.sequence_idxs:
            d["sequence_idxs"] = hit.sequence_idxs
        rows.append(d)
    return rows
