"""Manifest keyframe - NGUON SU THAT cho thu tu row (A.6).

Bat bien phai giu bang moi gia:

    row i cua FAISS index CLIP
      == row i cua FAISS index SigLIP2
      == manifest[i]

Vi khong dung DB vector nen khong con primary key. Ca hai builder index PHAI
doc dung file manifest nay, DUNG THU TU, khong sort lai, khong loc them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

FIELDS = ("idx", "video_id", "frame_idx", "pts_time", "path")


@dataclass(frozen=True)
class KeyframeEntry:
    idx: int          # vi tri row trong FAISS index, 0..N-1, LIEN TUC
    video_id: str
    frame_idx: int    # so thu tu frame trong video goc
    pts_time: float   # giay, dung cho temporal search / can chinh ASR
    path: str         # duong dan anh keyframe, tuong doi so voi paths.keyframes


def write_manifest(path: str | Path, entries: Iterable[KeyframeEntry]) -> int:
    """Ghi manifest CSV. Tra ve so dong da ghi.

    Gan lai idx theo dung thu tu ghi de idx luon lien tuc va khop row FAISS.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for entry in entries:
            row = asdict(entry)
            row["idx"] = n
            writer.writerow(row)
            n += 1
    return n


def iter_manifest(path: str | Path) -> Iterator[KeyframeEntry]:
    """Doc manifest theo dong. Dung khi encode de khong nap het vao RAM."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Manifest sai cot: {reader.fieldnames} != {list(FIELDS)}")
        for expected_idx, row in enumerate(reader):
            entry = KeyframeEntry(
                idx=int(row["idx"]),
                video_id=row["video_id"],
                frame_idx=int(row["frame_idx"]),
                pts_time=float(row["pts_time"]),
                path=row["path"],
            )
            if entry.idx != expected_idx:
                raise ValueError(
                    f"Manifest lech ID tai dong {expected_idx}: idx={entry.idx}. "
                    "idx phai lien tuc 0..N-1 va khop row FAISS."
                )
            yield entry


def load_manifest(path: str | Path) -> list[KeyframeEntry]:
    return list(iter_manifest(path))


def write_index_meta(path: str | Path, *, n_manifest: int, ntotal: dict[str, int]) -> None:
    """Ghi so dong manifest + ntotal cua tung FAISS index de assert luc load."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n_manifest": n_manifest, "ntotal": ntotal}, f, indent=2)


def assert_alignment(n_manifest: int, ntotal: dict[str, int]) -> None:
    """Goi MOI LAN load index. Lech mot dong la sai toan bo ket qua."""
    bad = {name: n for name, n in ntotal.items() if n != n_manifest}
    if bad:
        raise AssertionError(
            f"FAISS index lech voi manifest ({n_manifest} dong): {bad}. "
            "Build lai index tu dung file manifest nay."
        )


def check_alignment_from_meta(meta_path: str | Path, manifest_path: str | Path) -> None:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    n_now = sum(1 for _ in iter_manifest(manifest_path))
    if n_now != meta["n_manifest"]:
        raise AssertionError(
            f"Manifest da doi sau khi build index: {n_now} != {meta['n_manifest']}"
        )
    assert_alignment(n_now, meta["ntotal"])


def group_by_video(entries: Sequence[KeyframeEntry]) -> dict[str, list[KeyframeEntry]]:
    """Dung cho temporal search (B.2) - gom keyframe theo video_id."""
    out: dict[str, list[KeyframeEntry]] = {}
    for e in entries:
        out.setdefault(e.video_id, []).append(e)
    return out
