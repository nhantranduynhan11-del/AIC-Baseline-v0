"""Chuan hoa text tieng Viet cho OCR/ASR (A.4).

Luu 2 dang:
  - text      : nguyen van CO dau, de hien thi cho nguoi dung
  - text_norm : bo dau + d->d, de SEARCH

Ly do phai tu lam thay vi de FTS5 lo: tokenizer
`unicode61 remove_diacritics 2` bo duoc dau nguyen am nhung KHONG xu ly
`d`/`D` (khong phai dau to hop).
"""

from __future__ import annotations

import unicodedata


def normalize_vi(text: str) -> str:
    """Bo dau tieng Viet, d->d, lowercase. Dung cho ca luc index lan luc query."""
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D").lower()
