"""Ep stdout/stderr sang UTF-8.

Console Windows mac dinh dung code page he thong (cp1252 o may dev cua nhom),
nen `print` mot chuoi tieng Viet co dau se nem UnicodeEncodeError va lam chet
script - du ban than du lieu hoan toan dung. Cac script co in text OCR / ten
video deu goi ham nay ngay sau khi khoi dong.

errors="replace": tha in ra dau hoi con hon lam dut ca me xu ly chi vi mot ky tu
la trong ten file.
"""

from __future__ import annotations

import sys


def use_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass          # stream da bi thay the (pytest capture, pipe...) - bo qua
