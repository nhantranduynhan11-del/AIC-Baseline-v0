"""B.1 buoc 2 - encode query DOC LAP bang ca hai model.

Phai dung DUNG hai model da dung luc index (ViT-L-14-quickgelu/dfn2b va
ViT-L-16-SigLIP2-256/webli). Dung model khac thi khong gian embedding khong khop
voi vector da index - ket qua van tra ve binh thuong, chi la vo nghia.

Vi vay lop nay khong tu chon model: no doc ten tu config va tai dung ClipEncoder
/ SiglipEncoder - cung hai lop da dung o A.2 va A.3, nen cach preprocess anh/text
la mot.

Nhan ca hai kieu query:
  - text  -> Textual KIS
  - anh   -> Video KIS (query bang khung hinh)

Tra ve dict {"clip": (D_clip,), "siglip2": (D_siglip,)} - hai vector RIENG BIET,
khac so chieu, khong bao gio ghep lai voi nhau.
"""

from __future__ import annotations

from typing import Any

import numpy as np

MODEL_KEYS = ("clip", "siglip2")


class QueryEncoder:
    """Giu ca hai model, encode mot query thanh hai vector rieng."""

    def __init__(self, cfg: Any, device: str | None = None):
        from aic.models.clip_encoder import ClipEncoder
        from aic.models.siglip_encoder import SiglipEncoder

        device = device or cfg.runtime.device
        self.clip = ClipEncoder(
            device=device, name=cfg.models.clip.name, pretrained=cfg.models.clip.pretrained
        )
        self.siglip = SiglipEncoder(
            device=device, name=cfg.models.siglip2.name, pretrained=cfg.models.siglip2.pretrained
        )
        self.encoders = {"clip": self.clip, "siglip2": self.siglip}

    @property
    def dims(self) -> dict[str, int]:
        return {key: enc.dim for key, enc in self.encoders.items()}

    def encode_text(self, text: str) -> dict[str, np.ndarray]:
        if not text or not text.strip():
            raise ValueError("Query text rong")
        
        try:
            import translators as ts
            translated = None
            try:
                translated = ts.translate_text(text, translator='google', from_language='vi', to_language='en')
            except Exception:
                pass
            
            if not translated:
                translated = ts.translate_text(text, translator='bing', from_language='vi', to_language='en')

            print(f"[QueryEncoder] Dich: '{text}' -> '{translated}'")
            text_to_encode = translated if translated else text
        except Exception as e:
            print(f"[QueryEncoder] Loi dich: {e}. Dung nguyen ban.")
            text_to_encode = text

        self.last_translated_text = text_to_encode
        return {key: enc.encode_texts([text_to_encode])[0] for key, enc in self.encoders.items()}

    def encode_image(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        """image_rgb: numpy uint8 HxWx3 RGB (KHONG phai BGR cua cv2)."""
        if image_rgb is None or image_rgb.ndim != 3:
            raise ValueError(f"Can anh HxWx3, nhan {None if image_rgb is None else image_rgb.shape}")
        return {key: enc.encode_images([image_rgb])[0] for key, enc in self.encoders.items()}

    def encode(self, text: str | None = None, image_rgb: np.ndarray | None = None):
        self.last_translated_text = None
        if (text is None) == (image_rgb is None):
            raise ValueError("Truyen dung mot trong hai: text hoac image_rgb")
        return self.encode_text(text) if text is not None else self.encode_image(image_rgb)
