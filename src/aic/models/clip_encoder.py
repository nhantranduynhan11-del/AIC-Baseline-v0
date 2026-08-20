"""CLIP ViT-L-14-quickgelu (DFN2B) - load bang open_clip.

Model NAY duoc dung o CA BA cho va PHAI encode y het nhau:
  - A.2 keyframe extraction (trich dac trung moi 8 frame, so L2)
  - A.3 indexing (dung LAI embedding cua keyframe da chon, KHONG encode lai)
  - B.1 encode query luc search
Vi vay cach preprocess anh/text chi duoc dinh nghia o day, mot cho duy nhat.
Sua o day la sua cho ca ba - dung bao gio viet preprocess rieng o cho khac.

⚠️ encode_images/encode_texts LUON tra ve vector da NORMALIZE L2. Do la yeu cau
cua ca hai phia:
  - A.2: nguong L2 = 0.4 chi co y nghia tren vector don vi (khoang cach L2 nam
    trong [0, 2], va d^2 = 2 - 2*cos). Vector CLIP tho co norm ~10 thi nguong 0.4
    vo nghia.
  - A.6: FAISS IndexFlatIP -> tich vo huong chi bang cosine khi ca hai da normalize.

torch/open_clip import LAZY de module nay import duoc tren may khong cai torch.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aic.models.embed_dim import resolve_embed_dim

MODEL_NAME = "ViT-L-14-quickgelu"
PRETRAINED = "dfn2b"


class ClipEncoder:
    """Boc open_clip. Giu model + preprocess di cung nhau."""

    def __init__(self, device: str = "auto", name: str = MODEL_NAME, pretrained: str = PRETRAINED):
        import open_clip
        import torch

        self.torch = torch
        self.device = _resolve_device(device, torch)
        self.name = name
        self.pretrained = pretrained

        model, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=self.device
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(name)
        # Chot so chieu NGAY luc nap: sai so chieu la sai ca FAISS index, phai lo
        # ra o day chu khong phai giua chung khi da encode duoc nua chung.
        self._dim = resolve_embed_dim(model, name, probe=self._probe_dim)

    def _probe_dim(self) -> int:
        """Do so chieu bang mot lan forward anh gia - duong cuoi khi khong doc
        duoc tu thuoc tinh lan config."""
        import numpy as np

        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        return int(self._encode_images_raw([dummy]).shape[1])

    @property
    def dim(self) -> int:
        return self._dim

    def encode_images(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """images_rgb: list anh numpy uint8 HxWx3 RGB. Tra ve (N, D) float32 da normalize."""
        from PIL import Image

        if not images_rgb:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode_images_raw(images_rgb)

    def _encode_images_raw(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """Phan forward thuan tuy. Tach ra de _probe_dim goi duoc ma khong cham
        vao self.dim (luc do self._dim chua ton tai)."""
        from PIL import Image

        batch = self.torch.stack(
            [self.preprocess(Image.fromarray(img)) for img in images_rgb]
        ).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Tra ve (N, D) float32 da normalize. Dung o B.1."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        tokens = self.tokenizer(list(texts)).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def _resolve_device(device: str, torch) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"
