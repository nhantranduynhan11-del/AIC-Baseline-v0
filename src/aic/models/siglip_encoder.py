"""SigLIP2 ViT-L-16-SigLIP2-256 (webli) - load bang open_clip.

Sigmoid loss thay vi softmax contrastive nhu CLIP truyen thong -> nhay hon voi
chi tiet nho. Hai model bo tro nhau, gop bang RRF o B.1.

DA XAC MINH tren open_clip:
  - Ten dung la "ViT-L-16-SigLIP2-256", KHONG phai "ViT-L-16-256".
  - embed_dim = 1024 (CLIP la 768) -> hai khong gian KHAC NHAU, khong bao gio
    tron vector giua chung, va phai nam o hai FAISS index rieng.
  - anh vao 256x256 (CLIP 224), context_length 64 (CLIP 77).
  - vision tower la timm model -> can `timm`; tokenizer la HF tokenizer
    (timm/ViT-L-16-SigLIP2-256, vocab 256k kieu Gemma) -> can `transformers`
    va `sentencepiece`.

Giong ClipEncoder: encode_* LUON tra ve vector da normalize L2, vi FAISS
IndexFlatIP chi bang cosine khi ca index lan query deu da normalize.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MODEL_NAME = "ViT-L-16-SigLIP2-256"
PRETRAINED = "webli"


class SiglipEncoder:
    """Boc open_clip cho SigLIP2. Cung giao dien voi ClipEncoder."""

    def __init__(self, device: str = "auto", name: str = MODEL_NAME, pretrained: str = PRETRAINED):
        import open_clip
        import torch

        self.torch = torch
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        self.name = name
        self.pretrained = pretrained

        model, _, preprocess = open_clip.create_model_and_transforms(
            name, pretrained=pretrained, device=self.device
        )
        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(name)

    @property
    def dim(self) -> int:
        return int(self.model.visual.output_dim)

    def encode_images(self, images_rgb: Sequence[np.ndarray]) -> np.ndarray:
        """images_rgb: list anh numpy uint8 HxWx3 RGB. Tra ve (N, D) float32 da normalize."""
        from PIL import Image

        if not images_rgb:
            return np.zeros((0, self.dim), dtype=np.float32)

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
