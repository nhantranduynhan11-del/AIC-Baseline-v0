"""Xac dinh so chieu embedding cua model open_clip.

Khong co MOT cach duy nhat dung cho moi model:

  - `model.visual.output_dim` chi co o `VisionTransformer` cua open_clip. SigLIP2
    dung vision tower cua timm (`TimmModel`), lop do KHONG co thuoc tinh nay ->
    AttributeError.
  - `open_clip.get_model_config(name)["embed_dim"]` co cho moi model dung ten
    trong registry, nhung khong co neu load tu local-dir/hf-hub tuy bien.
  - Do thang bang mot lan forward thi luon dung, nhung ton mot lan chay model.

Nen thu theo thu tu tren, va chi forward khi hai cach dau deu khong ra.

So chieu duoc chot MOT LAN luc khoi tao encoder, khong tinh lai moi lan goi:
sai so chieu la sai ca FAISS index, phai lo ra ngay luc nap model chu khong phai
giua chung khi da encode duoc nua chung.
"""

from __future__ import annotations

from typing import Any, Callable


def dim_from_attribute(model: Any) -> int | None:
    """Duong nhanh: VisionTransformer cua open_clip co san output_dim."""
    visual = getattr(model, "visual", None)
    if visual is None:
        return None
    value = getattr(visual, "output_dim", None)
    return int(value) if isinstance(value, int) and value > 0 else None


def dim_from_config(name: str) -> int | None:
    """Doc embed_dim tu config cua open_clip - dung cho ca model nen timm."""
    try:
        import open_clip

        config = open_clip.get_model_config(name)
    except Exception:
        return None
    if not isinstance(config, dict):
        return None
    value = config.get("embed_dim")
    return int(value) if isinstance(value, int) and value > 0 else None


def resolve_embed_dim(model: Any, name: str, probe: Callable[[], int] | None = None) -> int:
    """Tra ve so chieu embedding. `probe` la cach do bang mot lan forward.

    Raise neu ca ba cach deu that bai - tha dung han con hon di tiep voi so chieu
    doan mo.
    """
    for value in (dim_from_attribute(model), dim_from_config(name)):
        if value:
            return value

    if probe is not None:
        value = probe()
        if value:
            return int(value)

    raise RuntimeError(
        f"Khong xac dinh duoc so chieu embedding cua '{name}'. "
        "Kiem tra ten model co dung trong open_clip khong."
    )
