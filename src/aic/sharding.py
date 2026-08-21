"""Chia danh sách video thành nhiều phần rời nhau.

Dùng khi một máy có nhiều GPU (Kaggle 2×T4): mỗi tiến trình nhận một phần và
một GPU riêng qua CUDA_VISIBLE_DEVICES.

Chia theo kiểu XEN KẼ (round-robin) chứ không cắt thành khối liền nhau: độ dài
video chênh nhau nhiều, cắt khối dễ rơi vào cảnh một GPU ôm toàn video dài còn
GPU kia xong sớm rồi ngồi không. Xen kẽ thì trung bình hai bên cân nhau.

Phép chia chỉ phụ thuộc THỨ TỰ ĐÃ SẮP XẾP của danh sách, nên hai tiến trình
chạy độc lập vẫn ra đúng hai phần rời nhau mà không cần nói chuyện với nhau.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def parse_shard(value: str) -> tuple[int, int]:
    """Đọc chuỗi dạng "0/2" thành (0, 2)."""
    text = value.strip()
    if "/" not in text:
        raise ValueError(f"--shard phải có dạng I/N, ví dụ 0/2. Nhận '{value}'")

    left, right = text.split("/", 1)
    try:
        index, total = int(left), int(right)
    except ValueError:
        raise ValueError(f"--shard phải là hai số nguyên I/N. Nhận '{value}'") from None

    if total < 1:
        raise ValueError(f"--shard: N phải >= 1, nhận {total}")
    if not 0 <= index < total:
        raise ValueError(f"--shard: I phải trong khoảng 0..{total - 1}, nhận {index}")
    return index, total


def select_shard(items: Sequence[T], shard: str | None) -> list[T]:
    """Lấy phần thứ I trong N phần. shard=None thì trả về nguyên danh sách."""
    if shard is None:
        return list(items)
    index, total = parse_shard(shard)
    return [item for position, item in enumerate(items) if position % total == index]
