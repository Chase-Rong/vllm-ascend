"""Swap-memory helpers for Mooncake PD receive targets.

``torch_npu.empty_with_swapped_memory`` returns an NPU tensor backed by
host-side swap memory.  Mooncake cannot reliably write that allocation
directly with AscendDirect, so the connector uses this registry to identify
staged receive destinations and copies the data from an NPU staging buffer
into the swapped allocation after each transfer.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch

_SWAPPED_TENSORS: list[tuple[int, int, torch.Tensor]] = []


def iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield tensors from a tensor/list/tuple cache structure."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_tensors(item)


def register_swapped_tensor(tensor: torch.Tensor) -> None:
    """Keep the allocation alive and resolve transfer sub-ranges to it."""
    ptr = int(tensor.data_ptr())
    size = int(tensor.numel() * tensor.element_size())
    if ptr <= 0 or size <= 0:
        raise ValueError(f"Invalid swapped tensor range: ptr={ptr}, size={size}")
    if not any(start == ptr and length == size for start, length, _ in _SWAPPED_TENSORS):
        _SWAPPED_TENSORS.append((ptr, size, tensor))


def get_swapped_tensor(ptr: int, size: int) -> tuple[torch.Tensor, int] | None:
    """Return the owning swapped tensor and byte offset for a sub-range."""
    ptr = int(ptr)
    size = int(size)
    end = ptr + size
    for start, length, tensor in _SWAPPED_TENSORS:
        if ptr >= start and end <= start + length:
            return tensor, ptr - start
    return None


def is_swapped_range(ptr: int, size: int) -> bool:
    return get_swapped_tensor(ptr, size) is not None


def clear_swapped_tensors_for_testing() -> None:
    _SWAPPED_TENSORS.clear()


def empty_swapped_memory(shape: tuple[int, ...], *, dtype: torch.dtype) -> torch.Tensor:
    """Allocate zero-initialized NPU tensor storage in host-side swap memory."""
    try:
        import torch_npu
    except ImportError as exc:
        raise RuntimeError("Mooncake swap-memory receive requires torch_npu.empty_with_swapped_memory.") from exc

    allocator = getattr(torch_npu, "empty_with_swapped_memory", None)
    if allocator is None:
        raise RuntimeError("Mooncake swap-memory receive requires torch_npu.empty_with_swapped_memory.")
    tensor = allocator(shape, dtype=dtype, device="npu")
    # The replaced KV-cache path used torch.zeros. Do not depend on the
    # version-specific observation that swapped memory starts zeroed.
    tensor.zero_()
    return tensor
