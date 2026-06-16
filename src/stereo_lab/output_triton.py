from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _half_sbs_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = x < half_width
    src_x_out = tl.where(use_left, x, x - half_width)
    src_x = (src_x_out.to(tl.float32) + 0.5) * (width / half_width) - 0.5
    src_x_clamped = tl.minimum(tl.maximum(src_x, 0.0), width - 1.0)
    x0_float = tl.floor(src_x_clamped)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = src_x_clamped - x0_float
    base = channel * pixels + y * width

    left_v0 = tl.load(left + base + x0, mask=active & use_left, other=0.0)
    left_v1 = tl.load(left + base + x1, mask=active & use_left, other=0.0)
    right_v0 = tl.load(right + base + x0, mask=active & ~use_left, other=0.0)
    right_v1 = tl.load(right + base + x1, mask=active & ~use_left, other=0.0)
    value = tl.where(use_left, left_v0 + (left_v1 - left_v0) * frac, right_v0 + (right_v1 - right_v0) * frac)
    tl.store(out + offsets, value, mask=active)


def can_use_triton_half_sbs(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.is_cuda
        and right.is_cuda
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
        and left.shape[-1] % 2 == 0
    )


def make_half_sbs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    half_width = width // 2
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _half_sbs_kernel[grid](left, right, out, total, width, half_width, pixels, block)
    return out
