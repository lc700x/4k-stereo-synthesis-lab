from __future__ import annotations

import os
from typing import Literal

import torch
import torch.nn.functional as F

OutputFormat = Literal["half_sbs", "full_sbs"]


def ensure_bchw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"{name} must be CHW or BCHW, got shape {tuple(x.shape)}")


def ensure_b1hw(depth: torch.Tensor) -> torch.Tensor:
    if depth.ndim == 2:
        return depth.unsqueeze(0).unsqueeze(0)
    if depth.ndim == 3:
        return depth.unsqueeze(1)
    if depth.ndim == 4 and depth.shape[1] == 1:
        return depth
    raise ValueError(f"depth must be HW, BHW, or B1HW, got shape {tuple(depth.shape)}")


def match_depth(depth: torch.Tensor, height: int, width: int) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    if depth.shape[-2:] == (height, width):
        return depth
    return F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)


def make_sbs(left: torch.Tensor, right: torch.Tensor, output_format: OutputFormat, fused: bool = True) -> torch.Tensor:
    left = ensure_bchw(left, name="left")
    right = ensure_bchw(right, name="right")
    if left.shape != right.shape:
        raise ValueError(f"left and right shapes must match, got {left.shape} and {right.shape}")

    if output_format == "full_sbs":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_full_sbs":
            from .output_triton import make_full_sbs

            return make_full_sbs(left, right)
        return torch.cat([left, right], dim=-1)

    if output_format == "half_sbs":
        if sbs_backend(left, right, output_format, fused=fused) == "triton_half_sbs":
            from .output_triton import make_half_sbs

            return make_half_sbs(left, right)
        h, w = left.shape[-2:]
        left_w = max(1, w // 2)
        right_w = max(1, w - left_w)
        left_half = F.interpolate(left, size=(h, left_w), mode="bilinear", align_corners=False)
        right_half = F.interpolate(right, size=(h, right_w), mode="bilinear", align_corners=False)
        return torch.cat([left_half, right_half], dim=-1)

    raise ValueError(f"unknown output_format: {output_format}")


def sbs_backend(left: torch.Tensor, right: torch.Tensor, output_format: OutputFormat, fused: bool = True) -> str:
    if output_format == "full_sbs" and (not fused or _triton_disabled_by_env()):
        return "torch_cat"
    if output_format == "half_sbs" and (not fused or _triton_disabled_by_env()):
        return "torch_interpolate"
    if output_format not in {"half_sbs", "full_sbs"}:
        return "torch_interpolate"
    try:
        from .output_triton import can_use_triton_full_sbs, can_use_triton_half_sbs
    except Exception:
        return "torch_cat" if output_format == "full_sbs" else "torch_interpolate"
    if output_format == "full_sbs":
        return "triton_full_sbs" if can_use_triton_full_sbs(left, right) else "torch_cat"
    return "triton_half_sbs" if can_use_triton_half_sbs(left, right) else "torch_interpolate"


def _triton_disabled_by_env() -> bool:
    return os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}


def to_uint8_image(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().clamp(0, 1)
    return (x * 255.0).round().to(torch.uint8)
