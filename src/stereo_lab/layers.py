from __future__ import annotations

import torch
import torch.nn.functional as F

from .output import ensure_b1hw


def make_depth_layers(depth: torch.Tensor, layers: int = 2, softness: float = 0.08) -> torch.Tensor:
    if layers < 1:
        raise ValueError("layers must be >= 1")
    depth = ensure_b1hw(depth).clamp(0, 1)
    if layers == 1:
        return torch.ones_like(depth)

    centers = torch.linspace(0.0, 1.0, layers, device=depth.device, dtype=depth.dtype).view(1, layers, 1, 1)
    depth_blhw = depth.expand(-1, layers, -1, -1)
    weights = torch.exp(-((depth_blhw - centers) ** 2) / max(softness, 1e-4))
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)


def composite_layers(warped: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
    if not warped:
        raise ValueError("warped layer list is empty")
    weights = weights[:, : len(warped)]
    out = torch.zeros_like(warped[0])
    for idx, layer in enumerate(warped):
        out = out + layer * weights[:, idx : idx + 1]
    return out


def depth_edges(depth: torch.Tensor, threshold: float = 0.04) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    dx = F.pad((depth[..., :, 1:] - depth[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((depth[..., 1:, :] - depth[..., :-1, :]).abs(), (0, 0, 0, 1))
    return ((dx + dy) > threshold).float()
