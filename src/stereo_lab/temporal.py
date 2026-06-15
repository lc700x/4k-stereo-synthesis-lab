from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TemporalState:
    left: torch.Tensor | None = None
    right: torch.Tensor | None = None
    mask: torch.Tensor | None = None

    def reset(self) -> None:
        self.left = None
        self.right = None
        self.mask = None


def apply_temporal(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor | None,
    state: TemporalState | None,
    strength: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state is None:
        return left, right
    if state.left is None or state.right is None or state.left.shape != left.shape or state.right.shape != right.shape:
        state.left = left.detach()
        state.right = right.detach()
        state.mask = mask.detach() if mask is not None else None
        return left, right

    alpha = float(max(0.0, min(strength, 0.98)))
    if mask is None:
        blend = alpha
    else:
        blend = (mask * alpha).to(device=left.device, dtype=left.dtype)
    left_out = left * (1.0 - blend) + state.left.to(left.device) * blend
    right_out = right * (1.0 - blend) + state.right.to(right.device) * blend
    state.left = left_out.detach()
    state.right = right_out.detach()
    state.mask = mask.detach() if mask is not None else None
    return left_out, right_out
