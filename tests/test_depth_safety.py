import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.depth_safety import apply_depth_safety, evaluate_depth_safety


def test_depth_safety_accepts_natural_image_with_texture():
    rgb = _natural_image()
    depth = _smooth_depth()

    decision = evaluate_depth_safety(rgb, depth)

    assert decision.use_depth is True
    assert decision.force_flat_depth is False
    assert decision.depth_strength_scale == 1.0
    assert "thumbnail_grid" not in decision.triggers


def test_depth_safety_forces_flat_for_thumbnail_grid():
    rgb = _thumbnail_grid()
    depth = _smooth_depth(center_bias=True)

    adjusted, decision = apply_depth_safety(rgb, depth)

    assert decision.force_flat_depth is True
    assert decision.use_depth is False
    assert "thumbnail_grid" in decision.triggers
    assert float(adjusted.var(unbiased=False)) < 1e-6


def test_depth_safety_flattens_low_texture_center_bias():
    rgb = torch.full((1, 3, 64, 64), 0.12)
    depth = _smooth_depth(center_bias=True)

    adjusted, decision = apply_depth_safety(rgb, depth)

    assert decision.force_flat_depth is True
    assert "large_flat_area" in decision.triggers
    assert "center_bias" in decision.triggers
    assert float(adjusted.var(unbiased=False)) < float(depth.var(unbiased=False))


def test_depth_safety_stabilizes_low_texture_background_without_scaling_depth():
    rgb = torch.full((1, 3, 64, 64), 0.35)
    rgb[:, :, 20:44, 24:40] = torch.tensor([0.10, 0.75, 0.18]).view(1, 3, 1, 1)
    rgb[:, :, 20:21, 24:40] = 0.95
    rgb[:, :, 43:44, 24:40] = 0.02
    rgb[:, :, 20:44, 24:25] = 0.95
    rgb[:, :, 20:44, 39:40] = 0.02
    depth = _noisy_background_depth()

    adjusted, decision = apply_depth_safety(rgb, depth)

    assert decision.force_flat_depth is False
    assert decision.depth_strength_scale == 1.0
    assert decision.background_stabilization is True
    background = (slice(None), slice(None), slice(0, 16), slice(0, 16))
    subject = (slice(None), slice(None), slice(24, 40), slice(28, 36))
    assert float(adjusted[background].var(unbiased=False)) < float(depth[background].var(unbiased=False))
    assert torch.mean(torch.abs(adjusted[subject] - depth[subject])) < 0.02


def _natural_image() -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    y = torch.linspace(0, 1, 64).view(1, 1, 64, 1)
    x = torch.linspace(0, 1, 64).view(1, 1, 1, 64)
    texture = torch.rand((1, 1, 64, 64), generator=generator) * 0.20
    r = (x * 0.65 + y * 0.15 + texture).clamp(0, 1)
    g = (y * 0.55 + x * 0.20 + texture * 0.8).clamp(0, 1)
    b = ((x + y) * 0.25 + texture * 1.2).clamp(0, 1)
    return torch.cat([r, g, b], dim=1)


def _thumbnail_grid() -> torch.Tensor:
    rgb = torch.full((1, 3, 64, 64), 0.08)
    for y in range(4, 64, 16):
        for x in range(4, 64, 16):
            rgb[:, :, y : y + 10, x : x + 10] = torch.tensor([0.75, 0.25, 0.15]).view(1, 3, 1, 1)
            rgb[:, :, y : y + 1, x : x + 10] = 1.0
            rgb[:, :, y + 9 : y + 10, x : x + 10] = 1.0
            rgb[:, :, y : y + 10, x : x + 1] = 1.0
            rgb[:, :, y : y + 10, x + 9 : x + 10] = 1.0
    return rgb


def _smooth_depth(center_bias: bool = False) -> torch.Tensor:
    y = torch.linspace(-1, 1, 64).view(1, 1, 64, 1)
    x = torch.linspace(-1, 1, 64).view(1, 1, 1, 64)
    if center_bias:
        radius = (x.square() + y.square()).sqrt().clamp(0, 1)
        return 0.75 - radius * 0.35
    return (x + 1.0) * 0.5


def _noisy_background_depth() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    y = torch.linspace(-1, 1, 64).view(1, 1, 64, 1)
    x = torch.linspace(-1, 1, 64).view(1, 1, 1, 64)
    background = 0.55 + torch.rand((1, 1, 64, 64), generator=generator) * 0.12
    subject = 0.72 - (x.square() + y.square()).sqrt().clamp(0, 1) * 0.12
    depth = background.clone()
    depth[:, :, 20:44, 24:40] = subject[:, :, 20:44, 24:40]
    return depth.clamp(0, 1)
