from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .output import ensure_b1hw, ensure_bchw, match_depth


@dataclass(frozen=True)
class DepthSafetyConfig:
    enabled: bool = True
    flat_depth_value: float = 0.5
    min_force_flat_triggers: int = 2
    low_texture_threshold: float = 0.020
    large_flat_area_threshold: float = 0.88
    thumbnail_grid_threshold: float = 0.45
    edge_alignment_threshold: float = 0.22
    center_bias_threshold: float = 0.08
    stabilize_low_texture_background: bool = True
    background_flat_area_threshold: float = 0.55
    background_smoothing_kernel: int = 21
    background_stabilization_strength: float = 0.65


@dataclass(frozen=True)
class DepthSafetyDecision:
    use_depth: bool
    reason: str
    force_flat_depth: bool
    flatten_strength: float
    depth_strength_scale: float
    confidence: float
    metrics: dict[str, float]
    triggers: tuple[str, ...]
    background_stabilization: bool = False
    background_stabilization_strength: float = 0.0

    def to_report(self) -> dict[str, Any]:
        report = asdict(self)
        report["triggers"] = list(self.triggers)
        return report


def evaluate_depth_safety(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: DepthSafetyConfig | None = None,
) -> DepthSafetyDecision:
    config = config or DepthSafetyConfig()
    rgb = ensure_bchw(rgb, name="rgb").float()
    depth = match_depth(ensure_b1hw(depth).float(), rgb.shape[-2], rgb.shape[-1])

    if not config.enabled:
        return DepthSafetyDecision(
            use_depth=True,
            reason="depth safety disabled",
            force_flat_depth=False,
            flatten_strength=0.0,
            depth_strength_scale=1.0,
            confidence=0.0,
            metrics={},
            triggers=(),
        )

    gray = rgb.mean(dim=1, keepdim=True)
    rgb_grad = _gradient_magnitude(gray)
    depth_grad = _gradient_magnitude(depth)

    rgb_texture_score = float(rgb_grad.mean().detach().cpu())
    large_flat_area_ratio = float((rgb_grad < 0.015).float().mean().detach().cpu())
    rgb_edge_density = float((rgb_grad > 0.075).float().mean().detach().cpu())
    depth_edge_density = float((depth_grad > 0.035).float().mean().detach().cpu())
    thumbnail_grid_score = _thumbnail_grid_score(rgb_grad)
    depth_variance = float(depth.var(unbiased=False).detach().cpu())
    center_bias_score = _center_bias_score(depth)
    depth_edge_rgb_edge_alignment = _edge_alignment(rgb_grad, depth_grad)

    metrics = {
        "rgb_texture_score": rgb_texture_score,
        "large_flat_area_ratio": large_flat_area_ratio,
        "rgb_edge_density": rgb_edge_density,
        "depth_edge_density": depth_edge_density,
        "thumbnail_grid_score": thumbnail_grid_score,
        "depth_variance": depth_variance,
        "center_bias_score": center_bias_score,
        "depth_edge_rgb_edge_alignment": depth_edge_rgb_edge_alignment,
    }

    triggers: list[str] = []
    if rgb_texture_score < config.low_texture_threshold and large_flat_area_ratio > 0.70:
        triggers.append("low_texture")
    if large_flat_area_ratio >= config.large_flat_area_threshold:
        triggers.append("large_flat_area")
    if thumbnail_grid_score >= config.thumbnail_grid_threshold and large_flat_area_ratio > 0.55:
        triggers.append("thumbnail_grid")
    if (
        depth_edge_density > 0.015
        and rgb_edge_density > 0.005
        and depth_edge_rgb_edge_alignment < config.edge_alignment_threshold
    ):
        triggers.append("depth_rgb_edge_mismatch")
    if center_bias_score >= config.center_bias_threshold and large_flat_area_ratio > 0.65:
        triggers.append("center_bias")

    force_flat = (
        "thumbnail_grid" in triggers
        or (
            "large_flat_area" in triggers
            and ("low_texture" in triggers or "center_bias" in triggers or "depth_rgb_edge_mismatch" in triggers)
            and large_flat_area_ratio > 0.95
            and rgb_edge_density < 0.006
        )
        or (
            "low_texture" in triggers
            and "center_bias" in triggers
            and large_flat_area_ratio > 0.92
            and rgb_edge_density < 0.010
        )
    )
    if force_flat:
        confidence = min(1.0, 0.45 + 0.18 * len(triggers) + max(0.0, thumbnail_grid_score - 0.45))
        return DepthSafetyDecision(
            use_depth=False,
            reason="force flat depth for unsafe still image content",
            force_flat_depth=True,
            flatten_strength=1.0,
            depth_strength_scale=0.0,
            confidence=confidence,
            metrics=metrics,
            triggers=tuple(triggers),
            background_stabilization=False,
            background_stabilization_strength=0.0,
        )

    if triggers:
        stabilize_background = _should_stabilize_background(config, large_flat_area_ratio, triggers)
        return DepthSafetyDecision(
            use_depth=True,
            reason="depth accepted with low-texture background stabilization" if stabilize_background else "depth accepted with low-confidence warning",
            force_flat_depth=False,
            flatten_strength=0.0,
            depth_strength_scale=1.0,
            confidence=min(0.65, 0.30 + 0.12 * len(triggers)),
            metrics=metrics,
            triggers=tuple(triggers),
            background_stabilization=stabilize_background,
            background_stabilization_strength=float(config.background_stabilization_strength) if stabilize_background else 0.0,
        )

    return DepthSafetyDecision(
        use_depth=True,
        reason="depth accepted for still image content",
        force_flat_depth=False,
        flatten_strength=0.0,
        depth_strength_scale=1.0,
        confidence=0.0,
        metrics=metrics,
        triggers=(),
        background_stabilization=False,
        background_stabilization_strength=0.0,
    )


def apply_depth_safety(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: DepthSafetyConfig | None = None,
) -> tuple[torch.Tensor, DepthSafetyDecision]:
    config = config or DepthSafetyConfig()
    rgb = ensure_bchw(rgb, name="rgb").float()
    depth = match_depth(ensure_b1hw(depth).float(), rgb.shape[-2], rgb.shape[-1])
    decision = evaluate_depth_safety(rgb, depth, config)
    if not config.enabled:
        return depth, decision

    adjusted = depth
    if decision.flatten_strength > 0.0:
        flat = torch.full_like(depth, float(config.flat_depth_value))
        strength = max(0.0, min(1.0, float(decision.flatten_strength)))
        adjusted = adjusted.lerp(flat, strength)
    if decision.background_stabilization:
        adjusted = _stabilize_low_texture_background(rgb, adjusted, config)
    return adjusted, decision


def _gradient_magnitude(tensor: torch.Tensor) -> torch.Tensor:
    dx = tensor[..., :, 1:] - tensor[..., :, :-1]
    dy = tensor[..., 1:, :] - tensor[..., :-1, :]
    dx = torch.nn.functional.pad(dx.abs(), (0, 1, 0, 0))
    dy = torch.nn.functional.pad(dy.abs(), (0, 0, 0, 1))
    return (dx + dy).clamp(0, 1)


def _thumbnail_grid_score(rgb_grad: torch.Tensor) -> float:
    edge = (rgb_grad > 0.08).float()
    if edge.shape[-1] < 16 or edge.shape[-2] < 16:
        return 0.0
    col_density = edge.mean(dim=(-3, -2)).flatten()
    row_density = edge.mean(dim=(-3, -1)).flatten()
    col_peaks = _count_projection_peaks(col_density, threshold=0.18)
    row_peaks = _count_projection_peaks(row_density, threshold=0.18)
    peak_score = min(1.0, (col_peaks + row_peaks) / 12.0)
    balance = min(col_peaks, row_peaks) / max(1, max(col_peaks, row_peaks))
    return float(peak_score * (0.5 + 0.5 * balance))


def _count_projection_peaks(values: torch.Tensor, *, threshold: float) -> int:
    mask = (values > threshold).detach().cpu().tolist()
    count = 0
    in_peak = False
    for item in mask:
        if item and not in_peak:
            count += 1
            in_peak = True
        elif not item:
            in_peak = False
    return count


def _center_bias_score(depth: torch.Tensor) -> float:
    h, w = depth.shape[-2:]
    y0, y1 = h // 3, max(h // 3 + 1, (h * 2) // 3)
    x0, x1 = w // 3, max(w // 3 + 1, (w * 2) // 3)
    center = depth[..., y0:y1, x0:x1].mean()
    corner_h = max(1, h // 4)
    corner_w = max(1, w // 4)
    corners = torch.cat(
        [
            depth[..., :corner_h, :corner_w].flatten(),
            depth[..., :corner_h, -corner_w:].flatten(),
            depth[..., -corner_h:, :corner_w].flatten(),
            depth[..., -corner_h:, -corner_w:].flatten(),
        ]
    )
    return float((center - corners.mean()).abs().detach().cpu())


def _edge_alignment(rgb_grad: torch.Tensor, depth_grad: torch.Tensor) -> float:
    rgb_edge = rgb_grad > 0.075
    depth_edge = depth_grad > 0.035
    union = (rgb_edge | depth_edge).float().sum()
    if float(union.detach().cpu()) <= 0.0:
        return 1.0
    overlap = (rgb_edge & depth_edge).float().sum()
    return float((overlap / union.clamp_min(1.0)).detach().cpu())


def _should_stabilize_background(config: DepthSafetyConfig, large_flat_area_ratio: float, triggers: list[str]) -> bool:
    if not config.stabilize_low_texture_background:
        return False
    if large_flat_area_ratio < config.background_flat_area_threshold:
        return False
    return "low_texture" in triggers or "center_bias" in triggers or "depth_rgb_edge_mismatch" in triggers


def _stabilize_low_texture_background(rgb: torch.Tensor, depth: torch.Tensor, config: DepthSafetyConfig) -> torch.Tensor:
    gray = rgb.mean(dim=1, keepdim=True)
    rgb_grad = _gradient_magnitude(gray)
    low_texture = (rgb_grad < config.low_texture_threshold).float()
    edge_guard = _dilate((rgb_grad > 0.045).float(), radius=5)
    mask = (low_texture * (1.0 - edge_guard)).clamp(0, 1)
    if float(mask.mean().detach().cpu()) <= 0.01:
        return depth

    kernel = max(3, int(config.background_smoothing_kernel) | 1)
    pad = kernel // 2
    padded = F.pad(depth, (pad, pad, pad, pad), mode="reflect")
    smoothed = F.avg_pool2d(padded, kernel_size=kernel, stride=1)
    strength = max(0.0, min(1.0, float(config.background_stabilization_strength)))
    return depth.lerp(smoothed, mask * strength).clamp(0, 1)


def _dilate(mask: torch.Tensor, *, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)
