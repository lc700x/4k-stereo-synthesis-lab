from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
TOOLS = ROOT / "scripts" / "tools"
for path in (SRC, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cinema_ipd64_quality_sweep import (  # noqa: E402
    DEFAULT_PRODUCTION_ENGINE,
    DEFAULT_PRODUCTION_ONNX,
    edge_mask,
    image_gradient,
    load_depth_for_sweep,
    make_sequence,
    masked_mean,
    resize_max_width,
    save_contact_sheet,
    tensor_mae,
)
from stereo_runtime.io import load_rgb, save_depth, save_rgb  # noqa: E402
from stereo_runtime.synthesis import StereoConfig, synthesize_stereo  # noqa: E402
from stereo_runtime.temporal import TemporalState  # noqa: E402
from utils.settings import read_yaml  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class QualityThresholds:
    max_occlusion_mask_ratio: float = 0.085
    max_occlusion_edge_overlap: float = 0.20
    max_edge_source_mae: float = 0.070
    max_edge_gradient_delta: float = 0.065
    max_ghost_risk_score: float = 8.0
    max_hole_risk_score: float = 10.0
    min_stereo_score: float = 2.0
    max_temporal_reset_rate: float = 0.15


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _setting_float(settings: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def _setting_int(settings: dict[str, Any], key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return default


def _normalize_backend(value: Any) -> str:
    raw = str(value or "quality_4k").strip().lower()
    aliases = {
        "synthetic view": "quality_4k",
        "quality": "quality_4k",
        "quality_4k": "quality_4k",
        "hq_4k": "hq_4k",
        "fast_plus": "fast_plus",
        "fast": "fast",
    }
    return aliases.get(raw, raw if raw in {"quality_4k", "hq_4k", "fast_plus", "fast"} else "quality_4k")


def runtime_ipd_from_settings(settings: dict[str, Any]) -> tuple[float, str]:
    raw = settings.get("IPD mm", settings.get("IPD (mm)", settings.get("IPD", 0.032)))
    value = float(raw)
    if value <= 1.0:
        return value, "settings IPD interpreted as runtime meters"
    return value / 1000.0, "settings IPD interpreted as runtime millimeters"


def build_stereo_config(settings: dict[str, Any]) -> tuple[StereoConfig, dict[str, Any]]:
    runtime_ipd_m, ipd_note = runtime_ipd_from_settings(settings)
    backend = _normalize_backend(settings.get("Stereo Quality", settings.get("Synthetic View", "quality_4k")))
    hole_fill_mode = str(settings.get("Hole Fill Mode", "balanced"))
    config = StereoConfig(
        backend=backend,
        layers=2,
        output_format="half_sbs",
        debug_output=True,
        temporal=_to_bool(settings.get("Temporal"), True),
        fused=True,
        depth_strength=_setting_float(settings, "Depth Strength", 2.5),
        convergence=_setting_float(settings, "Convergence", 0.25),
        ipd=runtime_ipd_m,
        ipd_mm=runtime_ipd_m * 1000.0,
        stereo_scale=_setting_float(settings, "Stereo Scale", 0.3),
        max_shift_ratio=_setting_float(settings, "Max Shift Ratio", 0.03),
        temporal_strength=_setting_float(settings, "Temporal Strength", 0.85),
        auto_reset_temporal=_to_bool(settings.get("Auto Scene Reset"), True),
        scene_reset_threshold=_setting_float(settings, "Scene Reset Threshold", 0.22),
        foreground_scale=_setting_float(settings, "Foreground Scale", 0.5),
        depth_antialias_strength=_setting_float(settings, "Depth Antialias Strength", _setting_float(settings, "Anti-aliasing", 1.0)),
        edge_dilation=_setting_int(settings, "Edge Dilation", 2),
        mask_feather_radius=_setting_int(settings, "Mask Feather Radius", 3),
        hole_fill_mode=hole_fill_mode,
        hole_fill_radius=_setting_int(settings, "Hole Fill Radius", 3),
        hole_fill_strength=_setting_float(settings, "Hole Fill Strength", 1.0),
        edge_threshold=_setting_float(settings, "Edge Threshold", 0.04),
        cross_eyed=_to_bool(settings.get("Cross Eyed"), False),
        anaglyph_method=str(settings.get("Anaglyph Method", "red_cyan")),
    )
    return config, {
        "ipd_note": ipd_note,
        "backend_source": settings.get("Stereo Quality", settings.get("Synthetic View", "quality_4k")),
    }


def find_sample_images(samples_dir: Path, include_private: bool = False) -> list[Path]:
    images = []
    for path in sorted(samples_dir.iterdir()):
        if path.is_dir():
            if include_private or path.name.lower() != "private":
                images.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTS))
            continue
        if path.suffix.lower() in IMAGE_EXTS:
            images.append(path)
    return images


def make_depth_args(args: argparse.Namespace, settings: dict[str, Any]) -> SimpleNamespace:
    depth_source = args.depth_source
    if depth_source == "auto":
        depth_source = "production" if _to_bool(settings.get("TensorRT"), True) else "proxy"
    return SimpleNamespace(
        depth_source=depth_source,
        depth_onnx=args.depth_onnx,
        depth_engine=args.depth_engine,
        profile_sync=bool(args.profile_sync),
        depth_upsample=args.depth_upsample,
        depth_upsample_edge_strength=float(args.depth_upsample_edge_strength),
    )


def evaluate_sample(
    image_path: Path,
    config: StereoConfig,
    depth_args: SimpleNamespace,
    args: argparse.Namespace,
    device: torch.device,
    out_dir: Path,
) -> dict[str, Any]:
    sample_slug = image_path.stem.replace(" ", "_")
    sample_dir = out_dir / sample_slug
    sample_dir.mkdir(parents=True, exist_ok=True)

    source_rgb = resize_max_width(load_rgb(image_path, device=device), int(args.max_width))
    rgb_frames = make_sequence(source_rgb, int(args.sequence_frames), int(args.sequence_shift_px))
    depth_frames = []
    depth_source = ""
    provider_info: dict[str, Any] = {}
    depth_timing_rows = []

    with torch.inference_mode():
        for frame in rgb_frames:
            depth, depth_source, provider_info, timing = load_depth_for_sweep(frame, depth_args)
            depth_frames.append(depth)
            depth_timing_rows.append(timing)

        temporal_state = TemporalState() if config.temporal else None
        result = None
        for frame_rgb, frame_depth in zip(rgb_frames, depth_frames, strict=True):
            result = synthesize_stereo(frame_rgb, frame_depth, config, temporal_state=temporal_state)
        if result is None:
            raise RuntimeError("no sequence frames were synthesized")

    rgb = rgb_frames[-1]
    depth = depth_frames[-1]
    edges = edge_mask(depth, threshold=float(args.edge_mask_threshold))
    shift = result.debug_info.get("shift_px")
    occ = result.debug_info.get("occlusion_mask")
    if not isinstance(shift, torch.Tensor):
        raise RuntimeError(f"missing shift_px for {image_path}")
    if not isinstance(occ, torch.Tensor):
        occ = torch.zeros_like(depth)

    avg_eye = (result.left_eye + result.right_eye) * 0.5
    edge_source_mae = masked_mean(tensor_mae(avg_eye, rgb), edges)
    global_source_mae = float(tensor_mae(avg_eye, rgb).mean().item())
    lr_mae = float(tensor_mae(result.left_eye, result.right_eye).mean().item())
    edge_lr_mae = masked_mean(tensor_mae(result.left_eye, result.right_eye), edges)
    grad_delta = (image_gradient(avg_eye) - image_gradient(rgb)).abs()
    edge_grad_delta = masked_mean(grad_delta, edges)
    occ_ratio = float(occ.float().mean().item())
    occ_edge_overlap = masked_mean(occ.float(), edges)
    max_shift_px = float(shift.abs().max().item())
    p95_shift_px = float(torch.quantile(shift.abs().flatten(), 0.95).item())

    ghost_risk = edge_source_mae * 100.0 + edge_grad_delta * 45.0 + occ_edge_overlap * 8.0
    hole_risk = occ_ratio * 100.0 + edge_grad_delta * 35.0
    stereo_score = edge_lr_mae * 100.0
    realism_score = stereo_score - ghost_risk * 0.75 - hole_risk * 0.35

    sbs_path = sample_dir / f"{sample_slug}_half_sbs.png"
    mask_path = sample_dir / f"{sample_slug}_occlusion_mask.png"
    depth_path = sample_dir / f"{sample_slug}_depth.png"
    save_rgb(result.sbs.detach().cpu(), sbs_path)
    save_depth(occ.detach().cpu(), mask_path)
    save_rgb(depth.expand(-1, 3, -1, -1).detach().cpu(), depth_path)

    return {
        "sample": str(image_path),
        "sample_name": image_path.name,
        "frame_size": {"width": int(rgb.shape[-1]), "height": int(rgb.shape[-2])},
        "depth_source": depth_source,
        "depth_provider": provider_info,
        "depth_timing_sequence": depth_timing_rows,
        "max_shift_px": max_shift_px,
        "p95_shift_px": p95_shift_px,
        "occlusion_mask_ratio": occ_ratio,
        "occlusion_edge_overlap": occ_edge_overlap,
        "global_source_mae": global_source_mae,
        "edge_source_mae": edge_source_mae,
        "left_right_mae": lr_mae,
        "edge_left_right_mae": edge_lr_mae,
        "edge_gradient_delta": edge_grad_delta,
        "ghost_risk_score": ghost_risk,
        "hole_risk_score": hole_risk,
        "stereo_score": stereo_score,
        "realism_score": realism_score,
        "hole_fill_backend": str(result.debug_info.get("hole_fill_backend", "none")),
        "warp_composite_backend": str(result.debug_info.get("warp_composite_backend", "n/a")),
        "scene_delta": float(result.debug_info.get("scene_delta", 0.0)),
        "temporal_reset": int(result.debug_info.get("temporal_reset", 0)),
        "temporal_reset_count": int(result.debug_info.get("temporal_reset_count", 0)),
        "sbs_path": str(sbs_path),
        "occlusion_mask_path": str(mask_path),
        "depth_path": str(depth_path),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        "max_shift_px",
        "p95_shift_px",
        "occlusion_mask_ratio",
        "occlusion_edge_overlap",
        "edge_source_mae",
        "edge_left_right_mae",
        "edge_gradient_delta",
        "ghost_risk_score",
        "hole_risk_score",
        "stereo_score",
        "realism_score",
        "scene_delta",
        "temporal_reset_count",
    ]
    aggregate: dict[str, Any] = {"sample_count": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if key in row]
        if not values:
            continue
        aggregate[key] = {
            "mean": sum(values) / len(values),
            "max": max(values),
            "min": min(values),
        }
    aggregate["worst_ghost_sample"] = max(rows, key=lambda r: float(r["ghost_risk_score"]))["sample_name"] if rows else None
    aggregate["worst_hole_sample"] = max(rows, key=lambda r: float(r["hole_risk_score"]))["sample_name"] if rows else None
    aggregate["best_realism_sample"] = max(rows, key=lambda r: float(r["realism_score"]))["sample_name"] if rows else None
    return aggregate


def build_assessment(rows: list[dict[str, Any]], config: StereoConfig, thresholds: QualityThresholds) -> dict[str, Any]:
    aggregate = aggregate_rows(rows)
    failures: list[str] = []
    recommendations: list[dict[str, str]] = []

    if not rows:
        return {"verdict": "fail", "failures": ["no sample images were evaluated"], "recommendations": []}

    max_occ = aggregate["occlusion_mask_ratio"]["max"]
    max_occ_edge = aggregate["occlusion_edge_overlap"]["max"]
    max_edge_mae = aggregate["edge_source_mae"]["max"]
    max_grad = aggregate["edge_gradient_delta"]["max"]
    max_ghost = aggregate["ghost_risk_score"]["max"]
    max_hole = aggregate["hole_risk_score"]["max"]
    mean_stereo = aggregate["stereo_score"]["mean"]
    reset_rate = sum(1 for row in rows if int(row.get("temporal_reset", 0)) > 0) / len(rows)

    if max_occ > thresholds.max_occlusion_mask_ratio:
        failures.append(f"occlusion_mask_ratio max {max_occ:.4f} > {thresholds.max_occlusion_mask_ratio:.4f}")
        recommendations.append({
            "parameter": "Stereo Scale / Max Shift Ratio / Depth Strength",
            "action": "降低 Stereo Scale 或 Max Shift Ratio；若只在 enhanced/高深度感设置下出现，再降低 Depth Strength。",
            "reason": "遮挡区域过大通常来自视差位移过强。",
        })
    if max_occ_edge > thresholds.max_occlusion_edge_overlap:
        failures.append(f"occlusion_edge_overlap max {max_occ_edge:.4f} > {thresholds.max_occlusion_edge_overlap:.4f}")
        recommendations.append({
            "parameter": "Edge Dilation / Mask Feather Radius / Edge Threshold",
            "action": "先增加 Edge Dilation 到 3 或 Mask Feather Radius 到 4；如果边缘遮罩过宽，再提高 Edge Threshold。",
            "reason": "遮挡集中在深度边缘时，优先调边缘遮罩和羽化。",
        })
    if max_edge_mae > thresholds.max_edge_source_mae or max_grad > thresholds.max_edge_gradient_delta:
        failures.append(
            f"edge artifact max edge_source_mae={max_edge_mae:.4f}, edge_gradient_delta={max_grad:.4f}"
        )
        recommendations.append({
            "parameter": "Depth Antialias Strength / Edge Dilation / Foreground Scale",
            "action": "提高 Depth Antialias Strength；若前景轮廓虚影明显，降低 Foreground Scale 或增加 Edge Dilation。",
            "reason": "边缘源图偏差和梯度变化偏高，说明深度边缘或遮罩过硬。",
        })
    if max_ghost > thresholds.max_ghost_risk_score:
        failures.append(f"ghost_risk_score max {max_ghost:.2f} > {thresholds.max_ghost_risk_score:.2f}")
        recommendations.append({
            "parameter": "Hole Fill Mode / Hole Fill Strength / Stereo Scale",
            "action": "优先把 Hole Fill Mode 调成 soft_low_ghost；仍偏高时降低 Hole Fill Strength 或 Stereo Scale。",
            "reason": "虚影风险高通常来自补洞混合过强或视差过大。",
        })
    if max_hole > thresholds.max_hole_risk_score:
        failures.append(f"hole_risk_score max {max_hole:.2f} > {thresholds.max_hole_risk_score:.2f}")
        recommendations.append({
            "parameter": "Hole Fill Radius / Mask Feather Radius",
            "action": "适度提高 Hole Fill Radius 或 Mask Feather Radius；若画面变糊，再回退半径并降低 Max Shift Ratio。",
            "reason": "补洞风险高说明空洞面积或边缘补偿不足。",
        })
    if mean_stereo < thresholds.min_stereo_score and not failures:
        recommendations.append({
            "parameter": "Stereo Scale / Max Shift Ratio / Convergence",
            "action": "可以小幅提高 Stereo Scale 或 Max Shift Ratio；若舒适度不足，再调 Convergence 到 0.15-0.25。",
            "reason": "当前伪影风险可控但平均立体分离度偏低。",
        })
    if config.temporal and reset_rate > thresholds.max_temporal_reset_rate:
        failures.append(f"temporal reset rate {reset_rate:.2f} > {thresholds.max_temporal_reset_rate:.2f}")
        recommendations.append({
            "parameter": "Scene Reset Threshold / Temporal Strength",
            "action": "若真实视频中频繁重置，提高 Scene Reset Threshold；若拖影明显，降低 Temporal Strength。",
            "reason": "样本平移序列中频繁 reset 表示时序阈值可能过敏。",
        })

    config_notes = []
    if config.max_shift_ratio > 0.03:
        config_notes.append("Max Shift Ratio 高于 0.03 稳态基线，若报告边缘/遮挡风险高，优先回调到 0.03。")
    if config.convergence < 0.10:
        config_notes.append("Convergence 低于电影模式常用 0.15-0.25，若画面缺少舒适收敛点，可先试 0.25。")
    if config.foreground_scale < 0.25:
        config_notes.append("Foreground Scale 偏保守，前景立体感可能不足；只有在风险指标良好时再提高。")
    if config.hole_fill_mode != "soft_low_ghost":
        config_notes.append("Hole Fill Mode 不是 soft_low_ghost；若出现虚影，优先切换到 soft_low_ghost。")

    verdict = "pass" if not failures else "needs_tuning"
    return {
        "verdict": verdict,
        "thresholds": asdict(thresholds),
        "failures": failures,
        "recommendations": recommendations,
        "config_notes": config_notes,
        "aggregate": aggregate,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved cinema stereo settings on all samples images.")
    parser.add_argument("--settings", type=Path, default=ROOT / "src" / "settings.yaml")
    parser.add_argument("--samples", type=Path, default=ROOT / "samples")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "visual_regression" / "saved_cinema_settings_samples")
    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Maximum input width before evaluation; 0 keeps each sample at its original resolution.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--depth-source", choices=["auto", "production", "proxy"], default="auto")
    parser.add_argument("--depth-onnx", type=Path, default=DEFAULT_PRODUCTION_ONNX)
    parser.add_argument("--depth-engine", type=Path, default=DEFAULT_PRODUCTION_ENGINE)
    parser.add_argument("--depth-upsample", choices=["bilinear", "guided"], default="guided")
    parser.add_argument("--depth-upsample-edge-strength", type=float, default=0.35)
    parser.add_argument("--profile-sync", action="store_true")
    parser.add_argument("--sequence-frames", type=int, default=3)
    parser.add_argument("--sequence-shift-px", type=int, default=2)
    parser.add_argument("--edge-mask-threshold", type=float, default=0.08)
    parser.add_argument("--include-private", action="store_true")
    args = parser.parse_args()

    settings = read_yaml(args.settings)
    config, config_meta = build_stereo_config(settings)
    depth_args = make_depth_args(args, settings)
    if depth_args.depth_source == "production" and not torch.cuda.is_available() and (args.device or "cuda") == "cuda":
        raise RuntimeError("production depth requires CUDA; pass --depth-source proxy for script flow testing")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    sample_images = find_sample_images(args.samples, include_private=bool(args.include_private))
    if not sample_images:
        raise FileNotFoundError(f"no sample images found in {args.samples}")

    rows = []
    for image_path in sample_images:
        print(f"[sample] {image_path}")
        rows.append(evaluate_sample(image_path, config, depth_args, args, device, args.out))

    thresholds = QualityThresholds()
    assessment = build_assessment(rows, config, thresholds)
    contact_items = [
        (
            f"{row['sample_name']} ghost={row['ghost_risk_score']:.2f} hole={row['hole_risk_score']:.2f}",
            Path(str(row["sbs_path"])),
        )
        for row in sorted(rows, key=lambda r: float(r["ghost_risk_score"]) + float(r["hole_risk_score"]), reverse=True)[:9]
    ]
    save_contact_sheet(contact_items, args.out / "highest_risk_contact_sheet.png", columns=3)

    report = {
        "settings_path": str(args.settings),
        "samples_dir": str(args.samples),
        "output_dir": str(args.out),
        "depth_source_request": depth_args.depth_source,
        "max_width": int(args.max_width),
        "input_size_policy": "original_resolution" if int(args.max_width) <= 0 else "resize_to_max_width",
        "saved_settings_subset": {
            "Depth Model": settings.get("Depth Model"),
            "Depth Strength": settings.get("Depth Strength"),
            "IPD": settings.get("IPD"),
            "Convergence": settings.get("Convergence"),
            "Stereo Scale": settings.get("Stereo Scale"),
            "Max Shift Ratio": settings.get("Max Shift Ratio"),
            "Foreground Scale": settings.get("Foreground Scale"),
            "Depth Antialias Strength": settings.get("Depth Antialias Strength"),
            "Edge Dilation": settings.get("Edge Dilation"),
            "Edge Threshold": settings.get("Edge Threshold"),
            "Mask Feather Radius": settings.get("Mask Feather Radius"),
            "Hole Fill Mode": settings.get("Hole Fill Mode"),
            "Hole Fill Radius": settings.get("Hole Fill Radius"),
            "Hole Fill Strength": settings.get("Hole Fill Strength"),
            "Temporal": settings.get("Temporal"),
            "Temporal Strength": settings.get("Temporal Strength"),
            "Auto Scene Reset": settings.get("Auto Scene Reset"),
            "Scene Reset Threshold": settings.get("Scene Reset Threshold"),
            "Stereo Quality": settings.get("Stereo Quality"),
            "Synthetic View": settings.get("Synthetic View"),
            "TensorRT": settings.get("TensorRT"),
        },
        "runtime_config": {
            "backend": config.backend,
            "depth_strength": config.depth_strength,
            "ipd_m": config.ipd,
            "ipd_mm": config.ipd_mm,
            "convergence": config.convergence,
            "stereo_scale": config.stereo_scale,
            "max_shift_ratio": config.max_shift_ratio,
            "foreground_scale": config.foreground_scale,
            "depth_antialias_strength": config.depth_antialias_strength,
            "edge_dilation": config.edge_dilation,
            "edge_threshold": config.edge_threshold,
            "mask_feather_radius": config.mask_feather_radius,
            "hole_fill_mode": config.hole_fill_mode,
            "hole_fill_radius": config.hole_fill_radius,
            "hole_fill_strength": config.hole_fill_strength,
            "temporal": config.temporal,
            "temporal_strength": config.temporal_strength,
            "auto_reset_temporal": config.auto_reset_temporal,
            "scene_reset_threshold": config.scene_reset_threshold,
        },
        "config_meta": config_meta,
        "assessment": assessment,
        "rows": rows,
    }
    report_path = args.out / "saved_cinema_settings_samples_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "assessment": assessment}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
