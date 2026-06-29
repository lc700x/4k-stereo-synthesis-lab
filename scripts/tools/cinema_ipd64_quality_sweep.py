from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stereo_runtime.depth_provider import DepthProviderConfig, create_depth_provider  # noqa: E402
from stereo_runtime.io import load_rgb, save_depth, save_rgb  # noqa: E402
from stereo_runtime.openxr_visual_regression import make_depth_proxy_from_rgb  # noqa: E402
from stereo_runtime.synthesis import StereoConfig, synthesize_stereo  # noqa: E402
from stereo_runtime.temporal import TemporalState  # noqa: E402

DEPTH_OPTIONS = {
    "soft": 2.0,
    "standard": 2.5,
    "enhanced": 3.0,
}

DEFAULT_PRODUCTION_MODEL_DIR = ROOT / "src" / "models" / "models--lc700x--Distill-Any-Depth-Base-hf"
DEFAULT_PRODUCTION_ONNX = DEFAULT_PRODUCTION_MODEL_DIR / "model_fp16_294x518.onnx"
DEFAULT_PRODUCTION_ENGINE = DEFAULT_PRODUCTION_MODEL_DIR / "model_fp16_294x518.trt"


def display_ipd_mm_to_runtime_m(display_mm: float) -> float:
    return float(display_mm) * 30.0 / 60.0 / 1000.0


def resize_max_width(rgb: torch.Tensor, max_width: int) -> torch.Tensor:
    if max_width <= 0 or rgb.shape[-1] <= max_width:
        return rgb
    height = max(1, int(round(rgb.shape[-2] * max_width / rgb.shape[-1])))
    return F.interpolate(rgb, size=(height, max_width), mode="area")


def translate_frame(rgb: torch.Tensor, dx_px: int, dy_px: int = 0) -> torch.Tensor:
    if dx_px == 0 and dy_px == 0:
        return rgb
    _, _, height, width = rgb.shape
    pad_x = abs(int(dx_px))
    pad_y = abs(int(dy_px))
    padded = F.pad(rgb, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    x0 = pad_x - int(dx_px)
    y0 = pad_y - int(dy_px)
    return padded[..., y0 : y0 + height, x0 : x0 + width].contiguous()


def make_sequence(rgb: torch.Tensor, frames: int, shift_px: int) -> list[torch.Tensor]:
    frames = max(1, int(frames))
    center = (frames - 1) / 2.0
    return [translate_frame(rgb, int(round((idx - center) * shift_px))) for idx in range(frames)]


def edge_mask(depth: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=depth.dtype,
        device=depth.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=depth.dtype,
        device=depth.device,
    ).view(1, 1, 3, 3)
    edge = F.conv2d(depth, sobel_x, padding=1).abs() + F.conv2d(depth, sobel_y, padding=1).abs()
    edge = F.max_pool2d(edge, kernel_size=9, stride=1, padding=4)
    return edge > threshold


def image_gradient(x: torch.Tensor) -> torch.Tensor:
    luma = x.mean(dim=1, keepdim=True)
    dx = F.pad((luma[..., :, 1:] - luma[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((luma[..., 1:, :] - luma[..., :-1, :]).abs(), (0, 0, 0, 1))
    return dx + dy


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
    if mask.shape[-2:] != x.shape[-2:]:
        mask = F.interpolate(mask.float(), size=x.shape[-2:], mode="nearest") > 0.5
    if not bool(mask.any()):
        return float(x.mean().item())
    return float(x[mask.expand_as(x)].mean().item())


def load_depth_for_sweep(rgb: torch.Tensor, args: argparse.Namespace) -> tuple[torch.Tensor, str, dict[str, object], dict[str, float]]:
    if args.depth_source == "proxy":
        start = time.perf_counter()
        depth = make_depth_proxy_from_rgb(rgb)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return depth, "luma_proxy_from_sample_rgb", {"provider": "proxy"}, {"depth_total_ms": elapsed_ms}

    if rgb.device.type != "cuda":
        raise RuntimeError("production depth source requires CUDA; pass --device cuda or use --depth-source proxy")

    onnx_path = Path(args.depth_onnx)
    engine_path = Path(args.depth_engine)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"production ONNX not found: {onnx_path}")
    if not engine_path.is_file():
        raise FileNotFoundError(f"production TensorRT engine not found: {engine_path}")

    provider = create_depth_provider(
        DepthProviderConfig(
            backend="tensorrt_native",
            device=rgb.device,
            cache_dir=engine_path.parent,
            onnx_path=onnx_path,
            engine_path=engine_path,
            local_files_only=True,
            prefer_native_tensorrt=True,
            prefer_tensorrt=True,
            prefer_onnx=False,
            build_engine=False,
            force_rebuild=False,
            profile_sync=bool(args.profile_sync),
            depth_upsample=args.depth_upsample,
            depth_upsample_edge_strength=float(args.depth_upsample_edge_strength),
        )
    )
    profile = provider.predict_profile(rgb)
    info = provider.info.to_report() if hasattr(provider.info, "to_report") else dict(provider.info)
    info["engine_path"] = str(engine_path)
    return profile.depth, "production_tensorrt_native", info, profile.to_report()


def tensor_mae(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() - b.float()).abs().mean(dim=1, keepdim=True)


def save_contact_sheet(items: list[tuple[str, Path]], out: Path, columns: int = 3) -> None:
    thumbs = []
    label_h = 28
    thumb_w = 420
    thumb_h = 236
    for label, path in items:
        img = Image.open(path).convert("RGB")
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_w, thumb_h + label_h), (24, 24, 24))
        x = (thumb_w - img.width) // 2
        canvas.paste(img, (x, label_h))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 7), label[:72], fill=(235, 235, 235))
        thumbs.append(canvas)
    if not thumbs:
        return
    rows = (len(thumbs) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + label_h)), (12, 12, 12))
    for idx, img in enumerate(thumbs):
        sheet.paste(img, ((idx % columns) * thumb_w, (idx // columns) * (thumb_h + label_h)))
    sheet.save(out)


def run(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    source_rgb = resize_max_width(load_rgb(args.rgb, device=device), args.max_width)
    rgb_frames = make_sequence(source_rgb, args.sequence_frames, args.sequence_shift_px)
    depth_frames: list[torch.Tensor] = []
    depth_source = ""
    provider_info: dict[str, object] = {}
    depth_timing_rows: list[dict[str, float]] = []
    with torch.inference_mode():
        for frame in rgb_frames:
            depth, depth_source, provider_info, depth_timing = load_depth_for_sweep(frame, args)
            depth_frames.append(depth)
            depth_timing_rows.append(depth_timing)
    rgb = rgb_frames[-1]
    depth = depth_frames[-1]
    edges = edge_mask(depth, threshold=args.edge_mask_threshold)
    runtime_ipd_m = display_ipd_mm_to_runtime_m(args.ipd_mm)

    save_rgb(source_rgb.detach().cpu(), out_dir / "source_rgb.png")
    save_rgb(rgb.detach().cpu(), out_dir / "sequence_last_rgb.png")
    save_rgb(depth.expand(-1, 3, -1, -1).detach().cpu(), out_dir / "production_depth.png")
    save_depth(edges.float().detach().cpu(), out_dir / "depth_edge_mask.png")

    scales = [float(x) for x in args.stereo_scales.split(",") if x.strip()]
    ratios = [float(x) for x in args.max_shift_ratios.split(",") if x.strip()]
    depth_keys = [x.strip().lower() for x in args.depth_options.split(",") if x.strip()]

    rows: list[dict[str, object]] = []
    preview_items: list[tuple[float, str, Path]] = []
    with torch.inference_mode():
        for depth_key in depth_keys:
            depth_strength = DEPTH_OPTIONS[depth_key]
            for stereo_scale in scales:
                for max_shift_ratio in ratios:
                    config = StereoConfig(
                        backend=args.stereo_backend,
                        layers=args.layers,
                        output_format="half_sbs",
                        debug_output=True,
                        temporal=bool(args.temporal),
                        fused=True,
                        depth_strength=depth_strength,
                        convergence=args.convergence,
                        ipd=runtime_ipd_m,
                        ipd_mm=runtime_ipd_m * 1000.0,
                        stereo_scale=stereo_scale,
                        max_shift_ratio=max_shift_ratio,
                        temporal_strength=args.temporal_strength,
                        auto_reset_temporal=bool(args.auto_reset_temporal),
                        scene_reset_threshold=args.scene_reset_threshold,
                        foreground_scale=args.foreground_scale,
                        depth_antialias_strength=args.depth_antialias_strength,
                        edge_dilation=args.edge_dilation,
                        mask_feather_radius=args.mask_feather_radius,
                        hole_fill_mode=args.hole_fill_mode,
                        hole_fill_radius=args.hole_fill_radius,
                        hole_fill_strength=args.hole_fill_strength,
                        edge_threshold=args.edge_threshold,
                    )
                    temporal_state = TemporalState() if config.temporal else None
                    result = None
                    for frame_rgb, frame_depth in zip(rgb_frames, depth_frames, strict=True):
                        result = synthesize_stereo(frame_rgb, frame_depth, config, temporal_state=temporal_state)
                    if result is None:
                        raise RuntimeError("no sequence frames were synthesized")
                    shift = result.debug_info.get("shift_px")
                    occ = result.debug_info.get("occlusion_mask")
                    if not isinstance(shift, torch.Tensor):
                        raise RuntimeError("missing shift_px in debug output")
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

                    slug = f"{depth_key}_scale_{stereo_scale:.2f}_shift_{max_shift_ratio:.2f}".replace(".", "p")
                    sbs_path = out_dir / f"{slug}_half_sbs.png"
                    mask_path = out_dir / f"{slug}_occlusion_mask.png"
                    save_rgb(result.sbs.detach().cpu(), sbs_path)
                    save_depth(occ.detach().cpu(), mask_path)

                    row = {
                        "depth_option": depth_key,
                        "depth_strength": depth_strength,
                        "display_ipd_mm": float(args.ipd_mm),
                        "runtime_ipd_mm": runtime_ipd_m * 1000.0,
                        "stereo_scale": stereo_scale,
                        "max_shift_ratio": max_shift_ratio,
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
                        "sbs_path": str(sbs_path),
                        "occlusion_mask_path": str(mask_path),
                        "hole_fill_backend": str(result.debug_info.get("hole_fill_backend", "none")),
                        "warp_composite_backend": str(result.debug_info.get("warp_composite_backend", "n/a")),
                        "temporal_enabled": bool(config.temporal),
                        "temporal_strength": float(config.temporal_strength),
                        "auto_reset_temporal": bool(config.auto_reset_temporal),
                        "scene_reset_threshold": float(config.scene_reset_threshold),
                        "temporal_reset": int(result.debug_info.get("temporal_reset", 0)),
                        "scene_delta": float(result.debug_info.get("scene_delta", 0.0)),
                        "temporal_reset_count": int(result.debug_info.get("temporal_reset_count", 0)),
                    }
                    rows.append(row)
                    preview_items.append((realism_score, f"{depth_key} scale={stereo_scale:.2f} shift={max_shift_ratio:.2f}", sbs_path))

    ranked_low_artifact = sorted(rows, key=lambda r: (float(r["ghost_risk_score"]) + float(r["hole_risk_score"]), -float(r["stereo_score"])))
    ranked_realism = sorted(rows, key=lambda r: float(r["realism_score"]), reverse=True)
    by_depth = {}
    for depth_key in depth_keys:
        candidates = [r for r in rows if r["depth_option"] == depth_key]
        by_depth[depth_key] = {
            "best_low_artifact": min(candidates, key=lambda r: float(r["ghost_risk_score"]) + float(r["hole_risk_score"])),
            "best_realism": max(candidates, key=lambda r: float(r["realism_score"])),
        }

    top_items = [(label, path) for _, label, path in sorted(preview_items, reverse=True)[:9]]
    save_contact_sheet(top_items, out_dir / "top_realism_contact_sheet.png", columns=3)
    low_artifact_items = [
        (f"{r['depth_option']} scale={r['stereo_scale']:.2f} shift={r['max_shift_ratio']:.2f}", Path(str(r["sbs_path"])))
        for r in ranked_low_artifact[:9]
    ]
    save_contact_sheet(low_artifact_items, out_dir / "low_artifact_contact_sheet.png", columns=3)

    report = {
        "source": str(args.rgb),
        "frame_size": {"width": int(rgb.shape[-1]), "height": int(rgb.shape[-2])},
        "depth_source": depth_source,
        "depth_provider": provider_info,
        "depth_timing": depth_timing_rows[-1] if depth_timing_rows else {},
        "depth_timing_sequence": depth_timing_rows,
        "sequence": {
            "frames": len(rgb_frames),
            "shift_px_per_frame": int(args.sequence_shift_px),
            "mode": "translated_real_sample" if len(rgb_frames) > 1 else "single_real_sample",
        },
        "cinema_base": {
            "backend": args.stereo_backend,
            "layers": int(args.layers),
            "convergence": float(args.convergence),
            "foreground_scale": float(args.foreground_scale),
            "depth_antialias_strength": float(args.depth_antialias_strength),
            "edge_dilation": int(args.edge_dilation),
            "mask_feather_radius": int(args.mask_feather_radius),
            "hole_fill_mode": args.hole_fill_mode,
            "hole_fill_radius": int(args.hole_fill_radius),
            "hole_fill_strength": float(args.hole_fill_strength),
            "edge_threshold": float(args.edge_threshold),
            "temporal": bool(args.temporal),
            "temporal_strength": float(args.temporal_strength),
            "auto_reset_temporal": bool(args.auto_reset_temporal),
            "scene_reset_threshold": float(args.scene_reset_threshold),
        },
        "ipd_mapping": {
            "display_ipd_mm": float(args.ipd_mm),
            "runtime_ipd_mm": runtime_ipd_m * 1000.0,
            "formula": "runtime_ipd_mm = display_ipd_mm * 30 / 60",
        },
        "ranked_low_artifact": ranked_low_artifact[:10],
        "ranked_realism": ranked_realism[:10],
        "by_depth_option": by_depth,
        "rows": rows,
    }
    (out_dir / "cinema_ipd64_quality_sweep_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep cinema-mode stereo params with display IPD fixed at 64mm.")
    parser.add_argument("--rgb", type=Path, default=ROOT / "samples" / "4K.jpg")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "visual_regression" / "cinema_ipd64_quality_sweep")
    parser.add_argument("--ipd-mm", type=float, default=64.0)
    parser.add_argument("--stereo-scales", default="0.20,0.30,0.40,0.50")
    parser.add_argument("--max-shift-ratios", default="0.02,0.03,0.04,0.05")
    parser.add_argument("--depth-options", default="soft,standard,enhanced")
    parser.add_argument("--max-width", type=int, default=1280)
    parser.add_argument("--edge-mask-threshold", type=float, default=0.08)
    parser.add_argument("--device", default=None)
    parser.add_argument("--stereo-backend", choices=["quality_4k", "hq_4k", "fast_plus", "fast"], default="quality_4k")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--convergence", type=float, default=0.25)
    parser.add_argument("--foreground-scale", type=float, default=0.5)
    parser.add_argument("--depth-antialias-strength", type=float, default=1.0)
    parser.add_argument("--edge-dilation", type=int, default=2)
    parser.add_argument("--mask-feather-radius", type=int, default=3)
    parser.add_argument("--hole-fill-mode", default="soft_low_ghost")
    parser.add_argument("--hole-fill-radius", type=int, default=3)
    parser.add_argument("--hole-fill-strength", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--temporal-strength", type=float, default=0.85)
    parser.add_argument("--auto-reset-temporal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-reset-threshold", type=float, default=0.22)
    parser.add_argument("--sequence-frames", type=int, default=3)
    parser.add_argument("--sequence-shift-px", type=int, default=2)
    parser.add_argument("--depth-source", choices=["production", "proxy"], default="production")
    parser.add_argument("--depth-onnx", type=Path, default=DEFAULT_PRODUCTION_ONNX)
    parser.add_argument("--depth-engine", type=Path, default=DEFAULT_PRODUCTION_ENGINE)
    parser.add_argument("--depth-upsample", choices=["bilinear", "guided"], default="guided")
    parser.add_argument("--depth-upsample-edge-strength", type=float, default=0.35)
    parser.add_argument("--profile-sync", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
