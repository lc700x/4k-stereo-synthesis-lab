from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.io import load_rgb, save_rgb  # noqa: E402
from stereo_runtime.openxr_visual_regression import make_depth_proxy_from_rgb  # noqa: E402
from stereo_runtime.synthesis import StereoConfig, synthesize_stereo  # noqa: E402


def _resize_max_width(rgb: torch.Tensor, max_width: int) -> torch.Tensor:
    if max_width <= 0:
        return rgb
    _, _, height, width = rgb.shape
    if width <= max_width:
        return rgb
    out_height = max(1, int(round(height * (max_width / width))))
    return F.interpolate(rgb, size=(out_height, max_width), mode="area")


def _edge_mask(depth: torch.Tensor, threshold: float = 0.08) -> torch.Tensor:
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
    edge = (F.conv2d(depth, sobel_x, padding=1).abs() + F.conv2d(depth, sobel_y, padding=1).abs())
    edge = F.max_pool2d(edge, kernel_size=7, stride=1, padding=3)
    return edge > float(threshold)


def _masked_mae(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    diff = (a.float() - b.float()).abs().mean(dim=1, keepdim=True)
    if mask.shape[-2:] != diff.shape[-2:]:
        mask = F.interpolate(mask.float(), size=diff.shape[-2:], mode="nearest") > 0.5
    if not bool(mask.any()):
        return float(diff.mean().item())
    return float(diff[mask.expand_as(diff)].mean().item())


def _mean_mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def run_sweep(
    *,
    rgb_path: Path,
    output_dir: Path,
    backends: list[str],
    scales: list[float],
    mask_feathers: list[int],
    max_width: int,
) -> dict[str, object]:
    rgb = _resize_max_width(load_rgb(rgb_path), int(max_width))
    depth = make_depth_proxy_from_rgb(rgb)
    mask = _edge_mask(depth)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb(rgb, output_dir / "source_rgb.png")
    save_rgb(depth.expand(-1, 3, -1, -1), output_dir / "prepared_depth.png")

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for backend in backends:
            for scale in scales:
                for feather in mask_feathers:
                    config = StereoConfig(
                        backend=backend,  # type: ignore[arg-type]
                        output_format="half_sbs",
                        debug_output=True,
                        temporal=False,
                        fused=False,
                        ipd_mm=64.0,
                        stereo_scale=float(scale),
                        depth_strength=2.0,
                        convergence=0.0,
                        max_shift_ratio=0.05,
                        mask_feather_radius=int(feather),
                    )
                    result = synthesize_stereo(rgb, depth, config)
                    safe_backend = backend.replace("/", "_")
                    safe_scale = f"{float(scale):.2f}".replace(".", "p")
                    save_rgb(result.sbs, output_dir / f"{safe_backend}_scale_{safe_scale}_feather_{int(feather)}_sbs.png")

                    left_edge = _masked_mae(result.left_eye, rgb, mask)
                    right_edge = _masked_mae(result.right_eye, rgb, mask)
                    shift_px = result.debug_info.get("shift_px")
                    max_shift_px = float(shift_px.abs().max().item()) if isinstance(shift_px, torch.Tensor) else 0.0
                    occlusion_mask = result.debug_info.get("occlusion_mask")
                    occlusion_ratio = (
                        float(occlusion_mask.float().mean().item())
                        if isinstance(occlusion_mask, torch.Tensor)
                        else 0.0
                    )
                    rows.append(
                        {
                            "backend": backend,
                            "stereo_scale": float(scale),
                            "mask_feather_radius": int(feather),
                            "max_shift_px": max_shift_px,
                            "mean_left_to_source_mae": _mean_mae(result.left_eye, rgb),
                            "mean_right_to_source_mae": _mean_mae(result.right_eye, rgb),
                            "mean_left_right_mae": _mean_mae(result.left_eye, result.right_eye),
                            "edge_left_to_source_mae": left_edge,
                            "edge_right_to_source_mae": right_edge,
                            "edge_mean_to_source_mae": (left_edge + right_edge) * 0.5,
                            "occlusion_mask_ratio": occlusion_ratio,
                            "warp_composite_backend": str(result.debug_info.get("warp_composite_backend", "n/a")),
                            "hole_fill_backend": str(result.debug_info.get("hole_fill_backend", "n/a")),
                        }
                    )

    ranked = sorted(rows, key=lambda item: (str(item["backend"]), float(item["stereo_scale"]), int(item["mask_feather_radius"])))
    best_by_backend_scale: list[dict[str, object]] = []
    for backend in backends:
        for scale in scales:
            candidates = [
                row for row in rows
                if row["backend"] == backend and abs(float(row["stereo_scale"]) - float(scale)) < 1e-6
            ]
            if candidates:
                best_by_backend_scale.append(min(candidates, key=lambda row: float(row["edge_mean_to_source_mae"])))
    summary = {
        "source": str(rgb_path),
        "frame_size": {"width": int(rgb.shape[-1]), "height": int(rgb.shape[-2])},
        "backends": backends,
        "scales": scales,
        "mask_feathers": mask_feathers,
        "best_by_backend_scale": best_by_backend_scale,
        "rows": ranked,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep Stereo Scale and quantify edge movement in stereo modes.")
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "visual_regression" / "stereo_scale_sweep")
    parser.add_argument("--backends", default="fast,fast_plus,quality_4k,hq_4k")
    parser.add_argument("--scales", default="0.3,0.5,0.7,1.0")
    parser.add_argument("--mask-feathers", default="0,1,2,3")
    parser.add_argument("--max-width", type=int, default=960)
    args = parser.parse_args()

    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    scales = [float(item.strip()) for item in args.scales.split(",") if item.strip()]
    mask_feathers = [int(item.strip()) for item in args.mask_feathers.split(",") if item.strip()]
    summary = run_sweep(
        rgb_path=args.rgb,
        output_dir=args.out,
        backends=backends,
        scales=scales,
        mask_feathers=mask_feathers,
        max_width=args.max_width,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[stereo_scale_sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
