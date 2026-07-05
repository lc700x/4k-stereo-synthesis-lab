from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.openxr_visual_regression import (  # noqa: E402
    OpenXRViewerShaderParams,
    run_openxr_visual_regression,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline pixel regression for OpenXR rgb_depth viewer shader.")
    parser.add_argument("--rgb", type=Path, default=None, help="Optional source_rgb.png from D2S_OPENXR_RGB_DEPTH_DUMP_DIR.")
    parser.add_argument("--depth", type=Path, default=None, help="Optional prepared_depth.png from D2S_OPENXR_RGB_DEPTH_DUMP_DIR.")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "visual_regression" / "openxr_rgb_depth")
    parser.add_argument("--max-disparity-px", type=float, default=96.0)
    parser.add_argument("--convergence", type=float, default=0.0)
    parser.add_argument("--screen-roll", type=float, default=0.0)
    parser.add_argument("--shader-resolution-mode", choices=["source", "swapchain"], default="source")
    parser.add_argument("--swapchain-width", type=int, default=3648)
    parser.add_argument("--swapchain-height", type=int, default=3648)
    args = parser.parse_args()

    params = OpenXRViewerShaderParams(
        max_disparity_px=args.max_disparity_px,
        convergence=args.convergence,
        screen_roll=args.screen_roll,
        shader_resolution_mode=args.shader_resolution_mode,
        swapchain_width=args.swapchain_width,
        swapchain_height=args.swapchain_height,
    )
    metrics = run_openxr_visual_regression(
        output_dir=args.out,
        rgb_path=args.rgb,
        depth_path=args.depth,
        params=params,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"[openxr_visual_regression] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
