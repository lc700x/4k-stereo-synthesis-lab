from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_lab.io import load_depth, load_rgb, save_rgb
from stereo_lab.synthesis import StereoConfig, synthesize_stereo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--out-dir", default="outputs/compare")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-format", choices=["half_sbs", "full_sbs"], default="half_sbs")
    args = parser.parse_args()

    device = torch.device(args.device)
    rgb = load_rgb(args.rgb, device=device)
    depth = load_depth(args.depth, device=device)
    out_dir = Path(args.out_dir)

    configs = [
        StereoConfig(backend="fast", output_format=args.output_format),
        StereoConfig(backend="quality_4k", layers=2, output_format=args.output_format),
        StereoConfig(backend="hq_4k", layers=3, output_format=args.output_format),
    ]
    with torch.inference_mode():
        for config in configs:
            result = synthesize_stereo(rgb, depth, config)
            name = f"{config.backend}_{args.output_format}.png"
            save_rgb(result.sbs, out_dir / name)
            print(out_dir / name)


if __name__ == "__main__":
    main()
