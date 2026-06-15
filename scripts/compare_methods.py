from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    print("[1/6] parsing arguments ...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--out-dir", default="outputs/compare")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-format", choices=["half_sbs", "full_sbs"], default="half_sbs")
    parser.add_argument("--depth-strength", type=float, default=3.0)
    parser.add_argument("--convergence", type=float, default=0.0)
    args = parser.parse_args()

    print("[2/6] importing torch ...", flush=True)
    import torch

    print("[3/6] importing stereo_lab ...", flush=True)
    from stereo_lab.io import load_depth, load_rgb, save_depth, save_rgb
    from stereo_lab.report import absdiff, basic_image_metrics, make_contact_sheet, write_json
    from stereo_lab.synthesis import StereoConfig, synthesize_stereo

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    print(f"[info] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)

    print("[4/6] loading inputs ...", flush=True)
    rgb = load_rgb(args.rgb, device=device)
    depth = load_depth(args.depth, device=device)
    out_dir = Path(args.out_dir)

    configs = [
        StereoConfig(backend="fast", output_format=args.output_format, depth_strength=args.depth_strength, convergence=args.convergence),
        StereoConfig(backend="quality_4k", layers=2, output_format=args.output_format, depth_strength=args.depth_strength, convergence=args.convergence, debug_output=True),
        StereoConfig(backend="hq_4k", layers=3, output_format=args.output_format, depth_strength=args.depth_strength, convergence=args.convergence, debug_output=True),
    ]
    results = {}
    report = {
        "rgb": str(args.rgb),
        "depth": str(args.depth),
        "output_format": args.output_format,
        "depth_strength": args.depth_strength,
        "convergence": args.convergence,
        "device": str(device),
        "outputs": {},
        "comparisons": {},
    }

    print("[5/6] synthesizing methods ...", flush=True)
    with torch.inference_mode():
        for config in configs:
            result = synthesize_stereo(rgb, depth, config)
            key = f"{config.backend}_{args.output_format}"
            results[key] = result
            save_rgb(result.left_eye.cpu(), out_dir / f"{key}_left.png")
            save_rgb(result.right_eye.cpu(), out_dir / f"{key}_right.png")
            save_rgb(result.sbs.cpu(), out_dir / f"{key}.png")
            mask = result.debug_info.get("occlusion_mask")
            if mask is not None:
                save_depth(mask.cpu(), out_dir / f"{key}_occlusion_mask.png")
            report["outputs"][key] = {
                "left_shape": list(result.left_eye.shape),
                "right_shape": list(result.right_eye.shape),
                "sbs_shape": list(result.sbs.shape),
            }
            print(f"  wrote {key}", flush=True)

        fast_key = f"fast_{args.output_format}"
        sheet_items = [rgb.cpu(), depth.repeat(1, 3, 1, 1).cpu()]
        for key, result in results.items():
            sheet_items.append(result.sbs.cpu())
            if key != fast_key:
                diff = absdiff(results[fast_key].sbs.cpu(), result.sbs.cpu())
                save_rgb(diff, out_dir / f"{fast_key}_vs_{key}_absdiff.png")
                report["comparisons"][f"{fast_key}_vs_{key}"] = basic_image_metrics(results[fast_key].sbs.cpu(), result.sbs.cpu())
                sheet_items.append(diff)

        contact = make_contact_sheet(sheet_items, columns=2)
        save_rgb(contact, out_dir / "contact_sheet.png")
        write_json(report, out_dir / "report.json")

    print(f"[6/6] comparison written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
