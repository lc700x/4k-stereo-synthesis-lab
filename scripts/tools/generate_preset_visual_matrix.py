from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VISUAL_SCRIPT = ROOT / "scripts" / "tools" / "generate_visual_regression_set.py"

PRESETS = ("cinema", "game_low_latency", "still_image_hq")
IMAGE_CATEGORIES = (
    "cinema",
    "game",
    "image_natural",
    "image_unsafe_ui",
    "image_thumbnail_grid",
    "image_low_texture",
)


@dataclass(frozen=True)
class VisualMatrixSample:
    sample_id: str
    path: Path
    category: str
    expected_preset: str
    checks: tuple[str, ...]
    depth: Path | None = None
    auto_depth: bool = True
    depth_backend: str | None = None
    depth_policy: str | None = None
    notes: str = ""
    presets: tuple[str, ...] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a manifest-driven visual regression matrix across product presets.")
    parser.add_argument("--manifest", required=True, help="JSON manifest describing test images.")
    parser.add_argument("--out-dir", default="outputs/visual_preset_matrix", help="Output directory for the matrix run.")
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS), help="Preset list to test for each sample.")
    parser.add_argument(
        "--depth-backend",
        choices=["tensorrt_native", "onnx_cuda_dlpack", "onnx_cuda_iobinding", "pytorch_cuda", "luma"],
        default="tensorrt_native",
        help="Default auto-depth backend when a sample does not override it.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--trt-engine", default=None)
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch per-sample regression jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Only validate manifest and write the planned command matrix.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue remaining jobs if one preset/sample fails.")
    args = parser.parse_args()

    samples = load_manifest(args.manifest)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for sample in samples:
        sample_presets = sample.presets or tuple(args.presets)
        for preset in sample_presets:
            if preset not in PRESETS:
                raise SystemExit(f"sample {sample.sample_id!r} uses unsupported preset {preset!r}")
            job_dir = out_dir / preset / sample.sample_id
            command = build_command(
                python_exe=args.python,
                sample=sample,
                preset=preset,
                out_dir=job_dir,
                default_depth_backend=args.depth_backend,
                device=args.device,
                onnx=args.onnx,
                trt_engine=args.trt_engine,
            )
            jobs.append(
                {
                    "sample_id": sample.sample_id,
                    "category": sample.category,
                    "expected_preset": sample.expected_preset,
                    "preset": preset,
                    "depth_policy": sample.depth_policy,
                    "checks": list(sample.checks),
                    "out_dir": str(job_dir),
                    "command": command,
                    "status": "planned",
                }
            )

    summary: dict[str, Any] = {
        "manifest": str(Path(args.manifest)),
        "run_id": run_id,
        "dry_run": bool(args.dry_run),
        "default_depth_backend": args.depth_backend,
        "samples": [sample_to_report(sample) for sample in samples],
        "jobs": jobs,
    }

    if not args.dry_run:
        for job in jobs:
            print(f"[matrix] {job['sample_id']} / {job['preset']}", flush=True)
            started = time.perf_counter()
            result = subprocess.run(job["command"], cwd=ROOT)
            job["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
            job["returncode"] = result.returncode
            job["status"] = "passed" if result.returncode == 0 else "failed"
            if result.returncode != 0 and not args.continue_on_error:
                write_summary(out_dir, summary)
                raise SystemExit(result.returncode)
    else:
        for job in jobs:
            job["status"] = "dry_run"

    write_summary(out_dir, summary)
    print(f"[matrix] wrote summary: {out_dir}", flush=True)


def load_manifest(path: str | Path) -> list[VisualMatrixSample]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_samples = data.get("samples", data) if isinstance(data, dict) else data
    if not isinstance(raw_samples, list):
        raise SystemExit("manifest must be a JSON list or an object with a samples list")

    samples = [_parse_sample(item, manifest_path.parent) for item in raw_samples]
    _validate_sample_set(samples)
    return samples


def build_command(
    *,
    python_exe: str,
    sample: VisualMatrixSample,
    preset: str,
    out_dir: Path,
    default_depth_backend: str,
    device: str | None,
    onnx: str | None,
    trt_engine: str | None,
) -> list[str]:
    command = [
        python_exe,
        "-B",
        str(VISUAL_SCRIPT),
        "--rgb",
        str(sample.path),
        "--preset",
        preset,
        "--out-dir",
        str(out_dir),
    ]
    if sample.depth:
        command.extend(["--depth", str(sample.depth)])
    elif sample.auto_depth:
        command.extend(["--auto-depth", "--depth-backend", sample.depth_backend or default_depth_backend])
    else:
        raise SystemExit(f"sample {sample.sample_id!r} must provide depth or enable auto_depth")

    if device:
        command.extend(["--device", device])
    if onnx:
        command.extend(["--onnx", onnx])
    if trt_engine:
        command.extend(["--trt-engine", trt_engine])
    return command


def sample_to_report(sample: VisualMatrixSample) -> dict[str, Any]:
    return {
        "id": sample.sample_id,
        "path": str(sample.path),
        "category": sample.category,
        "expected_preset": sample.expected_preset,
        "checks": list(sample.checks),
        "depth": str(sample.depth) if sample.depth else None,
        "auto_depth": sample.auto_depth,
        "depth_backend": sample.depth_backend,
        "depth_policy": sample.depth_policy,
        "notes": sample.notes,
        "presets": list(sample.presets) if sample.presets else None,
    }


def write_summary(out_dir: Path, summary: dict[str, Any]) -> None:
    summary_json = out_dir / "summary.json"
    summary_md = out_dir / "summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Preset Visual Matrix Summary",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Run ID: `{summary['run_id']}`",
        f"- Dry run: `{summary['dry_run']}`",
        f"- Default depth backend: `{summary['default_depth_backend']}`",
        "",
        "## Jobs",
        "",
        "| Sample | Category | Preset | Expected | Status | Checks |",
        "|---|---|---|---|---|---|",
    ]
    for job in summary["jobs"]:
        checks = ", ".join(job["checks"])
        lines.append(
            f"| `{job['sample_id']}` | `{job['category']}` | `{job['preset']}` | "
            f"`{job['expected_preset']}` | `{job['status']}` | {checks} |"
        )
    lines.append("")
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def _parse_sample(item: Any, base_dir: Path) -> VisualMatrixSample:
    if not isinstance(item, dict):
        raise SystemExit("each manifest sample must be an object")
    sample_id = str(item.get("id", "")).strip()
    if not sample_id:
        raise SystemExit("each manifest sample requires a non-empty id")

    path = _resolve_path(item.get("path"), base_dir, field="path", sample_id=sample_id)
    depth_value = item.get("depth")
    depth = _resolve_path(depth_value, base_dir, field="depth", sample_id=sample_id) if depth_value else None
    category = str(item.get("category", "")).strip()
    expected_preset = str(item.get("expected_preset", "")).strip()
    checks = tuple(str(check).strip() for check in item.get("checks", []) if str(check).strip())
    presets_value = item.get("presets")
    presets = tuple(str(preset).strip() for preset in presets_value) if presets_value else None

    return VisualMatrixSample(
        sample_id=sample_id,
        path=path,
        category=category,
        expected_preset=expected_preset,
        checks=checks,
        depth=depth,
        auto_depth=bool(item.get("auto_depth", depth is None)),
        depth_backend=item.get("depth_backend"),
        depth_policy=item.get("depth_policy"),
        notes=str(item.get("notes", "")),
        presets=presets,
    )


def _resolve_path(value: Any, base_dir: Path, *, field: str, sample_id: str) -> Path:
    if not value:
        raise SystemExit(f"sample {sample_id!r} requires {field}")
    path = Path(str(value))
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        raise SystemExit(f"sample {sample_id!r} {field} does not exist: {path}")
    return path


def _validate_sample_set(samples: list[VisualMatrixSample]) -> None:
    seen: set[str] = set()
    categories: set[str] = set()
    for sample in samples:
        if sample.sample_id in seen:
            raise SystemExit(f"duplicate sample id: {sample.sample_id}")
        seen.add(sample.sample_id)
        if sample.category not in IMAGE_CATEGORIES:
            raise SystemExit(f"sample {sample.sample_id!r} has unsupported category {sample.category!r}")
        if sample.expected_preset not in PRESETS:
            raise SystemExit(f"sample {sample.sample_id!r} has unsupported expected_preset {sample.expected_preset!r}")
        if not sample.checks:
            raise SystemExit(f"sample {sample.sample_id!r} requires at least one visual check")
        categories.add(sample.category)

    required = {"cinema", "game", "image_natural", "image_unsafe_ui"}
    missing = sorted(required - categories)
    if missing:
        raise SystemExit(f"manifest is missing required sample categories: {missing}")


if __name__ == "__main__":
    main()
