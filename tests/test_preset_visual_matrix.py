import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tools" / "generate_preset_visual_matrix.py"


def test_preset_visual_matrix_dry_run_writes_summary(tmp_path):
    manifest = _write_manifest(tmp_path)
    out_dir = tmp_path / "matrix"

    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--manifest",
            str(manifest),
            "--out-dir",
            str(out_dir),
            "--depth-backend",
            "luma",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    runs = list(out_dir.iterdir())
    assert len(runs) == 1

    summary = json.loads((runs[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert len(summary["samples"]) == 4
    assert len(summary["jobs"]) == 12
    assert {job["preset"] for job in summary["jobs"]} == {"cinema", "game_low_latency", "still_image_hq"}
    assert all(job["status"] == "dry_run" for job in summary["jobs"])
    assert all("--auto-depth" in job["command"] for job in summary["jobs"])
    assert (runs[0] / "summary.md").exists()


def test_preset_visual_matrix_requires_core_sample_categories(tmp_path):
    image_path = tmp_path / "cinema.png"
    Image.new("RGB", (8, 8), (32, 32, 32)).save(image_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "cinema_dark",
                        "path": image_path.name,
                        "category": "cinema",
                        "expected_preset": "cinema",
                        "checks": ["dark_scene"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "--manifest", str(manifest), "--out-dir", str(tmp_path / "out"), "--dry-run"],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "missing required sample categories" in result.stderr


def _write_manifest(tmp_path: Path) -> Path:
    samples = [
        ("cinema_face", "cinema", "cinema", (80, 32, 32), ["face_edges", "subtitle_safe"]),
        ("game_hud", "game", "game_low_latency", (32, 80, 32), ["hud_edges", "low_latency"]),
        ("image_portrait", "image_natural", "still_image_hq", (32, 32, 80), ["subject_edges", "depth_not_flat"]),
        ("image_ui_grid", "image_unsafe_ui", "still_image_hq", (80, 80, 80), ["force_flat_depth", "text_stable"]),
    ]
    manifest_samples = []
    for sample_id, category, expected_preset, color, checks in samples:
        image_path = tmp_path / f"{sample_id}.png"
        Image.new("RGB", (8, 8), color).save(image_path)
        manifest_samples.append(
            {
                "id": sample_id,
                "path": image_path.name,
                "category": category,
                "expected_preset": expected_preset,
                "checks": checks,
            }
        )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": manifest_samples}, indent=2), encoding="utf-8")
    return manifest
