import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke" / "d2s_depth_runtime_smoke.py"


def test_d2s_depth_runtime_smoke_queue_contract():
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--device",
            "cpu",
            "--width",
            "64",
            "--height",
            "40",
            "--target-height",
            "32",
            "--out",
            "-",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(proc.stdout)

    assert report["mode"] == "d2s_depth_runtime"
    assert report["raw_shape"] == [40, 64, 4]
    assert report["frame_rgb_shape"] == [32, 50, 3]
    assert report["runtime_rgb_shape"] == [3, 32, 50]
    assert report["runtime_rgb_dtype"] == "torch.float32"
    assert 0.0 <= report["runtime_rgb_min"] <= report["runtime_rgb_max"] <= 1.0
    assert report["depth_shape"] == [1, 1, 32, 50]
    assert report["depth_dtype"] == "torch.float32"
    assert report["depth_min"] == 0.0
    assert report["depth_max"] == 1.0
    assert report["capture_timestamp_type"] == "float"
    assert report["queue_contract"] == "(frame_rgb, depth, capture_start_time)"
    assert report["provider_load_count"] == 1
    assert report["provider_predict_count"] == 1
