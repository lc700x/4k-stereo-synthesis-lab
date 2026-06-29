from types import SimpleNamespace

from utils.breakdown import FPSBreakdown


def test_fps_breakdown_prefers_structured_runtime_output_fields():
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    result = SimpleNamespace(
        output_format="openxr_eye_views",
        output_dtype="uint8",
        output_pack_backend="none",
        timing={"total_ms": 12.5},
        debug_info={
            "backend": "openxr_roll_adaptive_grid_sample",
            "runtime_depth_backend": "metric3d",
            "runtime_output_format": "legacy_format",
            "runtime_output_dtype": "legacy_dtype",
            "runtime_output_pack_backend": "legacy_pack",
        },
    )

    breakdown.add_runtime_timing(result)

    assert breakdown.stats["rt_output_format"] == "openxr_eye_views"
    assert breakdown.stats["rt_output_dtype"] == "uint8"
    assert breakdown.stats["rt_output_pack"] == "none"


def test_fps_breakdown_supports_legacy_debug_runtime_output_fields():
    breakdown = FPSBreakdown(enabled=True, target_fps=60)
    result = SimpleNamespace(
        timing={},
        debug_info={
            "runtime_output_format": "half_sbs",
            "runtime_output_dtype": "float32",
            "runtime_output_pack_backend": "torch_float_to_uint8",
        },
    )

    breakdown.add_runtime_timing(result)

    assert breakdown.stats["rt_output_format"] == "half_sbs"
    assert breakdown.stats["rt_output_dtype"] == "float32"
    assert breakdown.stats["rt_output_pack"] == "torch_float_to_uint8"


def test_fps_breakdown_logs_openxr_loop_segments(capsys):
    breakdown = FPSBreakdown(enabled=True, target_fps=72)
    breakdown.last_log -= 1.1
    breakdown.inc("openxr_loop", 3)
    breakdown.inc("openxr_should_render", 2)
    breakdown.inc("openxr_no_render", 1)
    breakdown.inc("openxr_no_fresh", 1)
    breakdown.inc("openxr_no_renderable", 1)
    breakdown.inc("runtime_overwrite", 1)
    breakdown.add_time("runtime_eye_total", 0.050)
    breakdown.add_time("runtime_eye_mipmap", 0.040)
    breakdown.add_time("openxr_wait_frame", 0.020)
    breakdown.add_time("openxr_predicted_period", 0.022)
    breakdown.add_time("openxr_submit_frame", 0.011)
    breakdown.add_time("openxr_render_eyes", 0.004)
    breakdown.add_time("openxr_end_frame", 0.006)
    breakdown.add_time("rt_gpu_total", 0.030)
    breakdown.add_time("rt_gpu_depth", 0.010)
    breakdown.add_time("rt_gpu_synth", 0.015)
    breakdown.add_time("rt_gpu_pack", 0.003)
    breakdown.add_time("rt_gpu_openxr_pack", 0.002)

    breakdown.log()

    output = capsys.readouterr().out
    assert "xr_loop=" in output
    assert "xr_should=" in output
    assert "xr_no_render=" in output
    assert "xr_no_fresh=" in output
    assert "xr_no_renderable=" in output
    assert "rt_overwrite=" in output
    assert "eye_total=50.00ms" in output
    assert "eye_mipmap=40.00ms" in output
    assert "xr_wait=20.00ms" in output
    assert "xr_pred=22.00ms" in output
    assert "xr_submit=11.00ms" in output
    assert "xr_render=4.00ms" in output
    assert "xr_end=6.00ms" in output
    assert "rt_gpu_total=30.00ms" in output
    assert "rt_gpu_depth=10.00ms" in output
    assert "rt_gpu_synth=15.00ms" in output
    assert "rt_gpu_pack=3.00ms" in output
    assert "rt_gpu_openxr_pack=2.00ms" in output
