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
