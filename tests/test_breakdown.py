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
    now = breakdown.last_log + 1.0
    breakdown.inc("openxr_loop", 3)
    breakdown.inc("openxr_should_render", 2)
    breakdown.inc("openxr_no_render", 1)
    breakdown.inc("openxr_no_fresh", 1)
    breakdown.inc("openxr_no_renderable", 1)
    breakdown.inc("runtime_overwrite", 1)
    breakdown.inc("openxr_new_screen_frame", 2)
    breakdown.inc("openxr_reused_screen_frame", 1)
    breakdown.inc("openxr_screen_quality_failed", 1)
    breakdown.inc("openxr_quad_layer_failed", 1)
    breakdown.inc("openxr_projection_render_failed", 1)
    breakdown.inc("openxr_overlay_render_failed", 2)
    breakdown.inc("openxr_controller_render_failed", 3)
    breakdown.inc("openxr_laser_render_failed", 4)
    breakdown.inc("openxr_quad_unavailable_not_runtime_direct", 2)
    breakdown.inc("openxr_quad_unavailable_missing_swapchain", 1)
    breakdown.inc("openxr_layer_count", 5)
    breakdown.inc("openxr_d3d11_pbo_readback", 1)
    breakdown.inc("openxr_background_panorama", 2)
    breakdown.inc("openxr_background_env_model", 1)
    breakdown.inc("openxr_background_env_model_failed", 4)
    breakdown.inc("openxr_background_idle", 3)
    breakdown.inc("openxr_effect_source_ready_publish", 3)
    breakdown.inc("openxr_effect_source_safe_publish", 2)
    breakdown.inc("openxr_effect_source_promote_reuse", 1)
    breakdown.inc("openxr_screen_effect_source_reuse", 4)
    breakdown.inc("openxr_effect_source_reused_safe", 6)
    breakdown.inc("openxr_effect_submit_overwrite", 1)
    breakdown.inc("openxr_effect_submit_budget_skip", 1)
    breakdown.inc("openxr_effect_submit_failed", 1)
    breakdown.inc("openxr_glow_downsample_render", 2)
    breakdown.inc("openxr_glow_downsample_reuse", 5)
    breakdown.inc("openxr_effect_downsample_prewarm", 3)
    breakdown.inc("openxr_glow_downsample_failed", 1)
    breakdown.inc("openxr_effect_downsample_prewarm_failed", 2)
    breakdown.inc("openxr_effect_downsample_prewarm_skip", 1)
    breakdown.inc("openxr_screen_light_downsample_source", 4)
    breakdown.inc("openxr_screen_light_source_reuse", 3)
    breakdown.inc("openxr_screen_light_bind_failed", 1)
    breakdown.inc("openxr_wall_light_mask_loaded", 1)
    breakdown.inc("openxr_wall_light_mask_missing", 2)
    breakdown.inc("openxr_wall_light_mask_disabled", 3)
    breakdown.inc("openxr_wall_light_mask_failed", 4)
    breakdown.inc("openxr_input_trigger_failed", 1)
    breakdown.add_value("openxr_screen_frame_age_frames", 2)
    breakdown.add_value("openxr_effect_ready_age_frames", 4)
    breakdown.add_time("openxr_source_latency", 0.015)
    breakdown.add_time("openxr_effect_submit", 0.007)
    breakdown.add_time("runtime_eye_total", 0.050)
    breakdown.add_time("runtime_eye_d3d11", 0.009)
    breakdown.add_time("runtime_eye_mipmap", 0.040)
    breakdown.add_time("openxr_wait_frame", 0.020)
    breakdown.add_time("openxr_swapchain_wait", 0.013)
    breakdown.add_time("openxr_predicted_period", 0.022)
    breakdown.add_time("openxr_submit_frame", 0.011)
    breakdown.add_time("openxr_render_eyes", 0.004)
    breakdown.add_time("openxr_d3d11_upload", 0.012)
    breakdown.add_time("openxr_quad_update", 0.003)
    breakdown.add_time("openxr_background", 0.005)
    breakdown.add_time("openxr_end_frame", 0.006)
    breakdown.add_time("rt_gpu_total", 0.030)
    breakdown.add_time("rt_gpu_depth", 0.010)
    breakdown.add_time("rt_gpu_depth_preprocess", 0.001)
    breakdown.add_time("rt_gpu_depth_model", 0.002)
    breakdown.add_time("rt_gpu_depth_normalize", 0.003)
    breakdown.add_time("rt_gpu_depth_upsample", 0.004)
    breakdown.add_time("rt_gpu_depth_postprocess", 0.005)
    breakdown.add_time("rt_gpu_synth", 0.015)
    breakdown.add_time("rt_gpu_synth_scene", 0.001)
    breakdown.add_time("rt_gpu_synth_depth_shift", 0.002)
    breakdown.add_time("rt_gpu_synth_warp", 0.003)
    breakdown.add_time("rt_gpu_synth_occ", 0.004)
    breakdown.add_time("rt_gpu_synth_fill", 0.005)
    breakdown.add_time("rt_gpu_synth_refine", 0.006)
    breakdown.add_time("rt_gpu_synth_temporal", 0.007)
    breakdown.add_time("rt_gpu_synth_output_depth", 0.008)
    breakdown.add_time("rt_gpu_synth_sbs", 0.009)
    breakdown.add_time("rt_gpu_pack", 0.003)
    breakdown.add_time("rt_gpu_openxr_pack", 0.002)

    breakdown.log(now=now)

    output = capsys.readouterr().out
    assert "xr_loop=" in output
    assert "xr_should=" in output
    assert "xr_no_render=" in output
    assert "xr_no_fresh=" in output
    assert "xr_no_renderable=" in output
    assert "rt_overwrite=" in output
    assert "screen_new=2.0" in output
    assert "screen_reuse=1.0" in output
    assert "screen_age=2.00f" in output
    assert "screen_quality_failed=1.0" in output
    assert "source_lat=15.00ms" in output
    assert "fx_submit=7.00ms" in output
    assert "fx_age=4.00f" in output
    assert "fx_ready=3.0" in output
    assert "fx_safe=2.0" in output
    assert "fx_promote_reuse=1.0" in output
    assert "fx_source_reuse=4.0" in output
    assert "fx_safe_reuse=6.0" in output
    assert "fx_overwrite=1.0" in output
    assert "fx_budget_skip=1.0" in output
    assert "fx_submit_failed=1.0" in output
    assert "fx_ds_render=2.0" in output
    assert "fx_ds_reuse=5.0" in output
    assert "fx_ds_prewarm=3.0" in output
    assert "fx_ds_failed=1.0,prewarm:2.0,prewarm_skip:1.0" in output
    assert "light_ds=4.0" in output
    assert "light_reuse=3.0" in output
    assert "light_bind_failed=1.0" in output
    assert "wall_mask=loaded:1.0,missing:2.0,disabled:3.0,failed:4.0" in output
    assert "eye_total=50.00ms" in output
    assert "eye_sync=" not in output
    assert "eye_d3d11=9.00ms" in output
    assert "eye_mipmap=40.00ms" in output
    assert "xr_wait=20.00ms" in output
    assert "swapchain_wait=13.00ms" in output
    assert "xr_pred=22.00ms" in output
    assert "xr_submit=11.00ms" in output
    assert "xr_render=4.00ms" in output
    assert "projection_failed=1.0" in output
    assert "overlay_failed=2.0" in output
    assert "controller_failed=3.0" in output
    assert "laser_failed=4.0" in output
    assert "input_trigger_failed=1.0" in output
    assert "d3d11_upload=12.00ms" in output
    assert "d3d11_pbo=1.0" in output
    assert "quad_update=3.00ms" in output
    assert "quad_failed=1.0" in output
    assert "quad_unavail=missing_swapchain:1.0,not_runtime_direct:2.0" in output
    assert "background=5.00ms" in output
    assert "bg_path=panorama:2.0,env:1.0,env_failed:4.0,idle:3.0" in output
    assert "layer_count=5.0" in output
    assert "xr_end=6.00ms" in output
    assert "rt_gpu_total=30.00ms" in output
    assert "rt_gpu_depth=10.00ms" in output
    assert "rt_gpu_depth_pre=1.00ms" in output
    assert "rt_gpu_depth_model=2.00ms" in output
    assert "rt_gpu_depth_norm=3.00ms" in output
    assert "rt_gpu_depth_up=4.00ms" in output
    assert "rt_gpu_depth_post=5.00ms" in output
    assert "rt_gpu_synth=15.00ms" in output
    assert "rt_gpu_syn_scene=1.00ms" in output
    assert "rt_gpu_syn_shift=2.00ms" in output
    assert "rt_gpu_syn_warp=3.00ms" in output
    assert "rt_gpu_syn_occ=4.00ms" in output
    assert "rt_gpu_syn_fill=5.00ms" in output
    assert "rt_gpu_syn_refine=6.00ms" in output
    assert "rt_gpu_syn_temporal=7.00ms" in output
    assert "rt_gpu_syn_out_depth=8.00ms" in output
    assert "rt_gpu_syn_sbs=9.00ms" in output
    assert "rt_gpu_pack=3.00ms" in output
    assert "rt_gpu_openxr_pack=2.00ms" in output
