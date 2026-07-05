import logging
import os
import queue
import subprocess
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from xr_viewer.openxr_runtime import (
    OpenXRRuntimeCallbacks,
    OpenXRRuntimeConfig,
    frame_size_from_eye,
    frame_size_from_runtime_result,
    load_openxr_viewer,
    run_openxr_mode,
    use_environment_viewer,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_openxr_runtime_import_does_not_load_xr_implementation(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    code = "from xr_viewer.openxr_runtime import use_environment_viewer; print(use_environment_viewer('none'))"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_use_environment_viewer_requires_real_environment_name():
    assert not use_environment_viewer(None)
    assert not use_environment_viewer("")
    assert not use_environment_viewer(" none ")
    assert use_environment_viewer("Default")
    assert use_environment_viewer("studio")


def test_frame_size_from_eye_uses_height_width_shape_for_array_like():
    eye = SimpleNamespace(shape=(720, 1280, 4))

    assert frame_size_from_eye(eye) == (1280, 720)


def test_frame_size_from_runtime_result_prefers_display_size_metadata():
    result = SimpleNamespace(
        left_eye=SimpleNamespace(shape=(1, 3, 2160, 1920)),
        output_display_size=(3840, 2160),
        debug_info={"runtime_output_display_size": "1920x1080"},
    )

    assert frame_size_from_runtime_result(result) == (3840, 2160)


def test_frame_size_from_runtime_result_supports_legacy_debug_display_size():
    result = SimpleNamespace(
        left_eye=SimpleNamespace(shape=(1, 3, 2160, 1920)),
        debug_info={"runtime_output_display_size": "3840x2160"},
    )

    assert frame_size_from_runtime_result(result) == (3840, 2160)


def test_load_openxr_viewer_uses_environment_split_for_named_environment(monkeypatch):
    fake_base_viewer = object()
    fake_environment_viewer = object()
    monkeypatch.setitem(
        sys.modules,
        "xr_viewer.base",
        types.SimpleNamespace(OPENXR_AVAILABLE=True, OpenXRViewer=fake_base_viewer),
    )
    monkeypatch.setitem(
        sys.modules,
        "xr_viewer.environment",
        types.SimpleNamespace(OPENXR_AVAILABLE=True, OpenXRViewer=fake_environment_viewer),
    )

    assert load_openxr_viewer("Cinema") is fake_environment_viewer
    assert load_openxr_viewer("Default") is fake_environment_viewer
    assert load_openxr_viewer("none") is fake_base_viewer


def test_load_openxr_viewer_raises_when_runtime_unavailable(monkeypatch):
    fake_base = types.SimpleNamespace(OPENXR_AVAILABLE=False, OpenXRViewer=object)
    monkeypatch.setitem(sys.modules, "xr_viewer.base", fake_base)

    with pytest.raises(ImportError, match="pyopenxr not installed"):
        load_openxr_viewer(None)


def test_run_openxr_mode_passes_depth_strength_to_viewer(monkeypatch):
    calls = []

    class FakeViewer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self, **kwargs):
            calls.append({"run": kwargs})

    monkeypatch.setitem(
        sys.modules,
        "xr_viewer.base",
        types.SimpleNamespace(OPENXR_AVAILABLE=True, OpenXRViewer=FakeViewer),
    )
    runtime_q = queue.Queue()
    runtime_q.put(
        (
            SimpleNamespace(
                left_eye=SimpleNamespace(shape=(1, 3, 2160, 1920)),
                output_display_size=(3840, 2160),
                debug_info={"runtime_output_display_size": "1920x1080"},
            ),
            123.0,
        )
    )
    config = OpenXRRuntimeConfig(
        depth_strength=2.4,
        convergence=0.1,
        fps=72,
        show_fps=True,
        controller_model="pico",
        environment_model="none",
        screen_width=7.8,
        screen_distance=9.5,
        show_preview_window=False,
        capture_mode="Monitor",
        monitor_index=1,
    )
    callbacks = OpenXRRuntimeCallbacks(
        update_runtime_config=lambda *args, **kwargs: None,
        render_active_set=lambda: None,
        render_active_clear=lambda: None,
        source_active_set=lambda: None,
        wait_idle_clear=lambda: None,
        bootstrap_done_set=lambda: None,
    )

    viewer = run_openxr_mode(runtime_q, config, callbacks)

    assert isinstance(viewer, FakeViewer)
    assert calls[0]["depth_strength"] == 2.4
    assert "depth_ratio" not in calls[0]
    assert "ipd" not in calls[0]
    assert calls[0]["frame_size"] == (3840, 2160)
    assert calls[0]["openxr_screen_width"] == 7.8
    assert calls[0]["openxr_screen_distance"] == 9.5


def test_runtime_eye_tensor_hwc_u8_scales_near_normalized_float_range(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    frame = torch.full((1, 3, 2, 2), 0.5, dtype=torch.float32)
    frame[..., 0, 0] = 1.0001

    out = CoreRuntimeEyeMixin()._runtime_eye_tensor_hwc_u8(torch, frame)

    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (2, 2, 3)
    assert int(out[1, 1, 0]) == 127
    assert int(out[0, 0, 0]) == 255


def test_runtime_eye_stats_log_prefers_structured_output_fields(monkeypatch, caplog, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    left_eye = torch.zeros(1, 3, 2, 4, dtype=torch.float32)
    right_eye = torch.ones(1, 3, 2, 4, dtype=torch.float32)
    result = SimpleNamespace(
        left_eye=left_eye,
        right_eye=right_eye,
        output_format="openxr_full_synthesis_eyes",
        output_dtype="uint8",
        output_eye_size=(3840, 2160),
        output_pack_backend="none",
        debug_info={
            "runtime_output_format": "legacy_format",
            "runtime_output_dtype": "legacy_dtype",
            "runtime_output_eye_size": "1x1",
            "runtime_output_pack_backend": "legacy_pack",
        },
    )

    with caplog.at_level(logging.DEBUG, logger="xr_viewer.core_runtime_eye"):
        viewer._log_runtime_eye_stats_once(result, upload_path="cpu")

    output = caplog.text
    assert "format=openxr_full_synthesis_eyes" in output
    assert "runtime_dtype=uint8" in output
    assert "eye_size=(3840, 2160)" in output
    assert "pack=none" in output
    assert "legacy_format" not in output
    assert "legacy_dtype" not in output
    assert "eye_size=1x1" not in output
    assert "legacy_pack" not in output
    assert "[OpenXRViewer] runtime eye stats:" not in capsys.readouterr().out


def test_cpu_fallback_paths_emit_red_console_warnings(monkeypatch):
    monkeypatch.chdir(SRC)
    runtime_eye = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    frame_upload = (SRC / "xr_viewer" / "core_frame_upload.py").read_text(encoding="utf-8")
    environment_renderer = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    screen_quality = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    breakdown = (SRC / "utils" / "breakdown.py").read_text(encoding="utf-8")
    d3d11 = (SRC / "xr_viewer" / "d3d11_native_renderer.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    viewer = (SRC / "viewer" / "viewer.py").read_text(encoding="utf-8")
    gl_uploader = (SRC / "viewer" / "gl_texture_uploader.py").read_text(encoding="utf-8")
    metal = (SRC / "viewer" / "metal_viewer.py").read_text(encoding="utf-8")
    output_convert = (SRC / "stereo_runtime" / "output_convert.py").read_text(encoding="utf-8")
    depth_onnx = (SRC / "stereo_runtime" / "depth_onnx_provider.py").read_text(encoding="utf-8")
    tensorrt_ort = (SRC / "stereo_runtime" / "providers" / "nvidia" / "tensorrt_ort.py").read_text(encoding="utf-8")
    legacy_sbs = (SRC / "streaming" / "legacy_sbs.py").read_text(encoding="utf-8")
    warnings = (SRC / "utils" / "cpu_warnings.py").read_text(encoding="utf-8")

    assert "\\033[91m" in warnings
    assert "warn_cpu_fallback" in runtime_eye
    assert "warn_cpu_transfer" in runtime_eye
    assert "runtime_eye_not_cuda" in runtime_eye
    assert "texture_image=fallback" in runtime_eye
    assert "CudaGlTextureUploader" in runtime_eye
    assert "upload_rgba" in runtime_eye
    assert "runtime_eye_tensor" in runtime_eye
    assert "runtime_eye_image" in runtime_eye
    assert "runtime_eye_total" in runtime_eye
    assert "glGenerateMipmap" in gl_uploader
    assert "_upload_pbo" in gl_uploader
    assert "memcpy_2d_to_array" in gl_uploader
    assert "row_bytes, row_bytes" in gl_uploader
    assert "_runtime_eye_tensor_rgba_u8" in runtime_eye
    assert "D2S_OPENXR_RUNTIME_EYE_TEXTURE_GPU_UPLOAD" in implementation
    assert "os.environ.get('D2S_OPENXR_RUNTIME_EYE_TEXTURE_GPU_UPLOAD', '1')" in implementation
    assert "_runtime_eye_texture_components = 4" in implementation
    assert "OpenXR D3D11 projection submit" in implementation
    assert "pbo_glreadpixels" in implementation
    assert "openxr_d3d11_pbo_readback" in implementation
    assert "warn_cpu_fallback" in frame_upload
    assert "OpenXR RGB+depth texture upload" in frame_upload
    assert "OpenXR depth texture upload" in frame_upload
    assert "OpenXR D3D11 RGB+depth texture upload" in d3d11
    assert "using_cpu_update_subresource" in d3d11
    assert "StereoWindow runtime texture upload" in viewer
    assert "StereoWindow RGB+depth texture upload" in viewer
    assert "CudaGlTextureUploader" in viewer
    assert "CUDA/GL image texture upload failed" in gl_uploader
    assert "using PBO fallback" in gl_uploader
    assert "requires {name} support" in gl_uploader
    assert "metal_rgb_cpu_transfer" in metal
    assert "metal_depth_cpu_transfer" in metal
    assert "runtime_output_to_numpy_cpu_transfer" in output_convert
    assert "depth_onnx_input_cpu_transfer" in depth_onnx
    assert "tensorrt_ort_input_cpu_transfer" in tensorrt_ort
    assert "tensorrt_ort_output_numpy_transfer" in tensorrt_ort
    assert "legacy_sbs_output_cpu_transfer" in legacy_sbs


def test_openxr_rgb_depth_shaders_use_consistent_parallax_formula(monkeypatch):
    monkeypatch.chdir(SRC)
    source = (SRC / "xr_viewer" / "d3d11_native_renderer.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    viewer_source = (SRC / "viewer" / "viewer.py").read_text(encoding="utf-8")

    assert "#define roll params.w" in source
    assert "float2 shiftedUv = uv - float2(shift * cos(roll), shift * sin(roll));" in source
    assert "#define parallaxOffset params.x" in source
    assert "float depthResponse = depth - convergence;" in source
    assert "float shift = depthResponse * parallaxOffset * depthStrength;" in source
    assert "depthInv" not in source
    assert "def render_eye(self, swapchain_texture, width, height, eye_index, eye_offset, depth_strength, convergence, mvp, roll=0.0):" in source
    assert "constants[16:20] = np.array([eye_offset, depth_strength, convergence, roll]" in source
    assert "eye_sign * ipd * 0.5" not in source
    assert "self.runtime_eye_srv[eye_index], 0.0, 0.0, 0.0, mvp, roll=0.0" in source
    assert "screen_disparity_uv = max(0.0, runtime_rgb_depth_max_disparity_px) / float(runtime_rgb_depth_render_width)" in implementation
    assert "roll=self.screen_roll" in implementation
    assert "float depth_response = depth - u_convergence;" in viewer_source
    assert "float shift = depth_response;" in viewer_source
    assert "float px = u_eye_offset * shift * u_depth_strength * edge_falloff;" in viewer_source
    assert "float shift_amount = (depth - u_convergence) * u_depth_strength;" in viewer_source
    assert "float shift_amount = (depth - u_convergence);" in viewer_source
    assert "depth_shaped" not in viewer_source
    assert "float depth_inv = -depth;" not in viewer_source
    assert "depth_inv + u_convergence" not in viewer_source


def test_runtime_rgb_depth_config_prefers_structured_shader_uniforms(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    viewer._apply_runtime_rgb_depth_config(
        {
            "openxr_shader_uniforms": {
                "convergence": 9.0,
                "depth_strength": 9.0,
                "max_disparity_px": 9.0,
                "render_size": (9, 9),
            },
            "openxr_convergence": 8.0,
            "resolved_max_disparity_px": 8.0,
            "runtime_output_eye_size": "8x8",
        },
        shader_uniforms={
            "convergence": 0.25,
            "depth_strength": 2.4,
            "max_disparity_px": 18.0,
            "render_size": (1920, 1080),
        },
    )

    assert viewer.convergence == 0.25
    assert viewer._runtime_rgb_depth_depth_strength == 2.4
    assert viewer._runtime_rgb_depth_max_disparity_px == 18.0
    assert viewer._runtime_rgb_depth_render_width == 1920


def test_runtime_rgb_depth_config_keeps_debug_uniform_fallback(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    viewer._apply_runtime_rgb_depth_config(
        {
            "openxr_shader_uniforms": {
                "convergence": 0.25,
                "depth_strength": 1.7,
                "max_disparity_px": 18.0,
                "render_size": (1920, 1080),
            },
            "openxr_convergence": 9.0,
            "resolved_max_disparity_px": 9.0,
            "runtime_output_eye_size": "9x9",
        }
    )

    assert viewer.convergence == 0.25
    assert viewer._runtime_rgb_depth_depth_strength == 1.7
    assert viewer._runtime_rgb_depth_max_disparity_px == 18.0
    assert viewer._runtime_rgb_depth_render_width == 1920


def test_screen_frame_bridge_drains_latest_and_tracks_reuse():
    from xr_viewer.core_source_state import ScreenFrameBridge

    source_q = queue.Queue()
    stale_frame = object()
    latest_result = SimpleNamespace(left_eye=object(), right_eye=object(), depth=object())
    latest_frame = (latest_result, 12.5)
    source_q.put(stale_frame)
    source_q.put(latest_frame)

    bridge = ScreenFrameBridge(source_q)
    poll = bridge.drain_latest()

    assert poll.frame is latest_frame
    assert poll.dequeued == 2
    assert poll.dropped == 1
    assert poll.is_new
    assert poll.frame_id == 1
    assert poll.source_timestamp == 12.5
    assert bridge.latest_frame is latest_frame
    assert bridge.source_timestamp == 12.5
    assert bridge.reuse_presented().frame is None

    presented = bridge.mark_presented()
    reuse = bridge.reuse_presented()

    assert presented.frame is latest_frame
    assert presented.is_new
    assert not presented.reused
    assert presented.source_timestamp == 12.5
    assert reuse.frame is latest_frame
    assert reuse.frame_id == 1
    assert reuse.source_timestamp == 12.5
    assert reuse.reused

    empty_poll = bridge.drain_latest()

    assert empty_poll.frame is None
    assert empty_poll.dequeued == 0
    assert empty_poll.frame_id == 1

    rgbd_q = queue.Queue()
    rgbd_q.put(("rgb", "depth", 23.5))
    rgbd_poll = ScreenFrameBridge(rgbd_q).drain_latest()

    assert rgbd_poll.source_timestamp == 23.5


def test_runtime_direct_renderable_source_does_not_require_depth_texture():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    viewer._runtime_direct_source = True
    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_depth_texture = None

    assert viewer._has_renderable_source_frame()

    viewer._runtime_eye_textures[1] = None
    assert not viewer._has_renderable_source_frame()


def test_openxr_screen_upload_budget_reuses_presented_frame_without_dropping_pending():
    from xr_viewer.core_source_state import CoreSourceStateMixin, ScreenFrameBridge

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    viewer.depth_q = queue.Queue()
    viewer._openxr_screen_upload_budget_ms = 1.0
    viewer._openxr_screen_upload_budget_skip_armed = True
    viewer._pending_source_frame = object()
    time_calls = []
    viewer._fps_breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))
    value_calls = []
    viewer._fps_breakdown_add_value = lambda name, value: value_calls.append((name, value))
    inc_calls = []
    viewer._fps_breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._openxr_screen_frame_bridge = ScreenFrameBridge(viewer.depth_q)
    viewer._openxr_screen_frame_bridge.latest_frame = object()
    viewer._openxr_screen_frame_bridge.latest_frame_id = 1
    viewer._openxr_screen_frame_bridge.last_presented_frame = object()
    viewer._openxr_screen_frame_bridge.last_presented_frame_id = 1
    viewer._update_frame = lambda *args, **kwargs: pytest.fail("upload should be skipped")
    viewer._update_runtime_frame = lambda *args, **kwargs: pytest.fail("upload should be skipped")

    assert viewer._poll_source_frame(upload=True) is False

    assert viewer._pending_source_frame is not None
    assert viewer._openxr_screen_upload_budget_skip_armed is False
    assert ("openxr_reused_screen_frame", 1) in inc_calls
    assert ("openxr_screen_upload_budget_skip", 1) in inc_calls
    assert ("openxr_screen_frame_age_frames", 0.0) in value_calls


def test_openxr_effect_submit_is_timed_outside_screen_upload():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    runtime_result = SimpleNamespace(left_eye=object(), right_eye=object(), depth=object())
    source_q = queue.Queue()
    source_q.put((runtime_result, 10.0))
    viewer = Viewer()
    viewer.depth_q = source_q
    viewer._pending_source_frame = None
    viewer._openxr_screen_upload_budget_ms = 0.0
    viewer._openxr_screen_upload_budget_skip_armed = False
    viewer._last_source_frame_time = 0.0
    viewer._source_resume_grace_until = 0.0
    viewer._source_stalled = False
    viewer._source_stall_count = 0
    viewer._session_running = False
    viewer._session_ready_pending = False
    viewer._render_active_event = None
    viewer._sbs_ts_ring = []
    time_calls = []
    viewer._fps_breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))
    viewer._fps_breakdown_add_value = lambda name, value: None
    viewer._fps_breakdown_inc = lambda name, amount=1: None

    effect_source = object()
    viewer._update_runtime_frame = lambda result: effect_source

    assert viewer._poll_source_frame(upload=True) is True

    assert viewer._pending_runtime_effect_source is effect_source
    names = [name for name, _seconds in time_calls]
    assert names.index("openxr_upload") < names.index("openxr_poll")
    assert "openxr_effect_submit" not in names


def test_runtime_effect_submit_flushes_after_frame_submit():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    submitted = []
    source = object()
    newer_source = object()
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._submit_runtime_effect_source_texture = lambda value: submitted.append(value)

    viewer._queue_runtime_effect_submit(source)
    viewer._queue_runtime_effect_submit(newer_source)
    assert submitted == []
    assert ("openxr_effect_submit_overwrite", 1) in inc_calls

    viewer._flush_runtime_effect_submit()
    assert submitted == [newer_source]
    assert viewer._pending_runtime_effect_source is None

    viewer._flush_runtime_effect_submit()
    assert submitted == [newer_source]

    def _fail_submit(_value):
        raise RuntimeError("effect failed")

    viewer._submit_runtime_effect_source_texture = _fail_submit
    viewer._queue_runtime_effect_submit(source)
    viewer._flush_runtime_effect_submit()

    assert viewer._pending_runtime_effect_source is None
    assert ("openxr_effect_submit_failed", 1) in inc_calls


def test_runtime_effect_submit_prewarms_downsample_after_flush():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    source = object()
    safe_tex = object()
    prepared_tex = object()
    prepared = []
    inc_calls = []
    viewer._pending_runtime_effect_source = source
    viewer._runtime_effect_safe_source_tex = None
    viewer._runtime_effect_safe_source_size = None
    viewer._glow_mode = "surround"
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 1.0
    viewer._screen_light_intensity = 0.0
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._submit_runtime_effect_source_texture = lambda value: None

    def _promote():
        viewer._runtime_effect_safe_source_tex = safe_tex
        viewer._runtime_effect_safe_source_size = (640, 360)
        return safe_tex

    def _prepare(tex, size):
        prepared.append((tex, size))
        return prepared_tex

    viewer._promote_runtime_effect_ready_texture = _promote
    viewer._prepare_glow_downsample_texture = _prepare

    viewer._flush_runtime_effect_submit()

    assert viewer._pending_runtime_effect_source is None
    assert prepared == [(safe_tex, (640, 360))]
    assert ("openxr_effect_downsample_prewarm", 1) in inc_calls


def test_runtime_effect_submit_skips_downsample_prewarm_when_not_needed():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    safe_tex = object()
    viewer._runtime_effect_safe_source_tex = safe_tex
    viewer._runtime_effect_safe_source_size = (640, 360)
    viewer._glow_mode = "veil"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._screen_light_intensity = 0.0
    viewer._prepare_glow_downsample_texture = lambda *_args: pytest.fail("downsample should not be prewarmed")

    viewer._prewarm_runtime_effect_downsample()


def test_runtime_effect_submit_budget_skip_does_not_prewarm_downsample():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    inc_calls = []
    viewer._pending_runtime_effect_source = object()
    viewer._submit_runtime_effect_source_texture = lambda _value: False
    viewer._prewarm_runtime_effect_downsample = lambda: pytest.fail("budget skip should not prewarm")
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    viewer._flush_runtime_effect_submit()

    assert ("openxr_effect_downsample_prewarm_skip", 1) in inc_calls


def test_runtime_effect_submit_prewarm_failure_is_not_submit_failure():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    inc_calls = []
    viewer._pending_runtime_effect_source = object()
    viewer._submit_runtime_effect_source_texture = lambda _value: None
    viewer._prewarm_runtime_effect_downsample = lambda: (_ for _ in ()).throw(RuntimeError("prewarm failed"))
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    viewer._flush_runtime_effect_submit()

    assert ("openxr_effect_downsample_prewarm_failed", 1) in inc_calls
    assert ("openxr_effect_submit_failed", 1) not in inc_calls


def test_runtime_effect_submit_not_queued_when_effect_source_is_not_needed():
    from xr_viewer.core_source_state import CoreSourceStateMixin

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    source = object()
    submitted = []
    viewer._runtime_effects_need_source_texture = lambda: False
    viewer._released = False
    viewer._release_runtime_effect_source_texture = lambda: setattr(viewer, "_released", True)
    viewer._submit_runtime_effect_source_texture = lambda value: submitted.append(value)

    viewer._queue_runtime_effect_submit(source)
    viewer._flush_runtime_effect_submit()

    assert getattr(viewer, "_pending_runtime_effect_source", None) is None
    assert submitted == []
    assert viewer._released


def test_runtime_effect_source_uses_safe_texture_swap_and_reuses_on_failure():
    runtime_eye = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    source_state = (SRC / "xr_viewer" / "core_source_state.py").read_text(encoding="utf-8")
    effects = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    environment_renderer = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "class AsyncEffectResultPool" in runtime_eye
    assert "def ensure_staging" in runtime_eye
    assert "def mark_ready" in runtime_eye
    assert "def promote_ready" in runtime_eye
    assert "def publish" in runtime_eye
    assert "def _ensure_runtime_effect_staging_texture" in runtime_eye
    assert "def _publish_runtime_effect_staging_texture" in runtime_eye
    assert "def _promote_runtime_effect_ready_texture" in runtime_eye
    assert "_runtime_effect_spare_source_tex" in runtime_eye
    assert "self.ready_tex = self.staging_tex" in runtime_eye
    assert "self.safe_tex = self.ready_tex" in runtime_eye
    assert "self.staging_tex = self.spare_tex" in runtime_eye
    assert "self.spare_tex = old_safe_tex" in runtime_eye
    assert "openxr_effect_source_reused_safe" in runtime_eye
    assert "openxr_effect_source_ready_publish" in runtime_eye
    assert "D2S_OPENXR_EFFECT_SOURCE_INTERVAL" in runtime_eye
    assert "openxr_effect_source_interval_skip" in runtime_eye
    assert "openxr_effect_submit" in runtime_eye
    assert "openxr_effect_submit_budget_skip" in runtime_eye
    assert "openxr_effect_submit_overwrite" in source_state
    assert "openxr_glow_downsample_render" in (
        SRC / "xr_viewer" / "core_screen_quality.py"
    ).read_text(encoding="utf-8")
    assert "openxr_glow_downsample_reuse" in (
        SRC / "xr_viewer" / "core_screen_quality.py"
    ).read_text(encoding="utf-8")
    assert "openxr_screen_light_source_reuse" in environment_renderer
    assert "openxr_effect_source_safe_publish" in runtime_eye
    assert "openxr_screen_effect_source_reuse" in effects
    assert "_openxr_effect_submit_budget_skip_armed" in runtime_eye
    assert "self._runtime_effect_result_state = pool.state" in runtime_eye
    assert "self.state = 'idle'" in runtime_eye
    assert "self.state = 'writing'" in runtime_eye
    assert "self.state = 'ready'" in runtime_eye
    assert "self.state = 'safe'" in runtime_eye
    assert "return effect_source_rgb" in runtime_eye
    assert "effect_source_rgb = self._update_runtime_frame(source_frame)" in (
        SRC / "xr_viewer" / "core_source_state.py"
    ).read_text(encoding="utf-8")
    assert "self._queue_runtime_effect_submit(effect_source_rgb)" in (
        SRC / "xr_viewer" / "core_source_state.py"
    ).read_text(encoding="utf-8")
    assert "self._flush_runtime_effect_submit()" in implementation
    submit_flush_block = implementation.split("_submit_openxr_frame(composition_layers)", 1)[1].split(
        "if loop_perf_log_enabled:", 1
    )[0]
    assert submit_flush_block.index("openxr_submit_frame") < submit_flush_block.index(
        "self._flush_runtime_effect_submit()"
    )
    assert "self._runtime_effect_safe_source_frame_id = pool.safe_frame_id" in runtime_eye
    assert "self.ready_frame_id = int(frame_id or 0)" in runtime_eye
    assert "self.safe_frame_id = self.ready_frame_id" in runtime_eye
    update_block = runtime_eye.split("def _update_runtime_effect_source_texture", 1)[1].split(
        "def _release_runtime_eye_texture_resources", 1
    )[0]
    publish_block = runtime_eye.split("def _publish_runtime_effect_staging_texture", 1)[1].split(
        "def _promote_runtime_effect_ready_texture", 1
    )[0]
    promote_block = runtime_eye.split("def _promote_runtime_effect_ready_texture", 1)[1].split(
        "def _try_update_runtime_effect_source_texture_gpu", 1
    )[0]
    assert "self._ensure_runtime_effect_staging_texture(w, h)" in update_block
    assert "pool.publish(w, h, getattr(self, '_frame_count', 0))" in publish_block
    assert "pool.promote_ready()" not in publish_block
    assert "pool.promote_ready()" in promote_block
    assert "self._release_runtime_effect_source_texture()" not in update_block.split(
        "if self._try_update_runtime_effect_source_texture_gpu(frame, w, h):", 1
    )[1]
    assert "getattr(self, '_runtime_effect_safe_source_tex', None)" in effects
    assert "openxr_effect_ready_age_frames" in effects
    assert "getattr(self, '_runtime_effect_source_tex', None)" not in effects.split(
        "def _screen_effect_source_texture", 1
    )[1].split("def _render_glow", 1)[0]
    assert "self._runtime_effect_source_tex" not in implementation
    assert "getattr(self, '_runtime_effect_source_tex'" not in implementation
    assert "_runtime_effect_source_size" not in implementation


def test_async_effect_result_pool_promotes_ready_without_touching_writing_slot(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import AsyncEffectResultPool

    class Tex:
        pass

    class Ctx:
        def __init__(self):
            self.created = []

        def texture(self, size, components, dtype):
            tex = Tex()
            tex.size = size
            tex.components = components
            tex.dtype = dtype
            tex.filter = None
            self.created.append(tex)
            return tex

    pool = AsyncEffectResultPool()
    ctx = Ctx()

    staging = pool.ensure_staging(ctx, 4, 2)
    assert pool.state == "writing"
    pool.mark_ready(4, 2, 7)
    assert pool.state == "ready"
    assert pool.safe_tex is None
    assert pool.ready_tex is staging
    assert pool.staging_tex is None

    assert pool.promote_ready()
    assert pool.state == "safe"
    assert pool.safe_tex is staging
    assert pool.safe_size == (4, 2)
    assert pool.safe_frame_id == 7
    assert pool.ready_tex is None

    next_staging = pool.ensure_staging(ctx, 4, 2)
    assert next_staging is not staging
    assert pool.publish(4, 2, 8)
    assert pool.state == "ready"
    assert pool.ready_tex is next_staging
    assert pool.safe_tex is staging
    assert pool.safe_frame_id == 7

    assert pool.promote_ready()
    assert pool.state == "safe"
    assert pool.safe_tex is next_staging
    assert pool.safe_frame_id == 8


def test_async_effect_result_pool_reuses_overwritten_ready_as_spare(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import AsyncEffectResultPool

    class Tex:
        def __init__(self):
            self.release_calls = 0

        def release(self):
            self.release_calls += 1

    class Ctx:
        def texture(self, size, components, dtype):
            tex = Tex()
            tex.size = size
            tex.components = components
            tex.dtype = dtype
            tex.filter = None
            return tex

    pool = AsyncEffectResultPool()
    ctx = Ctx()

    first_ready = pool.ensure_staging(ctx, 4, 2)
    pool.publish(4, 2, 7)
    second_ready = pool.ensure_staging(ctx, 4, 2)
    pool.publish(4, 2, 8)

    assert pool.ready_tex is second_ready
    assert pool.spare_tex is first_ready
    assert first_ready.release_calls == 0

    assert pool.promote_ready()
    assert pool.safe_tex is second_ready
    assert pool.staging_tex is first_ready

def test_runtime_effect_ready_promotes_once_per_frame(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    class Viewer(CoreRuntimeEyeMixin):
        pass

    ready_tex = object()
    viewer = Viewer()
    viewer._frame_count = 11
    viewer._runtime_effect_result_pool = SimpleNamespace(
        ready_tex=ready_tex,
        safe_tex=None,
        promote_calls=0,
        promote_ready=lambda: False,
    )
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._sync_runtime_effect_pool_attrs = lambda: None

    def _promote_ready():
        viewer._runtime_effect_result_pool.promote_calls += 1
        viewer._runtime_effect_result_pool.safe_tex = ready_tex
        viewer._runtime_effect_result_pool.ready_tex = None
        return True

    viewer._runtime_effect_result_pool.promote_ready = _promote_ready

    assert viewer._promote_runtime_effect_ready_texture() is ready_tex
    assert viewer._promote_runtime_effect_ready_texture() is ready_tex
    assert viewer._runtime_effect_result_pool.promote_calls == 1
    assert ("openxr_effect_source_safe_publish", 1) in inc_calls
    assert ("openxr_effect_source_promote_reuse", 1) in inc_calls

    viewer._frame_count = 12
    assert viewer._promote_runtime_effect_ready_texture() is ready_tex
    assert viewer._runtime_effect_result_pool.promote_calls == 2


def test_runtime_effect_submit_budget_reuses_safe_texture_on_next_frame(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    class Viewer(CoreRuntimeEyeMixin):
        pass

    viewer = Viewer()
    viewer._openxr_effect_submit_budget_ms = 0.001
    viewer._openxr_effect_submit_budget_skip_armed = False
    viewer._updated = 0
    inc_calls = []
    time_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))

    def _update(_frame):
        viewer._updated += 1
        time.sleep(0.001)

    viewer._update_runtime_effect_source_texture = _update

    assert viewer._submit_runtime_effect_source_texture(object()) is True
    assert viewer._submit_runtime_effect_source_texture(object()) is False

    assert viewer._updated == 1
    assert viewer._openxr_effect_submit_budget_skip_armed is False
    assert any(name == "openxr_effect_submit" for name, _seconds in time_calls)
    assert ("openxr_effect_submit_budget_skip", 1) in inc_calls
    assert ("openxr_effect_source_reused_safe", 1) in inc_calls


def test_runtime_effect_source_missing_frame_reuses_safe_texture(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    class Viewer(CoreRuntimeEyeMixin):
        pass

    viewer = Viewer()
    viewer._runtime_effects_need_source_texture = lambda: True
    viewer._released = False
    viewer._release_runtime_effect_source_texture = lambda: setattr(viewer, "_released", True)
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    viewer._update_runtime_effect_source_texture(None)

    assert not viewer._released
    assert ("openxr_effect_source_reused_safe", 1) in inc_calls


def test_openxr_async_phase0_diagnostics_are_wired():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    source_state = (SRC / "xr_viewer" / "core_source_state.py").read_text(encoding="utf-8")
    runtime_eye = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    frame_upload = (SRC / "xr_viewer" / "core_frame_upload.py").read_text(encoding="utf-8")
    environment_renderer = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    screen_quality = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    breakdown = (SRC / "utils" / "breakdown.py").read_text(encoding="utf-8")

    for name in (
        "openxr_upload",
        "openxr_wait_frame",
        "openxr_swapchain_wait",
        "openxr_end_frame",
        "openxr_background",
        "openxr_quad_update",
        "openxr_quad_layer_failed",
        "openxr_projection_render_failed",
        "openxr_overlay_render_failed",
        "openxr_controller_render_failed",
        "openxr_laser_render_failed",
        "openxr_input_trigger_failed",
        "openxr_effect_submit",
        "runtime_eye_d3d11",
        "openxr_d3d11_upload",
        "openxr_layer_count",
        "openxr_new_screen_frame",
        "openxr_reused_screen_frame",
        "openxr_screen_frame_age_frames",
        "openxr_source_latency",
        "openxr_screen_quality_failed",
        "openxr_screen_upload_budget_skip",
        "openxr_projection_screen_skipped",
        "openxr_background_panorama",
        "openxr_background_panorama_failed",
        "openxr_background_env_model",
        "openxr_background_env_model_failed",
        "openxr_background_idle",
        "openxr_effect_source_promote_reuse",
        "openxr_wall_light_mask_loaded",
        "openxr_wall_light_mask_missing",
        "openxr_wall_light_mask_disabled",
        "openxr_wall_light_mask_failed",
        "openxr_glow_downsample_failed",
    ):
        assert (
            name in implementation
            or name in source_state
            or name in runtime_eye
            or name in frame_upload
            or name in environment_renderer
            or name in screen_quality
        )

    assert "wall_mask=" in breakdown
    assert "loaded:{rate('openxr_wall_light_mask_loaded')" in breakdown
    assert "fx_promote_reuse={rate('openxr_effect_source_promote_reuse')" in breakdown
    assert "screen_quality_failed={rate('openxr_screen_quality_failed')" in breakdown
    assert "fx_ds_failed={rate('openxr_glow_downsample_failed')" in breakdown
    assert "bg_path=panorama:{rate('openxr_background_panorama')" in breakdown
    assert "env_failed:{rate('openxr_background_env_model_failed')" in breakdown
    assert "overlay_failed={rate('openxr_overlay_render_failed')" in breakdown
    assert "controller_failed={rate('openxr_controller_render_failed')" in breakdown
    assert "laser_failed={rate('openxr_laser_render_failed')" in breakdown

    assert "D2S_OPENXR_SCREEN_QUAD" in implementation
    assert "D2S_OPENXR_ASYNC_EFFECTS" in implementation
    assert "D2S_OPENXR_PANORAMA_BACKGROUND" in implementation
    assert "D2S_OPENXR_SCREEN_UPLOAD_BUDGET_MS" in implementation
    assert "D2S_OPENXR_EFFECT_SUBMIT_BUDGET_MS" in implementation
    assert "'D2S_OPENXR_SCREEN_QUAD', '1'" in implementation
    assert "'D2S_OPENXR_ASYNC_EFFECTS', '1'" in implementation
    assert "'D2S_OPENXR_PANORAMA_BACKGROUND', '1'" in implementation
    assert "'D2S_OPENXR_SCREEN_UPLOAD_BUDGET_MS',\n            4.0" in implementation
    assert "'D2S_OPENXR_EFFECT_SUBMIT_BUDGET_MS',\n            4.0" in implementation
    assert "self._xr_quad_layer_enabled = bool(self._openxr_screen_quad_enabled)" in implementation
    assert "kwargs.get('xr_quad_layer_enabled', self._openxr_screen_quad_enabled)" not in implementation
    assert "viewer._fps_breakdown_add_value = callbacks.breakdown_add_value" in (
        SRC / "xr_viewer" / "openxr_runtime.py"
    ).read_text(encoding="utf-8")
    assert "def _wait_swapchain_image" in implementation
    assert implementation.count("xr.wait_swapchain_image") == 1
    assert "xr.wait_swapchain_image" not in (SRC / "xr_viewer" / "core_quad_layer.py").read_text(encoding="utf-8")
    quad_build_block = implementation.split("updated_quad_eyes = self._update_quad_layer_swapchains()", 1)[1].split(
        "if loop_trace_enabled:", 1
    )[0]
    assert "try:" in quad_build_block
    assert "quad_layer = self._make_quad_layer(quad_eye_index)" in quad_build_block
    assert "openxr_quad_layer_failed" in quad_build_block
    assert "self._xr_quad_layer_active = False" in quad_build_block
    assert "break" in quad_build_block
    trigger_block = implementation.split("# Trigger input -fires mouse clicks", 1)[1].split(
        "def _ensure_env_model_initialized", 1
    )[0]
    assert "try:" in trigger_block
    assert "self._handle_triggers()" in trigger_block
    assert "openxr_input_trigger_failed" in trigger_block
    pbo_projection_block = implementation.split("# PBO fallback: two-phase loop", 1)[1].split(
        "# Phase 2: map PBOs", 1
    )[0]
    assert "try:" in pbo_projection_block
    assert "self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)" in pbo_projection_block
    assert "openxr_projection_render_failed" in pbo_projection_block
    assert "xr.release_swapchain_image(swapchain, self._xr_sc_release_info)" in pbo_projection_block
    assert "for _pending in d3d11_pending:" in pbo_projection_block
    pbo_upload_block = implementation.split("# Phase 2: map PBOs", 1)[1].split(
        "else:\n                    for eye_index in range(2):", 1
    )[0]
    assert "try:" in pbo_upload_block
    assert "self._upload_pbo_to_d3d11(pbo_id, d3d11_tex, sc_w, sc_h)" in pbo_upload_block
    assert "openxr_projection_render_failed" in pbo_upload_block
    assert "xr.release_swapchain_image(swapchain, self._xr_sc_release_info)" in pbo_upload_block
    assert "eye_layer_views = []" in pbo_upload_block
    opengl_projection_block = implementation.split("raw_fbo, mgl_fbo = self._get_or_create_fbo", 1)[1].split(
        "eye_layer_views.append(xr.CompositionLayerProjectionView", 1
    )[0]
    assert "try:" in opengl_projection_block
    assert "self._render_eye(eye_index, mgl_fbo, view_mat, proj_mat)" in opengl_projection_block
    assert "openxr_projection_render_failed" in opengl_projection_block
    assert "xr.release_swapchain_image(swapchain, self._xr_sc_release_info)" in opengl_projection_block
    render_eye_aux_block = implementation.split("def _try_aux_render", 1)[1].split(
        "self.ctx.screen.use()", 1
    )[0]
    assert "try:" in render_eye_aux_block
    assert "callback()" in render_eye_aux_block
    assert "self._breakdown_inc(metric)" in render_eye_aux_block
    assert "openxr_overlay_render_failed" in implementation
    assert "openxr_controller_render_failed" in implementation
    assert "openxr_laser_render_failed" in implementation


def test_openxr_d3d11_interop_hot_path_has_no_glfinish_ext_memory_wait():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    interop = (SRC / "xr_viewer" / "core_d3d_interop.py").read_text(encoding="utf-8")
    d3d_interop = (SRC / "xr_viewer" / "d3d_interop.py").read_text(encoding="utf-8")

    assert not (SRC / "xr_viewer" / "d3d11_backend.py").exists()
    assert "glFinish" not in implementation
    assert "glFinish" not in interop
    assert "elif self._interop_mode == 'ext_mem'" not in implementation
    assert "_ext_shared_tex" not in implementation
    assert "_ext_shared_tex" not in interop
    assert "_load_ext_memory_object" not in interop
    assert "_load_ext_memory_object" not in d3d_interop
    assert "_create_d3d11_shared_texture" not in d3d_interop
    assert "_blit_ext_to_swapchain" not in interop
    assert "def _submit_openxr_frame(layers):" in implementation


def test_quad_layer_update_is_not_nested_under_projection_layer_views():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_tail = implementation.split("# On the first valid frame", 1)[1].split(
        "_submit_openxr_frame(composition_layers)", 1
    )[0]
    update_idx = render_tail.index("updated_quad_eyes = self._update_quad_layer_swapchains()")
    build_idx = render_tail.index("for quad_eye_index in updated_quad_eyes:")
    failure_idx = render_tail.index("self._xr_quad_layer_failed = True", build_idx)
    render_idx = render_tail.index("eye_layer_views = []")
    append_idx = render_tail.index("for quad_layer_header in quad_layer_headers:")
    assert update_idx < build_idx < failure_idx < render_idx < append_idx
    quad_layer_block = render_tail.split("for quad_layer_header in quad_layer_headers:", 1)[1]
    assert "composition_layers.append(quad_layer_header)" in quad_layer_block

    d3d11_native_block = implementation.split("# Native D3D11 renderer", 1)[0].rsplit(
        "if self._use_d3d11:", 1
    )[1]
    assert "not updated_quad_eyes" in d3d11_native_block
    assert "openxr_projection_screen_skipped" not in d3d11_native_block
    assert d3d11_native_block.index("not updated_quad_eyes") < d3d11_native_block.index(
        "self._d3d11_native_renderer is not None"
    )


def test_quad_layer_gate_requires_runtime_direct_textures_and_swapchains():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_enabled = True
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._quad_swapchains = {0: object(), 1: object()}
    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_eye_texture_size = (1920, 1080)

    assert viewer._quad_layer_can_replace_projection_screen() is True
    assert viewer._quad_layer_unavailable_reason() is None

    left_tex = viewer._runtime_eye_textures[0]
    viewer._runtime_eye_textures[1] = None
    assert viewer._quad_layer_source_texture(1)[0] is left_tex
    assert viewer._quad_layer_unavailable_reason() is None
    assert viewer._quad_layer_can_replace_projection_screen() is True

    viewer._runtime_eye_textures[0] = None
    assert viewer._quad_layer_unavailable_reason() == "missing_source_texture"
    assert viewer._quad_layer_can_replace_projection_screen() is False

    viewer._runtime_eye_textures = [object(), object()]
    viewer._screen_curved = True
    assert viewer._quad_layer_unavailable_reason() == "curved_screen"
    assert viewer._quad_layer_can_replace_projection_screen() is False

    viewer._screen_curved = False
    viewer._runtime_direct_source = False
    assert viewer._quad_layer_unavailable_reason() == "not_runtime_direct"
    assert viewer._quad_layer_can_replace_projection_screen() is False

    viewer._runtime_direct_source = True
    viewer._xr_quad_layer_active = False
    assert viewer._quad_layer_unavailable_reason() == "inactive"
    assert viewer._quad_layer_can_replace_projection_screen() is False

    viewer._xr_quad_layer_failed = True
    assert viewer._quad_layer_unavailable_reason() == "failed"
    assert viewer._quad_layer_can_replace_projection_screen() is False


def test_quad_layer_update_requires_both_eyes_for_quad_submit():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_enabled = True
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._quad_swapchains = {0: object(), 1: object()}
    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_eye_texture_size = (1920, 1080)
    viewer._quad_swapchain_array_size = {0: 1, 1: 1}
    viewer._update_quad_layer_swapchain = lambda eye_index: eye_index == 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._update_quad_layer_swapchains() == []
    assert viewer._xr_quad_layer_active is False
    assert viewer._xr_quad_layer_failed is True
    assert ("openxr_quad_layer_failed", 1) in inc_calls


def test_quad_layer_status_hotkey_does_not_toggle_back_to_projection():
    from xr_viewer.core_window_input import CoreWindowInputMixin

    class Viewer(CoreWindowInputMixin):
        def _publish_runtime_config(self):
            self.published += 1

    viewer = Viewer()
    viewer.published = 0
    viewer._xr_quad_layer_enabled = True
    viewer._xr_quad_layer_active = False
    viewer._xr_quad_layer_failed = False
    viewer._quad_swapchains = {0: object(), 1: object()}
    viewer._quad_swapchain_array_size = {0: 1}
    viewer._xr_quad_layer_stereo_boost = 1.0

    viewer._toggle_quad_layer_compare()
    assert viewer._xr_quad_layer_active is True
    assert viewer._preset_name_overlay == "Quad Layer Screen"

    viewer._toggle_quad_layer_compare()
    assert viewer._xr_quad_layer_active is True
    assert viewer._preset_name_overlay == "Quad Layer Screen"
    assert viewer.published == 2

    viewer._xr_quad_layer_failed = True
    viewer._toggle_quad_layer_compare()
    assert viewer._xr_quad_layer_active is False
    assert viewer._preset_name_overlay == "Projection Screen (Quad unavailable)"


def test_quad_layer_pose_state_is_cached_per_frame():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        def _ensure_screen_dimensions(self):
            self.screen_height = 1.35

        def _screen_pose_quat_xyzw(self):
            self.quat_calls += 1
            return 0.0, 0.0, 0.0, 1.0

    viewer = Viewer()
    viewer.quat_calls = 0
    viewer._frame_count = 7
    viewer.screen_yaw = 0.1
    viewer.screen_pitch = 0.2
    viewer.screen_roll = 0.3
    viewer.screen_pan_x = 0.4
    viewer.screen_pan_y = 0.5
    viewer.screen_distance = 2.0
    viewer.screen_width = 2.4
    viewer.screen_height = 1.35
    viewer._xr_quad_layer_debug_offset = 0.0
    viewer._xr_quad_layer_debug_logged = False

    first = viewer._quad_layer_pose_state()
    second = viewer._quad_layer_pose_state()
    viewer._frame_count = 8
    third = viewer._quad_layer_pose_state()

    assert first is second
    assert third is not first
    assert viewer.quat_calls == 2
