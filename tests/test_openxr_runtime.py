import logging
import os
import queue
import subprocess
import sys
import time
import types
import ctypes
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

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


def _openxr_config(**overrides):
    values = dict(
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
    values.update(overrides)
    return OpenXRRuntimeConfig(**values)


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
    config = _openxr_config()
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


def test_openxr_frame_pipeline_seeds_screen_bridge_with_renderable_bootstrap_frame():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    screen_presenter = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    run_body = implementation.split("def run(self, first_rgb=None", 1)[1].split("    # Cleanup", 1)[0]
    seed_block = frame_pipeline.split("def seed_first_frame", 1)[1].split("def render_frame", 1)[0]

    assert "frame_pipeline.seed_first_frame(" in run_body
    assert "first_source_frame" not in run_body
    assert "first_source_frame = (first_runtime_result, first_frame_ts)" in seed_block
    assert "first_source_frame = (first_rgb, first_depth, first_frame_ts)" in seed_block
    assert "if viewer._has_renderable_source_frame():" in seed_block
    assert seed_block.index("if viewer._has_renderable_source_frame():") < seed_block.index(
        "bridge.mark_presented(first_source_frame)"
    )
    assert "viewer._mark_source_frame_received()" in seed_block
    assert "_pending_source_frame" not in seed_block
    assert seed_block.index("bridge.mark_presented(first_source_frame)") < seed_block.index(
        "viewer._mark_source_frame_received()"
    )


def test_openxr_optional_extensions_filters_runtime_extensions(monkeypatch):
    import xr_viewer.implementation_support as support

    fake_xr = SimpleNamespace(
        enumerate_instance_extension_properties=lambda: [
            SimpleNamespace(extension_name=b"XR_KHR_composition_layer_equirect2"),
            SimpleNamespace(extension_name="XR_UNUSED"),
        ]
    )
    monkeypatch.setattr(support, "xr", fake_xr)

    assert support._openxr_optional_extensions(
        "XR_KHR_composition_layer_equirect2",
        "XR_MISSING",
        None,
    ) == ["XR_KHR_composition_layer_equirect2"]


def test_run_openxr_mode_bootstraps_without_waiting_for_first_runtime_frame(monkeypatch):
    calls = []
    callback_calls = []

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
    callbacks = OpenXRRuntimeCallbacks(
        update_runtime_config=lambda *args, **kwargs: None,
        render_active_set=lambda: callback_calls.append("render_set"),
        render_active_clear=lambda: callback_calls.append("render_clear"),
        source_active_set=lambda: callback_calls.append("source_set"),
        wait_idle_clear=lambda: callback_calls.append("wait_idle_clear"),
        bootstrap_done_set=lambda: callback_calls.append("bootstrap_done"),
    )

    viewer = run_openxr_mode(queue.Queue(), _openxr_config(frame_size=(1920, 1080)), callbacks)

    assert isinstance(viewer, FakeViewer)
    assert calls[0]["frame_size"] == (1920, 1080)
    assert calls[1]["run"] == {"first_runtime_result": None, "first_frame_ts": None}
    assert callback_calls == ["source_set", "render_clear", "wait_idle_clear", "bootstrap_done"]


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
    assert "OpenXR D3D11 projection submit" not in implementation
    assert "pbo_glreadpixels" not in implementation
    assert "openxr_d3d11_pbo_readback" not in implementation
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
    screen_presenter = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")
    quad_layer = (SRC / "xr_viewer" / "core_quad_layer.py").read_text(encoding="utf-8")
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
    assert "screen_disparity_uv = max(0.0, runtime_rgb_depth_max_disparity_px) / float(runtime_rgb_depth_render_width)" not in screen_presenter
    assert "cr = math.cos(self.screen_roll * 0.5)" in quad_layer
    assert "sr = math.sin(self.screen_roll * 0.5)" in quad_layer
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
    viewer._runtime_eye_has_frame = True
    viewer._runtime_depth_texture = None
    viewer._use_d3d11 = False
    viewer._d3d11_native_renderer = None

    assert viewer._has_renderable_source_frame()

    viewer._runtime_eye_textures[1] = None
    assert not viewer._has_renderable_source_frame()

    viewer._use_d3d11 = True
    viewer._d3d11_native_renderer = SimpleNamespace(has_frame=True)
    assert viewer._has_renderable_source_frame()

    viewer._d3d11_native_renderer.has_frame = False
    assert not viewer._has_renderable_source_frame()


def test_openxr_screen_upload_budget_reuses_presented_frame_without_dropping_pending():
    from xr_viewer.core_source_state import CoreSourceStateMixin, ScreenFrameBridge
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    viewer.depth_q = queue.Queue()
    viewer._openxr_screen_upload_budget_ms = 1.0
    viewer._openxr_screen_upload_budget_skip_armed = True
    time_calls = []
    viewer._fps_breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))
    value_calls = []
    viewer._fps_breakdown_add_value = lambda name, value: value_calls.append((name, value))
    inc_calls = []
    viewer._fps_breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._openxr_screen_frame_bridge = ScreenFrameBridge(viewer.depth_q)
    viewer._openxr_screen_frame_bridge.latest_frame = object()
    viewer._openxr_screen_frame_bridge.latest_frame_id = 2
    viewer._openxr_screen_frame_bridge.last_presented_frame = object()
    viewer._openxr_screen_frame_bridge.last_presented_frame_id = 1
    viewer._update_frame = lambda *args, **kwargs: pytest.fail("upload should be skipped")
    viewer._update_runtime_frame = lambda *args, **kwargs: pytest.fail("upload should be skipped")

    assert ScreenLayerPresenter(viewer).poll_screen_frame() is False

    assert viewer._openxr_screen_frame_bridge.has_unpresented_frame()
    assert viewer._openxr_screen_upload_budget_skip_armed is False
    assert ("openxr_reused_screen_frame", 1) in inc_calls
    assert ("openxr_screen_upload_budget_skip", 1) in inc_calls
    assert ("openxr_screen_frame_age_frames", 1.0) in value_calls


def test_openxr_upload_keeps_pending_until_frame_is_renderable():
    from xr_viewer.core_source_state import CoreSourceStateMixin, ScreenFrameBridge
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class Viewer(CoreSourceStateMixin):
        pass

    runtime_result = SimpleNamespace(left_eye=object(), right_eye=object(), depth=object())
    pending_frame = (runtime_result, 10.0)
    viewer = Viewer()
    viewer.depth_q = queue.Queue()
    viewer._openxr_screen_frame_bridge = ScreenFrameBridge(viewer.depth_q)
    viewer._openxr_screen_frame_bridge.latest_frame = pending_frame
    viewer._openxr_screen_frame_bridge.latest_frame_id = 1
    viewer._openxr_screen_upload_budget_ms = 0.0
    viewer._openxr_screen_upload_budget_skip_armed = False
    viewer._runtime_direct_source = False
    viewer.color_tex = None
    viewer.depth_tex = None
    viewer._sbs_ts_ring = []
    inc_calls = []
    time_calls = []
    viewer._fps_breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._fps_breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))
    viewer._fps_breakdown_add_value = lambda name, value: None
    viewer._update_runtime_frame = lambda _result: None

    assert ScreenLayerPresenter(viewer).poll_screen_frame() is False

    assert viewer._openxr_screen_frame_bridge.has_unpresented_frame()
    assert viewer._openxr_screen_frame_bridge.last_presented_frame is None
    assert ("openxr_screen_upload_not_renderable", 1) in inc_calls
    assert "openxr_upload" in [name for name, _seconds in time_calls]


def test_openxr_upload_does_not_present_reused_runtime_eye_as_new_frame():
    from xr_viewer.core_source_state import CoreSourceStateMixin, ScreenFrameBridge
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class Viewer(CoreSourceStateMixin):
        pass

    runtime_result = SimpleNamespace(left_eye=object(), right_eye=object(), depth=object())
    pending_frame = (runtime_result, 10.0)
    viewer = Viewer()
    viewer.depth_q = queue.Queue()
    viewer._openxr_screen_frame_bridge = ScreenFrameBridge(viewer.depth_q)
    viewer._openxr_screen_frame_bridge.latest_frame = pending_frame
    viewer._openxr_screen_frame_bridge.latest_frame_id = 1
    viewer._openxr_screen_upload_budget_ms = 0.0
    viewer._openxr_screen_upload_budget_skip_armed = False
    viewer._runtime_direct_source = True
    viewer._runtime_eye_has_frame = True
    viewer._runtime_eye_textures = [object(), object()]
    viewer._sbs_ts_ring = []
    viewer._fps_breakdown_inc = lambda name, amount=1: None
    viewer._fps_breakdown_add_time = lambda name, seconds: None
    viewer._fps_breakdown_add_value = lambda name, value: None

    def _update_runtime_frame(_result):
        viewer._runtime_eye_reused_previous_frame = True
        return None

    viewer._update_runtime_frame = _update_runtime_frame

    assert ScreenLayerPresenter(viewer).poll_screen_frame() is False

    assert viewer._openxr_screen_frame_bridge.has_unpresented_frame()
    assert viewer._openxr_screen_frame_bridge.last_presented_frame is None


def test_openxr_effect_submit_is_timed_outside_screen_upload():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class Viewer(CoreSourceStateMixin):
        pass

    runtime_result = SimpleNamespace(left_eye=object(), right_eye=object(), depth=object())
    source_q = queue.Queue()
    source_q.put((runtime_result, 10.0))
    viewer = Viewer()
    viewer.depth_q = source_q
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

    def _update_runtime_frame(_result):
        viewer._runtime_direct_source = True
        viewer._runtime_eye_has_frame = True
        viewer._runtime_eye_textures = [object(), object()]
        return effect_source

    viewer._update_runtime_frame = _update_runtime_frame

    assert ScreenLayerPresenter(viewer).poll_screen_frame() is True

    assert viewer._runtime_effect_submit_scheduler().pending_source is effect_source
    names = [name for name, _seconds in time_calls]
    assert names.index("openxr_upload") < names.index("openxr_poll")
    assert "openxr_effect_submit" not in names


def test_effect_scheduler_owns_latest_only_pending_submit():
    from xr_viewer.effect_scheduler import EffectScheduler

    scheduler = EffectScheduler()
    first = object()
    second = object()
    submitted = []

    assert scheduler.queue_source(first) is False
    assert scheduler.queue_source(second) is True
    assert scheduler.flush_pending_source(lambda value: submitted.append(value)) == 'submitted'
    assert submitted == [second]
    assert scheduler.pending_source is None

    scheduler.queue_source(first)
    assert scheduler.flush_pending_source(lambda _value: False) == 'skipped'
    assert scheduler.pending_source is None


def test_runtime_effect_submit_flushes_after_frame_submit():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_submitter import EffectSubmitter

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    submitter = EffectSubmitter(viewer)
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

    assert submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)
    assert submitted == [newer_source]
    assert viewer._runtime_effect_submit_scheduler().pending_source is None

    assert not submitter.flush_after_submit(should_render=True, screen_frame_uploaded=False)
    assert submitted == [newer_source]

    viewer._queue_runtime_effect_submit(source)
    assert not submitter.flush_after_submit(should_render=True, screen_frame_uploaded=False)
    assert submitted == [newer_source]
    assert viewer._runtime_effect_submit_scheduler().pending_source is None
    assert ("openxr_effect_source_reused_safe", 1) in inc_calls

    def _fail_submit(_value):
        raise RuntimeError("effect failed")

    viewer._submit_runtime_effect_source_texture = _fail_submit
    viewer._queue_runtime_effect_submit(source)
    assert submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)

    assert viewer._runtime_effect_submit_scheduler().pending_source is None
    assert ("openxr_effect_submit_failed", 1) in inc_calls


def _publish_effect_safe(scheduler, tex, size=(640, 360), frame_id=3):
    pool = scheduler.pool
    slot = pool._idle_slot()
    slot.tex = tex
    slot.size = size
    pool.writing_slot = slot
    scheduler.publish_completed(size[0], size[1], frame_id)
    scheduler.poll_completed()


def test_runtime_effect_submit_flush_prewarms_downsample_after_submit():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_submitter import EffectSubmitter
    from xr_viewer.effect_worker import EffectWorker

    class Viewer(CoreSourceStateMixin):
        pass

    class Worker(EffectWorker):
        def prewarm_after_submit(self):
            prewarm_calls.append(True)

    viewer = Viewer()
    prewarm_calls = []
    viewer._runtime_effect_submit_scheduler().queue_source(object())
    viewer._submit_runtime_effect_source_texture = lambda _value: None
    submitter = EffectSubmitter(viewer)
    submitter.worker = Worker(viewer)

    assert submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)

    assert viewer._runtime_effect_submit_scheduler().pending_source is None
    assert prewarm_calls == [True]


def test_runtime_effect_worker_failure_does_not_escape_submitter():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_submitter import EffectSubmitter

    class Viewer(CoreSourceStateMixin):
        pass

    class Worker:
        def __init__(self):
            self.calls = 0

        def prewarm_after_submit(self):
            self.calls += 1
            raise RuntimeError("worker failed")

    viewer = Viewer()
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._runtime_effect_submit_scheduler().queue_source(object())
    viewer._submit_runtime_effect_source_texture = lambda _value: None
    submitter = EffectSubmitter(viewer)
    worker = Worker()
    submitter.worker = worker

    assert submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)
    assert viewer._openxr_effect_worker_disabled is True
    assert ("openxr_effect_worker_failed", 1) in inc_calls

    viewer._runtime_effect_submit_scheduler().queue_source(object())
    assert submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)
    assert worker.calls == 1
    assert ("openxr_effect_worker_disabled", 1) in inc_calls


def test_runtime_effect_submit_skips_downsample_prewarm_when_not_needed():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_worker import EffectWorker

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    safe_tex = object()
    scheduler = viewer._runtime_effect_submit_scheduler()
    _publish_effect_safe(scheduler, safe_tex)
    viewer._glow_mode = "veil"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._screen_light_intensity = 0.0
    viewer._prepare_glow_downsample_texture = lambda *_args: pytest.fail("downsample should not be prewarmed")

    EffectWorker(viewer).prewarm_after_submit()


def test_runtime_effect_downsample_prewarm_failure_does_not_escape():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_worker import EffectWorker

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    worker = EffectWorker(viewer)
    scheduler = viewer._runtime_effect_submit_scheduler()
    _publish_effect_safe(scheduler, object())
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._screen_light_intensity = 0.0
    inc_calls = []
    time_calls = []
    prepare_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))
    viewer._prepare_glow_downsample_texture = lambda *args: prepare_calls.append(args) or (_ for _ in ()).throw(RuntimeError("downsample failed"))

    worker.prewarm_after_submit()
    worker.prewarm_after_submit()
    _publish_effect_safe(scheduler, object())
    worker.prewarm_after_submit()

    assert inc_calls.count(("openxr_effect_downsample_prewarm_failed", 1)) == 2
    assert ("openxr_effect_downsample_prewarm_suppressed", 1) in inc_calls
    assert ("openxr_effect_downsample_prewarm", 1) not in inc_calls
    assert [name for name, _seconds in time_calls] == [
        "openxr_effect_downsample_prewarm",
        "openxr_effect_downsample_prewarm",
    ]
    assert len(prepare_calls) == 2


def test_effect_worker_interval_skips_downsample_prewarm():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_worker import EffectWorker

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    scheduler = viewer._runtime_effect_submit_scheduler()
    _publish_effect_safe(scheduler, object(), (640, 360), 5)
    viewer._frame_count = 5
    viewer._openxr_effect_worker_interval = 2
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._screen_light_intensity = 0.0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._prepare_glow_downsample_texture = lambda *_args: pytest.fail("worker interval should skip prewarm")

    EffectWorker(viewer).prewarm_after_submit()

    assert ("openxr_effect_worker_interval_skip", 1) in inc_calls
    assert scheduler.latest_safe_downsample() == (None, None, 5)


def test_effect_worker_interval_allows_due_downsample_publish():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_worker import EffectWorker

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    scheduler = viewer._runtime_effect_submit_scheduler()
    _publish_effect_safe(scheduler, object(), (640, 360), 6)
    downsampled = SimpleNamespace(size=(32, 18))
    viewer._frame_count = 6
    viewer._openxr_effect_worker_interval = 2
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._screen_light_intensity = 0.0
    viewer._breakdown_inc = lambda *_args, **_kwargs: None
    viewer._breakdown_add_time = lambda *_args, **_kwargs: None
    viewer._prepare_glow_downsample_texture = lambda *_args: downsampled

    EffectWorker(viewer).prewarm_after_submit()

    assert scheduler.latest_safe_downsample() == (downsampled, (32, 18), 6)


def test_runtime_effect_submit_budget_includes_prewarm():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_submitter import EffectSubmitter
    from xr_viewer.effect_worker import EffectWorker

    class Viewer(CoreSourceStateMixin):
        pass

    class Worker(EffectWorker):
        def prewarm_after_submit(self):
            time.sleep(0.001)

    viewer = Viewer()
    viewer._openxr_effect_submit_budget_ms = 0.001
    viewer._openxr_effect_submit_budget_skip_armed = False
    viewer._runtime_effect_submit_scheduler().queue_source(object())
    viewer._submit_runtime_effect_source_texture = lambda _value: None
    submitter = EffectSubmitter(viewer)
    submitter.worker = Worker(viewer)

    assert submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)

    assert viewer._openxr_effect_submit_budget_skip_armed is True


def test_runtime_effect_submit_budget_skip_does_not_call_submit():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_submitter import EffectSubmitter

    class Viewer(CoreSourceStateMixin):
        pass

    viewer = Viewer()
    submitter = EffectSubmitter(viewer)
    inc_calls = []
    pending = object()
    viewer._openxr_effect_submit_budget_skip_armed = True
    viewer._runtime_effect_submit_scheduler().queue_source(pending)
    viewer._submit_runtime_effect_source_texture = lambda _value: pytest.fail("budget skip should not submit")
    submitter.worker.prewarm_after_submit = lambda: pytest.fail("budget skip should not prewarm")
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert not submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)

    assert viewer._runtime_effect_submit_scheduler().pending_source is None
    assert viewer._openxr_effect_submit_budget_skip_armed is False
    assert ("openxr_effect_submit_budget_skip", 1) in inc_calls
    assert ("openxr_effect_source_reused_safe", 1) in inc_calls

    submitted = []
    viewer._submit_runtime_effect_source_texture = lambda value: submitted.append(value)
    assert not submitter.flush_after_submit(should_render=True, screen_frame_uploaded=False)
    assert submitted == []


def test_runtime_effect_submit_not_queued_when_effect_source_is_not_needed():
    from xr_viewer.core_source_state import CoreSourceStateMixin
    from xr_viewer.effect_submitter import EffectSubmitter

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
    assert not EffectSubmitter(viewer).flush_after_submit(should_render=True, screen_frame_uploaded=True)

    assert viewer._runtime_effect_submit_scheduler().pending_source is None
    assert submitted == []
    assert viewer._released


def test_runtime_effect_source_uses_safe_texture_swap_and_reuses_on_failure():
    runtime_eye = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    source_state = (SRC / "xr_viewer" / "core_source_state.py").read_text(encoding="utf-8")
    screen_presenter = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")
    effects = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    environment_renderer = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    scheduler_text = (SRC / "xr_viewer" / "effect_scheduler.py").read_text(encoding="utf-8")
    submitter_text = (SRC / "xr_viewer" / "effect_submitter.py").read_text(encoding="utf-8")
    worker_text = (SRC / "xr_viewer" / "effect_worker.py").read_text(encoding="utf-8")

    assert "class EffectResultSlot" in scheduler_text
    assert "class AsyncEffectResultPool" in scheduler_text
    assert "class EffectScheduler" in scheduler_text
    assert "def ensure_staging" in scheduler_text
    assert "def mark_ready" in scheduler_text
    assert "def promote_ready" in scheduler_text
    assert "def publish" in scheduler_text
    assert "def submit_screen_frame" in scheduler_text
    assert "def publish_completed" in scheduler_text
    assert "def poll_completed" in scheduler_text
    assert "def latest_safe" in scheduler_text
    assert "def latest_safe_glow" in scheduler_text
    assert "def latest_safe_light_probe" in scheduler_text
    assert "def latest_safe_downsample" in scheduler_text
    assert "def publish_downsample" in scheduler_text
    assert "prepare_downsample" not in scheduler_text
    assert "prepare(source_tex, source_size)" not in source_state
    assert "prepare(source_tex, source_size)" in worker_text
    assert "def _ensure_runtime_effect_staging_texture" in runtime_eye
    assert "def _publish_runtime_effect_staging_texture" in runtime_eye
    assert "def _promote_runtime_effect_ready_texture" not in runtime_eye
    assert "def _runtime_effect_latest_safe" not in runtime_eye
    assert "def promote_ready_once" in scheduler_text
    assert "_runtime_effect_spare_source_tex" not in runtime_eye
    assert "self.slots = [EffectResultSlot() for _ in range(3)]" in scheduler_text
    assert "self.writing_slot = slot" in scheduler_text
    assert "self.ready_slot = slot" in scheduler_text
    assert "self.safe_slot = self.ready_slot" in scheduler_text
    assert "spare_tex" not in scheduler_text
    assert "openxr_effect_source_reused_safe" in runtime_eye
    assert "openxr_effect_source_ready_publish" in runtime_eye
    assert "D2S_OPENXR_EFFECT_SOURCE_INTERVAL" in runtime_eye
    assert "openxr_effect_source_interval_skip" in submitter_text
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
    assert "openxr_effect_source_safe_publish" in submitter_text
    assert "class EffectWorker" in worker_text
    assert "prewarm_after_submit" in submitter_text
    assert "_run_worker_after_submit" in submitter_text
    assert "openxr_effect_worker_failed" in submitter_text
    assert "_openxr_effect_worker_disabled" in submitter_text
    assert "D2S_OPENXR_EFFECT_WORKER_INTERVAL" in worker_text
    assert "openxr_effect_worker_interval_skip" in worker_text
    assert "_prewarm_runtime_effect_downsample" not in source_state
    assert "_prewarm_runtime_effect_downsample" not in submitter_text
    assert "openxr_screen_effect_source_reuse" in effects
    assert "scheduler.latest_safe_downsample(" in environment_renderer
    assert "scheduler.latest_safe_light_probe()" in environment_renderer
    assert "self._runtime_effect_submit_scheduler().latest_safe_glow()" in effects
    assert "self._runtime_effect_submit_scheduler().latest_safe_downsample(" in effects
    assert "_openxr_effect_submit_budget_skip_armed" in runtime_eye
    assert "_openxr_background_upload_budget_skip_armed" in implementation
    assert "self._runtime_effect_result_state = pool.state" not in runtime_eye
    assert "self.state = 'idle'" in scheduler_text
    assert "self.state = 'writing'" in scheduler_text
    assert "self.state = 'ready'" in scheduler_text
    assert "self.state = 'safe'" in scheduler_text
    assert "return effect_source_rgb" in runtime_eye
    assert "effect_source_rgb = viewer._update_runtime_frame(source_frame)" in screen_presenter
    assert "viewer._queue_runtime_effect_submit(effect_source_rgb)" in screen_presenter
    frame_submitter_text = (SRC / "xr_viewer" / "frame_submitter.py").read_text(encoding="utf-8")
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    assert "self.effect_submitter.flush_after_submit(" in frame_pipeline
    assert "self.frame_submitter.submit(" in frame_pipeline
    assert "openxr_submit_frame" in frame_submitter_text
    assert frame_pipeline.index("self.frame_submitter.submit(") < frame_pipeline.index(
        "background_renderer.flush_pending_upload_after_submit()"
    )
    assert frame_pipeline.index("self.frame_submitter.submit(") < frame_pipeline.index(
        "self.effect_submitter.flush_after_submit("
    )
    assert "self._runtime_effect_safe_source_frame_id = pool.safe_frame_id" not in runtime_eye
    assert "slot.frame_id = int(frame_id or 0)" in scheduler_text
    assert "self._safe_frame_id = self.safe_slot.frame_id" in scheduler_text
    update_block = runtime_eye.split("def _update_runtime_effect_source_texture", 1)[1].split(
        "def _release_runtime_eye_texture_resources", 1
    )[0]
    publish_block = runtime_eye.split("def _publish_runtime_effect_staging_texture", 1)[1].split(
        "def _try_update_runtime_effect_source_texture_gpu", 1
    )[0]
    flush_block = submitter_text.split("def flush_after_submit", 1)[1]
    assert "staging_tex = self._ensure_runtime_effect_staging_texture(w, h)" in runtime_eye
    assert "publish_completed(w, h, getattr(self, '_frame_count', 0))" in publish_block
    assert "poll_completed()" not in publish_block
    assert "promote_ready_once" in flush_block
    assert "openxr_effect_source_interval_skip" not in update_block
    assert "def _flush_runtime_effect_submit" not in source_state
    assert "self._release_runtime_effect_source_texture()" not in update_block.split(
        "if self._try_update_runtime_effect_source_texture_gpu(frame, w, h):", 1
    )[1]
    assert "self._runtime_effect_latest_safe()" not in effects
    assert "self._runtime_effect_submit_scheduler().latest_safe()" in effects
    assert "getattr(self, '_runtime_effect_safe_source_tex', None)" not in effects
    assert "openxr_effect_ready_age_frames" in effects
    assert "_promote_runtime_effect_ready_texture" not in effects
    assert "_promote_runtime_effect_ready_texture" not in environment_renderer
    assert "_promote_runtime_effect_ready_texture" not in source_state
    assert "getattr(self, '_runtime_effect_source_tex', None)" not in effects.split(
        "def _screen_effect_source_texture", 1
    )[1].split("def _render_glow", 1)[0]
    assert "self._runtime_effect_source_tex" not in implementation
    assert "getattr(self, '_runtime_effect_source_tex'" not in implementation
    assert "_runtime_effect_source_size" not in implementation


def test_async_effect_result_pool_promotes_ready_without_touching_writing_slot(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import AsyncEffectResultPool

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
    assert pool.ready_slot.tex is staging
    assert pool.writing_slot is None

    assert pool.promote_ready()
    assert pool.state == "safe"
    assert pool.safe_tex is staging
    assert pool.safe_size == (4, 2)
    assert pool.safe_frame_id == 7
    assert pool.ready_slot is None

    next_staging = pool.ensure_staging(ctx, 4, 2)
    assert next_staging is not staging
    assert pool.publish(4, 2, 8)
    assert pool.state == "ready"
    assert pool.ready_slot.tex is next_staging
    assert pool.safe_tex is staging
    assert pool.safe_frame_id == 7

    assert pool.promote_ready()
    assert pool.state == "safe"
    assert pool.safe_tex is next_staging
    assert pool.safe_frame_id == 8


def test_async_effect_result_pool_reuses_overwritten_ready_as_spare(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import AsyncEffectResultPool

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

    assert pool.ready_slot.tex is second_ready
    assert any(slot.tex is first_ready and slot.state == "idle" for slot in pool.slots)
    assert first_ready.release_calls == 0

    assert pool.promote_ready()
    assert pool.safe_tex is second_ready
    assert pool.writing_slot is None
    assert any(slot.tex is first_ready and slot.state == "idle" for slot in pool.slots)


def test_async_effect_result_pool_never_writes_safe_slot(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import AsyncEffectResultPool

    class Tex:
        pass

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
    safe = pool.ensure_staging(ctx, 4, 2)
    pool.publish(4, 2, 7)
    assert pool.promote_ready()

    staging = pool.ensure_staging(ctx, 4, 2)

    assert staging is not safe
    assert pool.safe_tex is safe
    assert pool.writing_slot is not pool.safe_slot
    assert len(pool.slots) == 3


def test_effect_scheduler_publishes_completed_result_before_consumers_read_safe(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import EffectScheduler

    class Tex:
        pass

    class Ctx:
        def texture(self, size, components, dtype):
            tex = Tex()
            tex.size = size
            tex.components = components
            tex.dtype = dtype
            tex.filter = None
            return tex

    scheduler = EffectScheduler()
    staging = scheduler.submit_screen_frame(Ctx(), 8, 4)
    assert scheduler.latest_safe() == (None, None, 0)
    assert scheduler.latest_safe_glow() == (None, None, 0)
    assert scheduler.latest_safe_light_probe() == (None, None, 0)

    scheduler.publish_completed(8, 4, 21)
    assert scheduler.latest_safe() == (None, None, 0)

    assert scheduler.poll_completed()
    assert scheduler.latest_safe() == (staging, (8, 4), 21)
    assert scheduler.latest_safe_glow() == (staging, (8, 4), 21)
    assert scheduler.latest_safe_light_probe() == (staging, (8, 4), 21)
    assert not scheduler.poll_completed()
    assert scheduler.latest_safe() == (staging, (8, 4), 21)


def test_effect_scheduler_downsample_does_not_fallback_to_full_safe_texture(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import EffectScheduler

    scheduler = EffectScheduler()
    _publish_effect_safe(scheduler, object(), (1920, 1080), 9)

    assert scheduler.latest_safe_downsample() == (None, None, 9)


def test_effect_scheduler_owns_safe_downsample_lookup(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import EffectScheduler

    scheduler = EffectScheduler()
    downsampled = SimpleNamespace(size=(2, 1))
    _publish_effect_safe(scheduler, object(), (8, 4), 21)
    scheduler.publish_downsample(downsampled, (2, 1), 21)

    assert scheduler.latest_safe_downsample() == (downsampled, (2, 1), 21)


def test_effect_scheduler_downsample_rejects_stale_publish(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import EffectScheduler

    scheduler = EffectScheduler()
    _publish_effect_safe(scheduler, object(), (1920, 1080), 9)
    scheduler.publish_downsample(object(), (96, 54), 8)

    assert scheduler.latest_safe_downsample() == (None, None, 9)


def test_effect_scheduler_promotes_ready_once_per_frame():
    from xr_viewer.effect_scheduler import EffectScheduler

    class Pool:
        def __init__(self):
            self.calls = 0
            self.safe_tex = object()
            self.safe_size = (8, 4)
            self.safe_frame_id = 11

        def promote_ready(self):
            self.calls += 1
            return True

    pool = Pool()
    scheduler = EffectScheduler(pool)

    assert scheduler.promote_ready_once(11) == 'promoted'
    assert scheduler.promote_ready_once(11) == 'reused'
    assert pool.calls == 1
    assert scheduler.promote_ready_once(12) == 'promoted'
    assert pool.calls == 2


def test_runtime_effect_submit_budget_arms_next_frame_skip(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    class Viewer(CoreRuntimeEyeMixin):
        pass

    viewer = Viewer()
    viewer._openxr_effect_submit_budget_ms = 0.001
    viewer._openxr_effect_submit_budget_skip_armed = False
    viewer._updated = 0
    time_calls = []
    viewer._breakdown_inc = lambda name, amount=1: None
    viewer._breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))

    def _update(_frame):
        viewer._updated += 1
        time.sleep(0.001)

    viewer._update_runtime_effect_source_texture = _update

    assert viewer._submit_runtime_effect_source_texture(object()) is True

    assert viewer._updated == 1
    assert viewer._openxr_effect_submit_budget_skip_armed is True
    assert any(name == "openxr_effect_submit" for name, _seconds in time_calls)


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
    effect_submitter = (SRC / "xr_viewer" / "effect_submitter.py").read_text(encoding="utf-8")
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    frame_submitter = (SRC / "xr_viewer" / "frame_submitter.py").read_text(encoding="utf-8")
    frame_timing = (SRC / "xr_viewer" / "openxr_frame_timing.py").read_text(encoding="utf-8")
    frame_renderer = (SRC / "xr_viewer" / "openxr_frame_renderer.py").read_text(encoding="utf-8")
    projection_presenter = (SRC / "xr_viewer" / "projection_layer_presenter.py").read_text(encoding="utf-8")
    screen_presenter = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")
    background_presenter = (SRC / "xr_viewer" / "background_presenter.py").read_text(encoding="utf-8")
    background_layer_renderer = (SRC / "xr_viewer" / "background_layer_renderer.py").read_text(encoding="utf-8")
    overlay_presenter = (SRC / "xr_viewer" / "overlay_layer_presenter.py").read_text(encoding="utf-8")
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
        "openxr_quad_screen_overlay",
        "openxr_background_panorama",
        "openxr_background_panorama_failed",
        "openxr_background_env_model",
        "openxr_background_env_model_failed",
        "openxr_background_idle",
        "openxr_background_projection_fallback",
        "openxr_background_layer",
        "openxr_background_layer_upload",
        "openxr_background_layer_upload_failed",
        "openxr_background_upload",
        "openxr_background_upload_budget_skip",
        "openxr_background_layer_failed",
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
            or name in effect_submitter
            or name in frame_pipeline
            or name in frame_submitter
            or name in frame_timing
            or name in frame_renderer
            or name in projection_presenter
            or name in screen_presenter
            or name in background_presenter
            or name in background_layer_renderer
            or name in overlay_presenter
        )

    assert "wall_mask=" in breakdown
    assert "loaded:{rate('openxr_wall_light_mask_loaded')" in breakdown
    assert "fx_promote_reuse={rate('openxr_effect_source_promote_reuse')" in breakdown
    assert "screen_quality_failed={rate('openxr_screen_quality_failed')" in breakdown
    assert "fx_ds_failed={rate('openxr_glow_downsample_failed')" in breakdown
    assert "bg_path=layer:{rate('openxr_background_layer')" in breakdown
    assert "bg_upload={avg_ms('openxr_background_upload')" in breakdown
    assert "upload:{rate('openxr_background_layer_upload')" in breakdown
    assert "budget_skip:{rate('openxr_background_upload_budget_skip')" in breakdown
    assert "upload_failed:{rate('openxr_background_layer_upload_failed')" in breakdown
    assert "fallback:{rate('openxr_background_projection_fallback')" in breakdown
    assert "layer_failed:{rate('openxr_background_layer_failed')" in breakdown
    assert "panorama:{rate('openxr_background_panorama')" in breakdown
    assert "env_failed:{rate('openxr_background_env_model_failed')" in breakdown
    assert "overlay_failed={rate('openxr_overlay_render_failed')" in breakdown
    assert "controller_failed={rate('openxr_controller_render_failed')" in breakdown
    assert "laser_failed={rate('openxr_laser_render_failed')" in breakdown

    assert "D2S_OPENXR_SCREEN_QUAD" not in implementation
    assert "D2S_OPENXR_ASYNC_EFFECTS" in implementation
    assert "D2S_OPENXR_PANORAMA_BACKGROUND" in implementation
    assert "D2S_OPENXR_SCREEN_UPLOAD_BUDGET_MS" in implementation
    assert "D2S_OPENXR_EFFECT_SUBMIT_BUDGET_MS" in implementation
    assert "D2S_OPENXR_BACKGROUND_UPLOAD_BUDGET_MS" in implementation
    assert "'D2S_OPENXR_ASYNC_EFFECTS', '1'" in implementation
    assert "'D2S_OPENXR_PANORAMA_BACKGROUND', '1'" in implementation
    assert "'D2S_OPENXR_SCREEN_UPLOAD_BUDGET_MS',\n            4.0" in implementation
    assert "'D2S_OPENXR_EFFECT_SUBMIT_BUDGET_MS',\n            4.0" in implementation
    assert "'D2S_OPENXR_BACKGROUND_UPLOAD_BUDGET_MS',\n            4.0" in implementation
    assert "_xr_quad_layer_enabled" not in implementation
    assert "_openxr_screen_quad_enabled" not in implementation
    assert "kwargs.get('xr_quad_layer_enabled'" not in implementation
    assert "viewer._fps_breakdown_add_value = callbacks.breakdown_add_value" in (
        SRC / "xr_viewer" / "openxr_runtime.py"
    ).read_text(encoding="utf-8")
    assert "def _wait_swapchain_image" in implementation
    assert implementation.count("xr.wait_swapchain_image") == 1
    assert "xr.wait_swapchain_image" not in (SRC / "xr_viewer" / "core_quad_layer.py").read_text(encoding="utf-8")
    screen_presenter = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")
    assert "from .openxr_frame_pipeline import OpenXRFramePipeline" in implementation
    assert "from .openxr_frame_renderer import OpenXRFrameRenderer" in frame_pipeline
    assert "from .screen_layer_presenter import ScreenLayerPresenter" in frame_renderer
    assert "ScreenLayerPresenter(viewer)" in frame_renderer
    assert "self.screen_presenter.poll_screen_frame()" in frame_renderer
    assert "self.screen_presenter.prepare_frame_layers(" in frame_renderer
    assert "screen_frame_uploaded=screen_frame_uploaded" in frame_renderer
    assert "class ScreenLayerPresenter" in screen_presenter
    assert "def poll_screen_frame" in screen_presenter
    assert "def update_or_reuse" in screen_presenter
    assert "def make_quad_layers" in screen_presenter
    assert "bridge = viewer._screen_frame_bridge()" in screen_presenter
    assert "effect_source_rgb = viewer._update_runtime_frame(source_frame)" in screen_presenter
    assert "self.viewer._update_quad_layer_swapchains(force=screen_frame_uploaded)" in screen_presenter
    assert "quad_layer = viewer._make_quad_layer(quad_eye_index)" in screen_presenter
    assert "raise RuntimeError(f\"missing quad layer for eye {quad_eye_index}\")" in screen_presenter
    assert "openxr_quad_layer_failed" in screen_presenter
    assert "viewer._xr_quad_layer_active = False" in screen_presenter
    trigger_block = implementation.split("# Trigger input -fires mouse clicks", 1)[1].split(
        "def _ensure_env_model_initialized", 1
    )[0]
    assert "try:" in trigger_block
    assert "self._handle_triggers()" in trigger_block
    assert "openxr_input_trigger_failed" in trigger_block
    assert "if updated_quad_eyes:" not in projection_presenter
    assert "openxr_projection_pbo_skipped_for_quad" not in projection_presenter
    assert "def render_d3d11_pbo" not in projection_presenter
    assert "_submit_pbo_readback" not in projection_presenter
    assert "_upload_pbo_to_d3d11" not in projection_presenter
    assert "openxr_projection_d3d11_no_interop_skip" in projection_presenter
    opengl_projection_block = projection_presenter.split("def render_opengl", 1)[1].split(
        "def _projection_view", 1
    )[0]
    assert "try:" in opengl_projection_block
    assert "viewer._render_eye(eye_index, mgl_fbo, view_mat, proj_mat)" in opengl_projection_block
    assert "viewer._preview_active and eye_index == 0 and not updated_quad_eyes" in opengl_projection_block
    assert "glfw.swap_buffers(viewer.window)" in projection_presenter
    assert "openxr_projection_render_failed" in opengl_projection_block
    assert "xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)" in opengl_projection_block
    overlay_aux_block = overlay_presenter.split("def _try_aux_render", 1)[1].split(
        "if viewer._keyboard_visible", 1
    )[0]
    assert "try:" in overlay_aux_block
    assert "callback()" in overlay_aux_block
    assert "viewer._breakdown_inc(metric)" in overlay_aux_block
    assert "openxr_overlay_render_failed" in overlay_presenter
    assert "openxr_controller_render_failed" in overlay_presenter
    assert "openxr_laser_render_failed" in overlay_presenter
    assert "def _try_aux_render" not in implementation


def test_openxr_d3d11_interop_hot_path_has_no_glfinish_ext_memory_wait():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    interop = (SRC / "xr_viewer" / "core_d3d_interop.py").read_text(encoding="utf-8")
    d3d11 = (SRC / "xr_viewer" / "core_openxr_d3d11.py").read_text(encoding="utf-8")
    d3d_interop = (SRC / "xr_viewer" / "d3d_interop.py").read_text(encoding="utf-8")

    assert "interop/PBO" not in d3d11
    assert "PBO readback path" not in d3d11
    assert "skipped instead of reintroducing D3D11 PBO readback" in d3d11
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
    frame_submitter = (SRC / "xr_viewer" / "frame_submitter.py").read_text(encoding="utf-8")
    assert "class FrameSubmitter" in frame_submitter
    assert "xr.end_frame(" in frame_submitter
    assert "def _submit_openxr_frame(layers):" not in implementation


def test_quad_layer_can_skip_empty_projection_layer(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.background_presenter import BackgroundPresenter
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    viewer = SimpleNamespace()
    viewer._quad_layer_screen_presentable = lambda: True
    viewer._background_presenter = BackgroundPresenter(viewer)
    viewer._preview_active = True
    viewer._panorama_background_path = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._keyboard_visible = False
    viewer._keyboard_tex = None
    viewer._aim_mat_l = None
    viewer._aim_mat_r = None
    viewer._grip_mat_l = None
    viewer._grip_mat_r = None
    viewer._border_alpha = 0.0
    viewer._depth_osd_tex = None
    viewer._screen_osd_tex = None
    viewer._preset_osd_tex = None
    viewer._seat_adjust_osd_tex = None
    viewer._brand_osd_tex = None
    viewer._hand_fps_visible = False
    viewer._overlay_tex = None
    viewer._team_fps_visible = False
    viewer._team_status_tex = None
    viewer._calibration_mode = False
    viewer._fps_overlay_visible = False
    viewer._help_tex = None
    viewer._team_status_visible = False
    viewer._team_help_visible = False
    viewer._team_help_tex = None
    presenter = ScreenLayerPresenter(viewer)

    assert presenter.projection_layer_needed() is False

    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    assert presenter.projection_layer_needed() is False
    viewer._env_model_visible = False
    viewer._env_model_prims = []

    viewer._preview_active = False
    assert presenter.projection_layer_needed() is False

    viewer._aim_mat_l = object()
    assert presenter.projection_layer_needed() is True
    viewer._aim_mat_l = None

    viewer._panorama_background_path = "room.hdr"
    viewer._panorama_texture_ready = lambda: None
    assert presenter.projection_layer_needed() is False
    viewer._panorama_texture_ready = lambda: object()
    assert presenter.projection_layer_needed() is True
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._background_layer_renderer.panorama_ready = lambda: (_ for _ in ()).throw(RuntimeError("background gate failed"))
    assert presenter.projection_layer_needed() is True
    assert ("openxr_background_layer_failed", 1) in inc_calls
    viewer._background_layer_renderer = None
    viewer._aim_mat_l = object()
    assert presenter.projection_layer_needed() is True
    viewer._aim_mat_l = None
    viewer._panorama_background_path = None
    viewer._panorama_texture_ready = lambda: None

    viewer._quad_layer_screen_presentable = lambda: True
    assert presenter.projection_layer_needed() is False

    viewer._quad_layer_screen_presentable = lambda: False
    assert presenter.projection_layer_needed() is False

    viewer._quad_layer_screen_presentable = lambda: False
    assert presenter.projection_layer_needed() is False


def test_active_openxr_presenter_drains_source_after_begin_frame():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    run_body = implementation.split("def run(self, first_rgb=None", 1)[1].split("    # Cleanup", 1)[0]
    pre_timing = frame_pipeline.split("frame_state, submit_start = self.timing.begin_frame", 1)[0]
    timing_to_upload = frame_pipeline.split("frame_state, submit_start = self.timing.begin_frame", 1)[1].split(
        "screen_frame_uploaded = False", 1
    )[0]

    assert "if viewer._session_ready_pending or not viewer._has_fresh_source_frame(now):" in pre_timing
    assert "viewer._poll_source_frame(upload=False)" in pre_timing
    assert "self.input.sync_actions()" in timing_to_upload
    assert "xr.begin_frame" not in run_body
    assert "_update_aim_poses" not in run_body
    assert "_poll_controller_input" not in run_body
    assert "self.renderer.render_frame(" in frame_pipeline
    assert "_default_fov" not in run_body
    assert "_default_proj" not in run_body
    assert "_frame_ts_ring.append" not in run_body
    assert "self.default_fov = xr.Fovf(" in frame_pipeline
    assert "def record_presented_frame(self):" in frame_pipeline
    assert "viewer._frame_ts_ring.append" in frame_pipeline
    assert "frame_pipeline.begin_loop_frame()" in run_body
    assert "frame_pipeline.handle_preview_only(now)" in run_body
    assert "frame_pipeline.begin_active_session_frame()" in run_body
    assert "glfw.poll_events()" not in run_body
    assert "self._poll_xr_events()" not in run_body
    assert frame_pipeline.index("self.timing.begin_frame(") < frame_pipeline.index("self.renderer.render_frame(")


def test_openxr_frame_pipeline_runs_hard_realtime_frame_order():
    from xr_viewer.openxr_frame_pipeline import OpenXRFramePipeline

    calls = []

    class FrameTsRing(list):
        def append(self, value):
            calls.append(("record", value))
            super().append(value)

    viewer = SimpleNamespace(
        _openxr_perf_log=False,
        _session_ready_pending=True,
        _frame_count=7,
        _frame_ts_ring=FrameTsRing(),
        actual_fps=0.0,
        _background_layer_renderer=SimpleNamespace(
            flush_pending_upload_after_submit=lambda: (_ for _ in ()).throw(RuntimeError("background failed"))
        ),
    )
    viewer._breakdown_inc = lambda name, amount=1: calls.append(("inc", name, amount))
    viewer._has_fresh_source_frame = lambda now: calls.append(("fresh", now)) or False
    viewer._poll_source_frame = lambda upload=False: calls.append(("poll", upload))
    viewer._has_renderable_source_frame = lambda: True

    pipeline = OpenXRFramePipeline(viewer)
    pipeline.timing = SimpleNamespace(
        begin_frame=lambda *, breakdown_enabled: calls.append(("timing", breakdown_enabled))
        or (SimpleNamespace(should_render=True, predicted_display_time=123), 1.5)
    )
    pipeline.input = SimpleNamespace(
        sync_actions=lambda: calls.append(("sync",)),
        update_controller_frame=lambda *, display_time, dt: calls.append(("input", display_time, dt))
        or "controller_input",
    )
    pipeline.gate = SimpleNamespace(
        handle_ready_or_stall=lambda **kwargs: calls.append(("gate", kwargs)) or (False, False),
        enter_idle_if_needed=lambda _idle: pytest.fail("non-idle frame should not enter idle"),
    )
    pipeline.renderer = SimpleNamespace(
        render_frame=lambda **kwargs: calls.append(("render", kwargs)) or (False, False, True)
    )
    pipeline.frame_submitter = SimpleNamespace(
        submit=lambda layers, **kwargs: calls.append(("submit", layers, kwargs))
    )
    pipeline.effect_submitter = SimpleNamespace(
        flush_after_submit=lambda **kwargs: calls.append(("effect", kwargs)) or True
    )

    pipeline.default_fov = "fov"
    pipeline.default_proj = "proj"
    pipeline.default_proj_d3d = "proj_d3d"

    assert pipeline.render_frame(now=10.0, dt=0.25) is True

    assert calls[0] == ("inc", "openxr_loop", 1)
    assert ("poll", False) in calls
    assert ("inc", "openxr_background_layer_upload_failed", 1) in calls
    names = [call[0] for call in calls]
    assert names.index("timing") < names.index("sync") < names.index("input")
    assert names.index("gate") < names.index("render") < names.index("submit") < names.index("effect") < names.index("record")
    render_call = next(call for call in calls if call[0] == "render")
    assert render_call[1]["display_time"] == 123
    assert render_call[1]["default_fov"] == "fov"
    assert render_call[1]["default_proj"] == "proj"
    assert render_call[1]["default_proj_d3d"] == "proj_d3d"
    effect_call = next(call for call in calls if call[0] == "effect")
    assert effect_call[1] == {"should_render": True, "screen_frame_uploaded": False}
    assert len(viewer._frame_ts_ring) == 1


def test_openxr_frame_timing_waits_and_begins_frame(monkeypatch):
    import xr_viewer.openxr_frame_timing as frame_timing_module
    from xr_viewer.openxr_frame_timing import OpenXRFrameTiming

    calls = []

    class FakeXR:
        @staticmethod
        def wait_frame(session, wait_info):
            calls.append(("wait", session, wait_info))
            return SimpleNamespace(predicted_display_time=2_000_000_000)

        @staticmethod
        def begin_frame(session, begin_info):
            calls.append(("begin", session, begin_info))

    viewer = SimpleNamespace(
        _xr_session="session",
        _xr_frame_wait_info="wait_info",
        _xr_frame_begin_info="begin_info",
        _last_xr_predicted_display_time=1_986_111_111,
    )
    metrics = []
    viewer._breakdown_add_time = lambda name, seconds: metrics.append((name, seconds))
    monkeypatch.setattr(frame_timing_module, "xr", FakeXR)

    frame_state, submit_start = OpenXRFrameTiming(viewer).begin_frame(breakdown_enabled=True)

    assert frame_state.predicted_display_time == 2_000_000_000
    assert submit_start > 0.0
    assert calls == [("wait", "session", "wait_info"), ("begin", "session", "begin_info")]
    assert viewer._last_xr_predicted_display_time == 2_000_000_000
    assert "openxr_wait_frame" in [name for name, _seconds in metrics]
    assert any(
        name == "openxr_predicted_period" and seconds == pytest.approx(1 / 72, rel=0.01)
        for name, seconds in metrics
    )


def test_openxr_frame_input_syncs_actions_and_updates_controllers(monkeypatch):
    import xr_viewer.openxr_frame_input as frame_input_module
    from xr_viewer.openxr_frame_input import OpenXRFrameInput

    sync_calls = []

    class FakeXR:
        @staticmethod
        def sync_actions(session, sync_info):
            sync_calls.append((session, sync_info))

    calls = []
    viewer = SimpleNamespace(
        _xr_actions_sync_info="sync_info",
        _xr_session="session",
        _aim_mat_l=object(),
        _aim_mat_r=None,
        _controller_miss_frames=12,
    )
    viewer._update_aim_poses = lambda display_time: calls.append(("aim", display_time))
    viewer._update_grip_poses = lambda display_time: calls.append(("grip", display_time))
    viewer._smooth_controller_poses = lambda: calls.append(("smooth",))
    viewer._update_trackpad_button_emu = lambda: calls.append(("trackpad",))
    viewer._poll_controller_input = lambda dt: calls.append(("input", dt))
    monkeypatch.setattr(frame_input_module, "xr", FakeXR)

    frame_input = OpenXRFrameInput(viewer)
    frame_input.sync_actions()
    mark = frame_input.update_controller_frame(display_time=123, dt=0.5)

    assert sync_calls == [("session", "sync_info")]
    assert mark == "controller_input"
    assert viewer._controller_miss_frames == 0
    assert calls == [("aim", 123), ("grip", 123), ("smooth",), ("trackpad",), ("input", 0.5)]


def test_openxr_frame_input_clears_missing_controller_state():
    from xr_viewer.openxr_frame_input import OpenXRFrameInput

    calls = []
    viewer = SimpleNamespace(
        _xr_actions_sync_info=None,
        _aim_mat_l=None,
        _aim_mat_r=None,
        _controller_miss_frames=29,
        _emu_y=True,
        _emu_x=True,
        _emu_b=True,
        _emu_a=True,
        _emu_lsc=True,
        _emu_rsc=True,
        _cursor_uv_l=object(),
        _cursor_uv_r=object(),
        _cursor_ctrl=object(),
        _cursor_smooth_uv=object(),
        _grabbed=True,
    )
    viewer._update_aim_poses = lambda display_time: calls.append(("aim", display_time))
    viewer._update_grip_poses = lambda display_time: calls.append(("grip", display_time))
    viewer._smooth_controller_poses = lambda: pytest.fail("missing controllers should not smooth")
    viewer._update_trackpad_button_emu = lambda: pytest.fail("missing controllers should not update trackpad")
    viewer._poll_controller_input = lambda dt: pytest.fail("missing controllers should not poll input")

    mark = OpenXRFrameInput(viewer).update_controller_frame(display_time=456, dt=0.25)

    assert mark == "controller_missing"
    assert calls == [("aim", 456), ("grip", 456)]
    assert viewer._controller_miss_frames == 30
    assert not any([viewer._emu_y, viewer._emu_x, viewer._emu_b, viewer._emu_a, viewer._emu_lsc, viewer._emu_rsc])
    assert viewer._cursor_uv_l is None
    assert viewer._cursor_uv_r is None
    assert viewer._cursor_ctrl is None
    assert viewer._cursor_smooth_uv is None
    assert viewer._grabbed is False


def test_openxr_frame_renderer_builds_layers_from_latest_screen_frame():
    from xr_viewer.openxr_frame_renderer import OpenXRFrameRenderer

    class Viewer:
        _xr_space = "space"

        def __init__(self):
            self.calls = []
            self.time_calls = []

        def _poll_source_frame(self, upload=False):
            self.calls.append(("poll", upload))
            return True

        def _breakdown_add_time(self, name, seconds):
            self.time_calls.append(name)

        def _breakdown_inc(self, name, amount=1):
            self.calls.append(("inc", name, amount))

    viewer = Viewer()
    renderer = OpenXRFrameRenderer(viewer)
    renderer.view_tracker = SimpleNamespace(
        locate_views=lambda *, display_time: viewer.calls.append(("locate", display_time)) or (["view"], True)
    )
    renderer.screen_presenter = SimpleNamespace(
        poll_screen_frame=lambda: viewer.calls.append(("poll", True)) or True,
        prepare_frame_layers=lambda *, screen_frame_uploaded: viewer.calls.append(
            ("prepare", screen_frame_uploaded)
        ) or (["quad"], ["quad_header"], [0, 1], True, ["background_header"]),
        append_frame_layers=lambda layers, **kwargs: viewer.calls.append(
            ("append", layers, kwargs)
        ) or layers.append("layer"),
    )
    renderer.projection_presenter = SimpleNamespace(
        render_projection=lambda **kwargs: viewer.calls.append(("projection", kwargs)) or ["eye_layer"]
    )
    composition_layers = []

    uploaded, adjusted, rendered = renderer.render_frame(
        composition_layers=composition_layers,
        display_time=123,
        default_fov="fov",
        default_proj="proj",
        default_proj_d3d="proj_d3d",
    )

    assert uploaded is True
    assert adjusted is True
    assert rendered is True
    assert composition_layers == ["layer"]
    assert viewer.calls[0] == ("poll", True)
    assert viewer.calls[1] == ("locate", 123)
    assert viewer.calls[2] == ("prepare", True)
    assert viewer.calls[3][0] == "projection"
    assert viewer.calls[3][1]["enabled"] is True
    assert viewer.calls[3][1]["views"] == ["view"]
    assert viewer.calls[3][1]["updated_quad_eyes"] == [0, 1]
    assert viewer.calls[4][0] == "append"
    assert viewer.calls[4][2]["projection_views"] == ["eye_layer"]
    assert viewer.calls[4][2]["projection_space"] == "space"
    assert viewer.calls[4][2]["background_layer_headers"] == ["background_header"]
    assert viewer.calls[4][2]["quad_layer_headers"] == ["quad_header"]
    assert "openxr_quad_update" in viewer.time_calls

    renderer.projection_presenter = SimpleNamespace(
        render_projection=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("projection failed"))
    )
    composition_layers = []
    viewer.calls.clear()

    uploaded, adjusted, rendered = renderer.render_frame(
        composition_layers=composition_layers,
        display_time=124,
        default_fov="fov",
        default_proj="proj",
        default_proj_d3d="proj_d3d",
    )

    assert uploaded is True
    assert adjusted is True
    assert rendered is False
    assert composition_layers == ["layer"]
    assert ("inc", "openxr_projection_render_failed", 1) in viewer.calls
    append_call = next(call for call in viewer.calls if call[0] == "append")
    assert append_call[2]["projection_views"] == []
    assert append_call[2]["background_layer_headers"] == ["background_header"]
    assert append_call[2]["quad_layer_headers"] == ["quad_header"]


def test_view_pose_tracker_owns_locate_cache_and_startup_screen(monkeypatch):
    monkeypatch.chdir(SRC)
    import xr_viewer.view_pose_tracker as view_pose_tracker
    from xr_viewer.view_pose_tracker import ViewPoseTracker

    class FakeXr:
        class ViewConfigurationType:
            PRIMARY_STEREO = "primary"

        class ViewLocateInfo:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

    locate_calls = []
    first_views = [object(), object()]
    second_views = [object(), object()]

    def locate_views(session, info):
        locate_calls.append((session, info.kwargs))
        return object(), first_views if len(locate_calls) == 1 else second_views

    FakeXr.locate_views = staticmethod(locate_views)
    monkeypatch.setattr(view_pose_tracker, "xr", FakeXr)

    class Viewer:
        _xr_session = "session"
        _xr_space = "space"

        def __init__(self):
            self._screen_eye_init = False
            self._head_pos_w = None
            self._head_fwd_w = None
            self._initial_head_y = 0.0
            self.adjust_calls = []
            self.aim_calls = []
            self.grip_calls = []
            self.reset_calls = []

        def _apply_profile_view_pose_to_xr_space(self, views):
            self.adjust_calls.append(views)
            return True

        def _update_aim_poses(self, display_time):
            self.aim_calls.append(display_time)

        def _update_grip_poses(self, display_time):
            self.grip_calls.append(display_time)

        def _head_model_mat4_from_views(self, views):
            assert views is second_views

            class Matrix:
                values = {
                    (0, 2): -0.25,
                    (1, 2): -0.50,
                    (2, 2): -0.75,
                    (0, 3): 1.5,
                    (1, 3): 2.5,
                    (2, 3): 3.5,
                }

                def __getitem__(self, key):
                    return self.values[key]

            return Matrix()

        def _reset_screen_to_default(self, show_border=False):
            self.reset_calls.append(show_border)

    viewer = Viewer()
    views, adjusted = ViewPoseTracker(viewer).locate_views(display_time=123)

    assert views is second_views
    assert adjusted is True
    assert len(locate_calls) == 2
    assert locate_calls[0][0] == "session"
    assert locate_calls[0][1]["display_time"] == 123
    assert locate_calls[0][1]["space"] == "space"
    assert viewer.adjust_calls == [first_views]
    assert viewer.aim_calls == [123]
    assert viewer.grip_calls == [123]
    assert viewer._last_located_views is second_views
    assert viewer._head_pos_w == (1.5, 2.5, 3.5)
    assert viewer._head_fwd_w == (0.25, 0.5, 0.75)
    assert viewer._initial_head_y == 2.5
    assert viewer.reset_calls == [False]
    assert viewer._screen_eye_init is True


def test_active_openxr_presenter_delegates_view_pose_tracking():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    frame_renderer = (SRC / "xr_viewer" / "openxr_frame_renderer.py").read_text(encoding="utf-8")
    view_pose_tracker = (SRC / "xr_viewer" / "view_pose_tracker.py").read_text(encoding="utf-8")
    run_body = implementation.split("def run(self, first_rgb=None", 1)[1].split("    # Cleanup", 1)[0]

    assert "OpenXRFrameRenderer(viewer)" in frame_pipeline
    assert "from .view_pose_tracker import ViewPoseTracker" in frame_renderer
    assert "ViewPoseTracker(viewer)" in frame_renderer
    assert "self.view_tracker.locate_views(" in frame_renderer
    assert "display_time=display_time" in frame_renderer
    assert "xr.locate_views(" not in run_body
    assert "_head_model_mat4_from_views" not in run_body
    assert "_reset_screen_to_default(show_border=False)" not in run_body
    assert "class ViewPoseTracker" in view_pose_tracker
    assert "xr.locate_views(" in view_pose_tracker
    assert "viewer._last_located_views = views" in view_pose_tracker
    assert "viewer._reset_screen_to_default(show_border=False)" in view_pose_tracker


def test_active_openxr_presenter_flushes_effect_after_frame_submit():
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    submit_tail = frame_pipeline.rsplit("self.frame_submitter.submit(", 1)[1].split(
        "if perf_log_enabled:", 1
    )[0]

    assert "EffectSubmitter" in frame_pipeline
    assert "self.effect_submitter.flush_after_submit(" in submit_tail
    assert "should_render=frame_state.should_render" in submit_tail
    assert "screen_frame_uploaded=screen_frame_uploaded" in submit_tail
    assert "if not screen_frame_uploaded or" not in submit_tail
    frame_submitter_text = (SRC / "xr_viewer" / "frame_submitter.py").read_text(encoding="utf-8")
    assert "openxr_submit_frame" in frame_submitter_text
    assert frame_pipeline.index("self.frame_submitter.submit(") < frame_pipeline.index(
        "self.effect_submitter.flush_after_submit("
    )
    assert "self.effect_submitter.flush_after_submit(" in submit_tail


def test_frame_submitter_owns_end_frame_metrics(monkeypatch):
    monkeypatch.chdir(SRC)
    import xr_viewer.frame_submitter as frame_submitter
    from xr_viewer.frame_submitter import FrameSubmitter

    calls = []

    class FakeXr:
        class EnvironmentBlendMode:
            OPAQUE = "opaque"

        class FrameEndInfo:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        @staticmethod
        def end_frame(session, info):
            calls.append((session, info.kwargs))

    class Viewer:
        _xr_session = "session"

        def __init__(self):
            self.inc_calls = []
            self.time_calls = []

        def _breakdown_inc(self, name, amount=1):
            self.inc_calls.append((name, amount))

        def _fps_breakdown_add_time(self, name, seconds):
            self.time_calls.append((name, seconds))

        def _breakdown_add_time(self, name, seconds):
            self.time_calls.append((name, seconds))

    monkeypatch.setattr(frame_submitter, "xr", FakeXr)
    viewer = Viewer()
    layers = [object(), object()]

    FrameSubmitter(viewer).submit(layers, display_time=123, submit_start=1.0)

    assert viewer.inc_calls == [("openxr_layer_count", 2)]
    assert calls[0][0] == "session"
    assert calls[0][1]["display_time"] == 123
    assert calls[0][1]["environment_blend_mode"] == "opaque"
    assert calls[0][1]["layers"] is layers
    assert "openxr_end_frame" in [name for name, _seconds in viewer.time_calls]
    assert "openxr_submit_frame" in [name for name, _seconds in viewer.time_calls]


def test_effect_submitter_flushes_after_rendered_frames(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.effect_scheduler import EffectScheduler
    from xr_viewer.effect_submitter import EffectSubmitter

    class Viewer:
        def __init__(self):
            self.allowed = True
            self.scheduler = EffectScheduler()
            self.submitted = []
            self.inc_calls = []

        def _should_submit_runtime_effect_source(self):
            return self.allowed

        def _runtime_effect_submit_scheduler(self):
            return self.scheduler

        def _submit_runtime_effect_source_texture(self, source):
            self.submitted.append(source)

        def _breakdown_inc(self, name, amount=1):
            self.inc_calls.append((name, amount))

    viewer = Viewer()
    submitter = EffectSubmitter(viewer)
    source = object()
    viewer.scheduler.queue_source(source)

    assert not submitter.flush_after_submit(should_render=False, screen_frame_uploaded=False)
    assert viewer.scheduler.pending_source is source
    assert submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)
    assert viewer.submitted == [source]
    assert viewer.scheduler.pending_source is None

    source = object()
    viewer.scheduler.queue_source(source)
    viewer.allowed = False
    assert not submitter.flush_after_submit(should_render=True, screen_frame_uploaded=True)
    assert viewer.scheduler.pending_source is None
    assert viewer.submitted[-1] is not source
    assert ("openxr_effect_source_interval_skip", 1) in viewer.inc_calls
    assert ("openxr_effect_source_reused_safe", 1) in viewer.inc_calls


def test_active_openxr_presenter_does_not_lazy_load_environment_assets():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    run_body = implementation.split("def run(self, first_rgb=None", 1)[1].split("    # Cleanup", 1)[0]
    preview_block = frame_pipeline.split("def handle_preview_only", 1)[1].split("def begin_active_session_frame", 1)[0]

    assert "viewer._ensure_env_model_initialized(\"Preview-only\")" in preview_block
    assert "_ensure_env_model_initialized(\"Preview-only\")" not in run_body
    assert "_ensure_env_model_initialized(\"Lazy\")" not in frame_pipeline


def test_runtime_direct_upload_failure_reuses_previous_frame_without_cpu_readback():
    runtime_eye = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    source_state = (SRC / "xr_viewer" / "core_source_state.py").read_text(encoding="utf-8")
    update_body = runtime_eye.split("def _update_runtime_frame(self, runtime_result):", 1)[1].split(
        "def _apply_runtime_rgb_depth_config", 1
    )[0]
    fallback_block = update_body.split("if not gpu_uploaded:", 1)[1].split("else:", 1)[0]
    renderable_block = source_state.split("def _has_renderable_source_frame(self):", 1)[1].split(
        "def _should_show_source_border", 1
    )[0]

    assert "openxr_runtime_eye_upload_reused_previous" in fallback_block
    assert "if not getattr(self, '_runtime_eye_has_frame', False):" in fallback_block
    assert "_runtime_eye_to_numpy" not in fallback_block
    assert "cpu_gl" not in fallback_block
    assert "_runtime_eye_has_frame" in renderable_block


def test_preview_only_frame_does_not_flush_soft_effect_submit():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    preview_frame = implementation.split("def _render_preview_only_frame", 1)[1].split(
        "def _screen_uv_to_world", 1
    )[0]

    assert "self._flush_runtime_effect_submit()" not in preview_frame


def test_empty_openxr_frames_do_not_flush_soft_effect_submit():
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    skip_block = frame_pipeline.split("if skip_render:", 1)[1].split("screen_frame_uploaded = False", 1)[0]
    assert "self.flush_background_after_submit()" in skip_block
    assert "self.effect_submitter.flush_after_submit(" not in skip_block

    from xr_viewer.openxr_frame_gate import OpenXRFrameGate

    class Submitter:
        def __init__(self):
            self.calls = []

        def submit(self, layers, *, display_time, submit_start=0.0):
            self.calls.append((layers, display_time, submit_start))

    class Viewer:
        def __init__(self):
            self._session_ready_pending = True
            self._hard_idle_active = False
            self._session_idle_render_timeout = 3.0
            self._source_resume_grace = 0.5
            self._headset_wait_inference_deadline = 1.0
            self._headset_wait_inference_paused = False
            self.inc_calls = []
            self.actions = []

        def _track_session_idle_render(self, should_render, now=None):
            self.actions.append(("idle_track", should_render, now))
            return True

        def _breakdown_inc(self, name, amount=1):
            self.inc_calls.append((name, amount))

        def _has_fresh_source_frame(self, now):
            return False

        def _pause_xr_output_for_source_stall(self):
            self.actions.append("pause")

        def _has_renderable_source_frame(self):
            return False

        def _resume_source_inference(self):
            self.actions.append("resume")

        def _set_render_active(self, value):
            self.actions.append(("render_active", value))

        def _set_source_active(self, value):
            self.actions.append(("source_active", value))

        def _enter_hard_idle_wait(self):
            self.actions.append("hard_idle")

    viewer = Viewer()
    submitter = Submitter()
    gate = OpenXRFrameGate(viewer, submitter)
    frame_state = SimpleNamespace(should_render=False, predicted_display_time=123)

    skip, idle_timeout = gate.handle_ready_or_stall(
        frame_state=frame_state,
        now=10.0,
        composition_layers=[],
        submit_start=1.0,
    )

    assert skip is True
    assert idle_timeout is True
    assert submitter.calls == [([], 123, 1.0)]
    assert ("openxr_no_render", 1) in viewer.inc_calls
    assert "hard_idle" in viewer.actions

    viewer._session_ready_pending = False
    submitter.calls.clear()
    skip, _idle_timeout = gate.handle_ready_or_stall(
        frame_state=SimpleNamespace(should_render=True, predicted_display_time=456),
        now=11.0,
        composition_layers=[],
        submit_start=2.0,
    )

    assert skip is True
    assert submitter.calls == [([], 456, 2.0)]
    assert ("openxr_no_fresh", 1) in viewer.inc_calls
    assert ("openxr_no_renderable", 1) in viewer.inc_calls
    assert "pause" in viewer.actions


def test_stale_source_keeps_rendering_when_quad_layer_has_presented_frame():
    from xr_viewer.openxr_frame_gate import OpenXRFrameGate

    class Submitter:
        def submit(self, *_args, **_kwargs):
            pytest.fail("stale reusable quad frame should not submit an empty frame")

    class Viewer:
        _session_ready_pending = False
        _hard_idle_active = False

        def __init__(self):
            self.inc_calls = []
            self.actions = []

        def _track_session_idle_render(self, should_render, now=None):
            return False

        def _breakdown_inc(self, name, amount=1):
            self.inc_calls.append((name, amount))

        def _has_fresh_source_frame(self, now):
            return False

        def _pause_xr_output_for_source_stall(self):
            self.actions.append("pause")

        def _has_renderable_source_frame(self):
            return False

        def _quad_layer_screen_presentable(self):
            return True

    viewer = Viewer()
    skip, idle_timeout = OpenXRFrameGate(viewer, Submitter()).handle_ready_or_stall(
        frame_state=SimpleNamespace(should_render=True, predicted_display_time=123),
        now=10.0,
        composition_layers=[],
        submit_start=1.0,
    )

    assert skip is False
    assert idle_timeout is False
    assert ("openxr_no_fresh", 1) in viewer.inc_calls
    assert ("openxr_no_renderable", 1) not in viewer.inc_calls
    assert viewer.actions == ["pause"]


def test_quad_layer_update_is_not_nested_under_projection_layer_views():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    frame_renderer = (SRC / "xr_viewer" / "openxr_frame_renderer.py").read_text(encoding="utf-8")
    render_frame = frame_renderer.split("def render_frame", 1)[1]
    poll_idx = render_frame.index("self.screen_presenter.poll_screen_frame()")
    locate_idx = render_frame.index("self.view_tracker.locate_views(")
    prepare_idx = render_frame.index("self.screen_presenter.prepare_frame_layers(")
    render_idx = render_frame.index("self.projection_presenter.render_projection(")
    append_idx = render_frame.index("self.screen_presenter.append_frame_layers(")
    assert poll_idx < locate_idx < prepare_idx < render_idx < append_idx
    frame_pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    assert "from .openxr_frame_pipeline import OpenXRFramePipeline" in implementation
    assert "from .openxr_frame_renderer import OpenXRFrameRenderer" in frame_pipeline
    preview_only = implementation.split("def _render_preview_only_frame", 1)[1].split("def _screen_uv_to_world", 1)[0]
    assert "ScreenLayerPresenter(self)" in preview_only
    assert "screen_presenter.prepare_projection_frame_state()" in preview_only
    assert "ProjectionLayerPresenter" not in implementation
    assert "ViewPoseTracker" not in implementation
    assert "_quad_layer_can_replace_projection_screen" not in frame_renderer
    assert "self.screen_presenter.make_quad_layers(" not in frame_renderer
    assert "viewer._projection_layer_needed()" not in frame_renderer
    assert "composition_layers.append(" not in frame_renderer
    assert "CompositionLayerProjection(" not in frame_renderer
    assert "ctypes.pointer(proj_layer)" not in frame_renderer
    screen_presenter_text = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")
    assert "openxr_projection_layer_skipped" in screen_presenter_text
    assert "xr.CompositionLayerProjection(" in screen_presenter_text
    render_eye_block = implementation.split("def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):", 1)[1]
    assert "draw_projection_screen" not in render_eye_block
    assert "_openxr_draw_projection_screen" not in render_eye_block
    assert "_openxr_projection_screen_source_ready" not in render_eye_block
    assert "quad_unavailable_reason == 'missing_source_texture'" not in render_eye_block
    assert "not self._quad_layer_screen_presentable()" not in render_eye_block
    assert "_openxr_quad_screen_unavailable_reason" in screen_presenter_text
    assert "_openxr_projection_screen_unavailable_reason" not in screen_presenter_text
    assert "_openxr_draw_projection_screen" not in screen_presenter_text
    assert "_openxr_projection_screen_source_ready" not in screen_presenter_text
    assert "_openxr_projection_screen_effects_enabled" not in screen_presenter_text
    prepare_frame_layers = screen_presenter_text.split("def prepare_frame_layers", 1)[1].split("def append_frame_layers", 1)[0]
    assert prepare_frame_layers.index("self.update_or_reuse(") < prepare_frame_layers.index("background_renderer.make_background_layers()")
    assert "def prepare_projection_frame_state" in screen_presenter_text
    assert "def render_projection_screen" not in screen_presenter_text
    assert "_render_screen_background_effects" not in screen_presenter_text
    assert "_render_screen_foreground_effects" not in screen_presenter_text
    assert "screen_presenter.render_projection_screen(" not in render_eye_block
    assert "render_quad_screen_overlay(" in render_eye_block
    assert "openxr_projection_screen_skipped" not in screen_presenter_text
    assert "runtime_rgb_depth_max_disparity_px = (" not in render_eye_block
    background_gate = render_eye_block.split("background_presenter.render_projection_background(", 1)[1].split(
        "if perf_enabled:", 1
    )[0]
    background_presenter = (SRC / "xr_viewer" / "background_presenter.py").read_text(encoding="utf-8")
    background_layer_renderer = (SRC / "xr_viewer" / "background_layer_renderer.py").read_text(encoding="utf-8")
    assert "projection_screen_enabled=" not in background_gate
    assert "background_presenter.projection_fallback_needed()" not in background_gate
    assert "def projection_fallback_needed" in background_presenter
    assert "BackgroundLayerRenderer" in background_presenter
    assert "ready = getattr(self.viewer, '_panorama_texture_ready', None)" in background_layer_renderer
    assert "projection_screen_enabled" not in background_presenter
    assert "if self.projection_fallback_needed():" in background_presenter
    layer_append_block = render_frame.split("self.screen_presenter.append_frame_layers(", 1)[1]
    assert "projection_views=eye_layer_views" in layer_append_block
    assert "projection_space=viewer._xr_space" in layer_append_block
    assert "quad_layer_headers=quad_layer_headers" in layer_append_block
    assert "background_layer_headers=background_layer_headers" in layer_append_block
    projection_call = render_frame.split("self.projection_presenter.render_projection(", 1)[1].split(
        "self.screen_presenter.append_frame_layers(", 1
    )[0]
    assert "enabled=render_projection_layer" in projection_call
    assert "default_proj_d3d=default_proj_d3d" in projection_call
    assert "updated_quad_eyes=updated_quad_eyes" in projection_call
    assert "render_d3d11_native(" not in frame_renderer
    assert "render_nv_dx_interop(" not in frame_renderer
    assert "render_d3d11_pbo(" not in frame_renderer
    assert "render_opengl(" not in frame_renderer
    assert "_get_or_create_fbo" not in frame_renderer
    assert "glBlitFramebuffer" not in frame_renderer
    projection_presenter = (SRC / "xr_viewer" / "projection_layer_presenter.py").read_text(encoding="utf-8")
    render_projection = projection_presenter.split("def render_projection", 1)[1].split(
        "def render_nv_dx_interop", 1
    )[0]
    assert "def render_projection(" in projection_presenter
    assert "if not enabled:" in render_projection
    assert "not viewer._use_d3d11" in render_projection
    assert "return self.render_opengl(" in render_projection
    assert "render_d3d11_native" not in projection_presenter
    assert "return self.render_nv_dx_interop(" in render_projection
    assert "openxr_projection_pbo_skipped_for_quad" not in render_projection
    assert "openxr_projection_d3d11_no_interop_skip" in render_projection
    assert "def render_opengl(" in projection_presenter
    assert "viewer._get_or_create_fbo(" in projection_presenter
    assert "glBlitFramebuffer" in projection_presenter
    assert "eye_sign * screen_disparity_uv" not in projection_presenter
    assert "def render_nv_dx_interop(" in projection_presenter
    assert "_wglDXLockObjectsNV" in projection_presenter
    assert "def render_d3d11_pbo(" not in projection_presenter
    assert "_submit_pbo_readback" not in projection_presenter
    assert "_upload_pbo_to_d3d11" not in projection_presenter


def test_projection_layer_presenter_owns_backend_selection(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.projection_layer_presenter import ProjectionLayerPresenter

    class Renderer:
        has_frame = True

    class Viewer:
        def __init__(self):
            self._use_d3d11 = False
            self._d3d11_native_renderer = None
            self._interop_mode = "none"
            self.inc_calls = []

        def _breakdown_inc(self, name, amount=1):
            self.inc_calls.append((name, amount))

    viewer = Viewer()
    presenter = ProjectionLayerPresenter(viewer)
    calls = []
    presenter.render_opengl = lambda *args, **kwargs: calls.append("opengl") or ["opengl"]
    presenter.render_nv_dx_interop = lambda *args, **kwargs: calls.append("nv_dx") or ["nv_dx"]
    kwargs = dict(
        views=[],
        default_fov=object(),
        default_proj=object(),
        default_proj_d3d=object(),
    )

    assert presenter.render_projection(enabled=False, updated_quad_eyes=(), **kwargs) == []
    assert calls == []

    assert presenter.render_projection(enabled=True, updated_quad_eyes=(), **kwargs) == ["opengl"]
    viewer._use_d3d11 = True
    viewer._d3d11_native_renderer = Renderer()
    assert presenter.render_projection(enabled=True, updated_quad_eyes=(), **kwargs) == []
    viewer._d3d11_native_renderer = None
    viewer._interop_mode = "nv_dx"
    assert presenter.render_projection(enabled=True, updated_quad_eyes=(), **kwargs) == ["nv_dx"]
    viewer._interop_mode = "none"
    assert presenter.render_projection(enabled=True, updated_quad_eyes=(0,), **kwargs) == []
    assert presenter.render_projection(enabled=True, updated_quad_eyes=(), **kwargs) == []
    assert calls == ["opengl", "nv_dx"]
    assert viewer.inc_calls == [("openxr_projection_d3d11_no_interop_skip", 1)] * 3


def test_quad_layer_gate_requires_runtime_direct_textures_and_swapchains():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._runtime_eye_has_frame = True
    viewer._quad_swapchains = {0: object(), 1: object()}
    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_eye_texture_size = (1920, 1080)

    assert viewer._quad_layer_screen_presentable() is True
    assert viewer._quad_layer_unavailable_reason() is None

    left_tex = viewer._runtime_eye_textures[0]
    viewer._runtime_eye_textures[1] = None
    assert viewer._quad_layer_source_texture(1)[0] is left_tex
    assert viewer._quad_layer_unavailable_reason() is None
    assert viewer._quad_layer_screen_presentable() is True

    viewer._runtime_eye_textures[0] = None
    assert viewer._quad_layer_unavailable_reason() == "missing_source_texture"
    assert viewer._quad_layer_screen_presentable() is False
    viewer._quad_swapchain_presented_eyes = {0, 1}
    assert viewer._quad_layer_screen_presentable() is True

    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_eye_has_frame = False
    assert viewer._quad_layer_unavailable_reason() == "missing_source_texture"
    assert viewer._quad_layer_screen_presentable() is True
    viewer._quad_swapchain_presented_eyes = set()
    assert viewer._quad_layer_screen_presentable() is False

    viewer._runtime_eye_textures = [object(), object()]
    viewer._screen_curved = True
    assert viewer._quad_layer_unavailable_reason() == "curved_screen"
    assert viewer._quad_layer_screen_presentable() is False

    viewer._screen_curved = False
    viewer._runtime_direct_source = False
    assert viewer._quad_layer_unavailable_reason() == "not_runtime_direct"
    assert viewer._quad_layer_screen_presentable() is False

    viewer._runtime_direct_source = True
    viewer._xr_quad_layer_active = False
    assert viewer._quad_layer_unavailable_reason() == "inactive"
    assert viewer._quad_layer_screen_presentable() is False

    viewer._xr_quad_layer_failed = True
    assert viewer._quad_layer_unavailable_reason() == "failed"
    viewer._xr_quad_layer_failure_reason = "swapchain_create_failed_RuntimeError"
    assert viewer._quad_layer_unavailable_reason() == "swapchain_create_failed_RuntimeError"
    assert viewer._quad_layer_screen_presentable() is False


def test_d3d11_quad_layer_path_uses_native_renderer_and_swapchains():
    core_quad = (SRC / "xr_viewer" / "core_quad_layer.py").read_text(encoding="utf-8")
    d3d11 = (SRC / "xr_viewer" / "core_openxr_d3d11.py").read_text(encoding="utf-8")

    assert "xr.SwapchainImageD3D11KHR" in d3d11
    assert "self._quad_swapchains[eye_index] = swapchain" in d3d11
    assert "Quad layer D3D11 swapchains" in d3d11
    assert "and self._d3d11_native_renderer is not None" in d3d11
    assert "renderer.has_frame and renderer.runtime_eye_size is not None" in core_quad
    assert "source_tex.render_runtime_eye(sc_image.texture, quad_w, quad_h, eye_index" in core_quad


def test_quad_layer_presented_state_resets_when_swapchains_reset():
    cleanup = (SRC / "xr_viewer" / "core_cleanup.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    opengl = (SRC / "xr_viewer" / "core_openxr_opengl.py").read_text(encoding="utf-8")
    d3d11 = (SRC / "xr_viewer" / "core_openxr_d3d11.py").read_text(encoding="utf-8")

    assert "self._quad_swapchain_presented_eyes = set()" in cleanup
    assert "self._quad_swapchain_presented_eyes = set()" in opengl
    assert "self._quad_swapchain_presented_eyes = set()" in d3d11
    assert "self._background_equirect_failed_key = None" in implementation
    assert "self._background_equirect_pending_tex = None" in implementation
    assert "self._runtime_effect_downsample_failed_key = None" in implementation
    assert "self._background_equirect_failed_key = None" in cleanup
    assert "self._background_equirect_pending_tex = None" in cleanup
    assert "self._runtime_effect_downsample_failed_key = None" in cleanup
    assert cleanup.index("self._background_equirect_uploaded_key = None") < cleanup.index(
        "self._quad_swapchain_images.clear()"
    )
    assert cleanup.index("self._quad_swapchain_array_size.clear()") < cleanup.index(
        "self._quad_swapchain_presented_eyes = set()"
    )


def test_quad_layer_reuses_existing_swapchain_when_screen_frame_is_reused():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._runtime_eye_has_frame = True
    viewer._quad_swapchains = {0: object(), 1: object()}
    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_eye_texture_size = (1920, 1080)
    viewer._quad_swapchain_presented_eyes = {0, 1}
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._update_quad_layer_swapchain = lambda _eye_index: pytest.fail("quad swapchain should be reused")

    assert viewer._update_quad_layer_swapchains(force=False) == [0, 1]
    assert ("openxr_quad_reused_screen_frame", 1) in inc_calls

    viewer._xr_quad_layer_failed = True
    assert viewer._update_quad_layer_swapchains(force=False) == []
    viewer._xr_quad_layer_failed = False

    updated = []
    viewer._update_quad_layer_swapchain = lambda eye_index: updated.append(eye_index) or True
    assert viewer._update_quad_layer_swapchains(force=True) == [0, 1]
    assert updated == [0, 1]


def test_quad_layer_update_requires_both_eyes_for_quad_submit():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._runtime_eye_has_frame = True
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
    assert viewer._quad_layer_unavailable_reason() == "partial_update_without_presented_frame"
    assert ("openxr_quad_layer_failed", 1) in inc_calls


def test_quad_layer_reuses_presented_frame_on_partial_update():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._runtime_eye_has_frame = True
    viewer._quad_swapchains = {0: object(), 1: object()}
    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_eye_texture_size = (1920, 1080)
    viewer._quad_swapchain_array_size = {0: 1, 1: 1}
    viewer._quad_swapchain_presented_eyes = {0, 1}
    viewer._update_quad_layer_swapchain = lambda eye_index: eye_index == 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._update_quad_layer_swapchains(force=True) == [0, 1]
    assert viewer._xr_quad_layer_active is True
    assert viewer._xr_quad_layer_failed is False
    assert ("openxr_quad_update_partial_reuse", 1) in inc_calls


def test_quad_layer_shared_swapchain_reuses_presented_frame_when_source_missing():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        pass

    shared_swapchain = object()
    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._runtime_eye_has_frame = False
    viewer._quad_swapchains = {0: shared_swapchain, 1: shared_swapchain}
    viewer._runtime_eye_textures = [None, None]
    viewer._runtime_eye_texture_size = None
    viewer._quad_swapchain_array_size = {0: 2, 1: 2}
    viewer._quad_swapchain_presented_eyes = {0, 1}
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._update_quad_layer_swapchains(force=True) == [0, 1]
    assert viewer._xr_quad_layer_active is True
    assert viewer._xr_quad_layer_failed is False
    assert ("openxr_quad_missing_source_reuse", 1) in inc_calls


def test_screen_layer_presenter_updates_or_reuses_and_builds_quad_layers(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class Viewer:
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer.render_projection_layer = False
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    update_calls = []
    viewer._update_quad_layer_swapchains = lambda force=False: update_calls.append(force) or [0, 1]
    viewer._quad_layer_screen_presentable = lambda: not viewer.render_projection_layer
    viewer._background_presenter = SimpleNamespace(projection_fallback_needed=lambda: False)
    viewer._keyboard_visible = False
    viewer._keyboard_tex = None
    viewer._aim_mat_l = None
    viewer._aim_mat_r = None
    viewer._grip_mat_l = None
    viewer._grip_mat_r = None
    viewer._border_alpha = 0.0
    viewer._depth_osd_tex = None
    viewer._screen_osd_tex = None
    viewer._preset_osd_tex = None
    viewer._seat_adjust_osd_tex = None
    viewer._brand_osd_tex = None
    viewer._hand_fps_visible = False
    viewer._overlay_tex = None
    viewer._team_fps_visible = False
    viewer._team_status_tex = None
    viewer._calibration_mode = False
    viewer._fps_overlay_visible = False
    viewer._help_tex = None
    viewer._team_status_visible = False
    viewer._team_help_visible = False
    viewer._team_help_tex = None
    viewer._runtime_direct_source = True
    viewer._runtime_eye_textures = [object(), None]
    viewer.color_tex = object()
    viewer.depth_tex = object()

    left_layer = ctypes.c_int(1)
    right_layer = ctypes.c_int(2)

    def _make_quad_layer(eye_index):
        return left_layer if eye_index == 0 else right_layer

    viewer._make_quad_layer = _make_quad_layer
    presenter = ScreenLayerPresenter(viewer)

    quad_layers, quad_layer_headers, updated, render_projection_layer, background_layer_headers = presenter.prepare_frame_layers(
        screen_frame_uploaded=True
    )

    assert update_calls == [True]
    assert quad_layers == [left_layer, right_layer]
    assert presenter._frame_quad_layers == [left_layer, right_layer]
    assert len(quad_layer_headers) == 2
    assert updated == [0, 1]
    assert render_projection_layer is False
    assert background_layer_headers == []
    assert ("openxr_projection_layer_skipped", 1) in inc_calls
    assert viewer._xr_quad_layer_active is True
    assert viewer._xr_quad_layer_failed is False

    import xr_viewer.screen_layer_presenter as screen_layer_presenter

    monkeypatch.setattr(screen_layer_presenter.xr, "CompositionLayerProjection", lambda **_kwargs: ctypes.c_int(3))
    composition_layers = []
    presenter.append_frame_layers(
        composition_layers,
        projection_views=[object()],
        projection_space=object(),
        quad_layer_headers=quad_layer_headers,
        background_layer_headers=[],
    )
    assert len(composition_layers) == 3
    assert composition_layers[1:] == quad_layer_headers
    assert presenter._frame_projection_layer is not None

    monkeypatch.setattr(
        screen_layer_presenter.xr,
        "CompositionLayerProjection",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("projection append failed")),
    )
    composition_layers = []
    presenter.append_frame_layers(
        composition_layers,
        projection_views=[object()],
        projection_space=object(),
        quad_layer_headers=quad_layer_headers,
        background_layer_headers=[],
    )
    assert composition_layers == quad_layer_headers
    assert presenter._frame_projection_layer is None
    assert ("openxr_projection_render_failed", 1) in inc_calls

    viewer._make_quad_layer = lambda _eye_index: None
    quad_layers, quad_layer_headers, updated = presenter.make_quad_layers([0])

    assert quad_layers == []
    assert quad_layer_headers == []
    assert updated == []
    assert viewer._xr_quad_layer_active is False
    assert viewer._xr_quad_layer_failed is True
    assert viewer._xr_quad_layer_failure_reason == "layer_build_failed_RuntimeError"
    assert ("openxr_quad_layer_failed", 1) in inc_calls


def test_screen_layer_presenter_quad_failure_produces_no_screen_layer():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._update_quad_layer_swapchains = lambda force=False: [0]
    viewer._make_quad_layer = lambda _eye_index: None
    viewer._background_layer_renderer = SimpleNamespace(
        make_background_layers=lambda: ([], False),
        _frame_background_layers=[],
        panorama_ready=lambda: False,
        native_background_available=lambda: False,
    )
    viewer._keyboard_visible = False
    viewer._keyboard_tex = None
    viewer._aim_mat_l = None
    viewer._aim_mat_r = None
    viewer._grip_mat_l = None
    viewer._grip_mat_r = None
    viewer._border_alpha = 0.0
    viewer._depth_osd_tex = None
    viewer._screen_osd_tex = None
    viewer._preset_osd_tex = None
    viewer._seat_adjust_osd_tex = None
    viewer._brand_osd_tex = None
    viewer._hand_fps_visible = False
    viewer._overlay_tex = None
    viewer._team_fps_visible = False
    viewer._team_status_tex = None
    viewer._calibration_mode = False
    viewer._fps_overlay_visible = False
    viewer._help_tex = None
    viewer._team_status_visible = False
    viewer._team_help_visible = False
    viewer._team_help_tex = None
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    presenter = ScreenLayerPresenter(viewer)

    quad_layers, quad_layer_headers, updated, render_projection_layer, background_layer_headers = presenter.prepare_frame_layers(
        screen_frame_uploaded=True
    )

    assert quad_layers == []
    assert quad_layer_headers == []
    assert updated == []
    assert render_projection_layer is False
    assert background_layer_headers == []
    assert presenter.quad_screen_unavailable_reason() == "layer_build_failed_RuntimeError"
    assert ("openxr_quad_layer_failed", 1) in inc_calls
    assert ("openxr_projection_layer_skipped", 1) in inc_calls


def test_quad_layer_failure_reason_does_not_enable_projection_screen():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class Viewer(CoreQuadLayerMixin):
        pass

    viewer = Viewer()
    viewer._xr_quad_layer_active = True
    viewer._xr_quad_layer_failed = False
    viewer._screen_curved = False
    viewer._runtime_direct_source = True
    viewer._runtime_eye_has_frame = True
    viewer._quad_swapchains = {0: object(), 1: object()}
    viewer._runtime_eye_textures = [object(), object()]
    viewer._runtime_eye_texture_size = (1920, 1080)
    viewer._quad_swapchain_array_size = {0: 1, 1: 1}
    viewer._update_quad_layer_swapchain = lambda _eye_index: (_ for _ in ()).throw(RuntimeError("boom"))
    viewer._breakdown_inc = lambda *args, **kwargs: None

    assert viewer._update_quad_layer_swapchains() == []
    presenter = ScreenLayerPresenter(viewer)

    assert presenter.quad_screen_unavailable_reason() == "update_failed_RuntimeError"


def test_quad_layer_status_hotkey_does_not_toggle_back_to_projection():
    from xr_viewer.core_window_input import CoreWindowInputMixin

    class Viewer(CoreWindowInputMixin):
        def _publish_runtime_config(self):
            self.published += 1

    viewer = Viewer()
    viewer.published = 0
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
    assert viewer._preset_name_overlay == "Quad Layer unavailable"


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
    assert tuple(first[1]) == (0.4, 0.5, -2.0)


def test_quad_layer_pose_state_uses_logical_screen_basis():
    from xr_viewer.core_quad_layer import CoreQuadLayerMixin

    class Viewer(CoreQuadLayerMixin):
        def _ensure_screen_dimensions(self):
            self.screen_height = 1.35

        def _screen_pose_quat_xyzw(self):
            return 0.0, 0.0, 0.0, 1.0

        def _screen_basis(self):
            self.basis_calls += 1
            return (
                1.25,
                np.array([1.0, 2.0, -3.0], dtype=np.float64),
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            )

    viewer = Viewer()
    viewer.basis_calls = 0
    viewer._frame_count = 1
    viewer.screen_yaw = 0.1
    viewer.screen_pitch = 0.2
    viewer.screen_roll = 0.3
    viewer.screen_pan_x = 9.0
    viewer.screen_pan_y = 9.0
    viewer.screen_distance = 9.0
    viewer.screen_width = 2.5
    viewer.screen_height = 1.35
    viewer._xr_quad_layer_debug_offset = 0.0
    viewer._xr_quad_layer_debug_logged = False

    _quat, pos, size = viewer._quad_layer_pose_state()

    assert viewer.basis_calls == 1
    assert tuple(pos) == (1.0, 2.0, -3.0)
    assert size == (2.5, 1.25)


def test_quad_layer_debug_offset_defaults_to_screen_plane():
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "kwargs.get('xr_quad_layer_debug_offset', 0.0)" in implementation
    assert "kwargs.get('xr_quad_layer_debug_offset', 0.05)" not in implementation
