import torch

from stereo_runtime import OpenXRRenderConfig, OpenXRRuntimeResult, StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import DepthProfileResult


class FakeDepthProvider:
    info = {
        "provider": "fake",
        "model_name": "fake",
        "model_id": "fake",
        "depth_backend": "fake",
    }

    def load(self):
        self.loaded = True

    def predict_profile(self, rgb):
        depth = torch.linspace(
            0.0,
            1.0,
            steps=rgb.shape[-2] * rgb.shape[-1],
            dtype=rgb.dtype,
            device=rgb.device,
        ).reshape(1, 1, rgb.shape[-2], rgb.shape[-1])
        return DepthProfileResult(depth=depth, preprocess_ms=1.0, model_ms=2.0, postprocess_ms=3.0)


def test_process_openxr_frame_defaults_to_rgb_depth_runtime_result():
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert isinstance(result, OpenXRRuntimeResult)
    assert result.depth.shape == (1, 1, 12, 16)
    assert result.source_rgb is rgb
    assert result.left_eye is rgb
    assert result.right_eye is rgb
    assert result.timing["depth_total_ms"] >= 0.0
    assert result.timing["openxr_render_ms"] == 0.0
    assert result.timing["pack_ms"] == 0.0
    assert result.debug_info["runtime_output_format"] == "openxr_rgb_depth"
    assert result.debug_info["runtime_output_dtype"] == "float32"
    assert result.debug_info["backend"] == "openxr_viewer_shader_dibr"


def test_process_openxr_frame_can_pack_prewarped_eye_views_to_uint8(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_PREWARP_EYES", "1")
    monkeypatch.setenv("D2S_RUNTIME_OUTPUT_UINT8", "1")
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert result.left_eye.dtype == torch.uint8
    assert result.right_eye.dtype == torch.uint8
    assert result.debug_info["runtime_output_dtype"] == "uint8"
    assert result.debug_info["runtime_output_pack_backend"] == "torch_float_eye_to_uint8"
    assert "pack_ms" in result.timing


def test_process_openxr_frame_can_keep_prewarped_float_eye_views_when_disabled(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_PREWARP_EYES", "1")
    monkeypatch.setenv("D2S_OPENXR_RUNTIME_OUTPUT_UINT8", "0")
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert result.left_eye.dtype == torch.float32
    assert result.right_eye.dtype == torch.float32
    assert result.debug_info["runtime_output_dtype"] == "float32"
    assert result.debug_info["runtime_output_pack_backend"] == "none"
