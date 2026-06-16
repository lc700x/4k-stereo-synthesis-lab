import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_lab.depth_provider import DepthProviderInfo
from stereo_lab.depth_onnx_provider import estimate_distill_any_depth_base_518_onnx_cuda
from stereo_lab.depth_trt_provider import estimate_distill_any_depth_base_518_nvidia


class FakeTorchProvider:
    def __init__(self, **kwargs):
        self.info = DepthProviderInfo(
            provider="fake",
            model_name="Distill-Any-Depth-Base",
            model_id="fake",
            depth_resolution=518,
            cache_dir=str(kwargs.get("cache_dir") or ""),
            load_mode="test",
            depth_backend="pytorch_cpu",
            runtime="fake",
        )

    def predict(self, rgb):
        return torch.zeros(rgb.shape[0], 1, rgb.shape[-2], rgb.shape[-1])


def test_nvidia_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_lab.depth_trt_provider as provider_module

    monkeypatch.setattr(provider_module, "DistillAnyDepthBase518", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_distill_any_depth_base_518_nvidia(
        rgb,
        device="cpu",
        onnx_path=missing,
        prefer_onnx=True,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")


def test_nvidia_provider_requires_tensorrt_when_requested(tmp_path):
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    try:
        estimate_distill_any_depth_base_518_nvidia(
            rgb,
            device="cpu",
            onnx_path=missing,
            require_tensorrt=True,
        )
    except RuntimeError as exc:
        assert "tensorrt" in str(exc)
    else:
        raise AssertionError("expected TensorRT requirement to fail")


def test_onnx_cuda_provider_falls_back_when_onnx_missing(monkeypatch, tmp_path):
    import stereo_lab.depth_onnx_provider as provider_module

    monkeypatch.setattr(provider_module, "DistillAnyDepthBase518", FakeTorchProvider)
    rgb = torch.zeros(1, 3, 8, 8)
    missing = tmp_path / "missing.onnx"

    depth, info = estimate_distill_any_depth_base_518_onnx_cuda(
        rgb,
        device="cpu",
        onnx_path=missing,
        allow_pytorch_fallback=True,
        local_files_only=True,
    )

    assert depth.shape == (1, 1, 8, 8)
    assert info.depth_backend == "pytorch_cpu"
    assert info.onnx_path == str(missing)
    assert "FileNotFoundError" in (info.fallback_reason or "")
