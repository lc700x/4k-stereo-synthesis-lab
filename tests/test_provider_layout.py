import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def test_nvidia_provider_compat_imports():
    from stereo_runtime.depth_onnx_provider import DistillAnyDepthBaseOnnxCuda as OldOnnx
    from stereo_runtime.depth_provider import DistillAnyDepthBase518 as OldPyTorch
    from stereo_runtime.depth_trt_native_provider import DistillAnyDepthBaseNativeTensorRt as OldNativeTrt
    from stereo_runtime.depth_trt_provider import DistillAnyDepthBaseTensorRtOrt as OldTrtOrt
    from stereo_runtime.providers.nvidia.onnx_cuda import DistillAnyDepthBaseOnnxCuda
    from stereo_runtime.providers.nvidia.pytorch_cuda import DistillAnyDepthBase518
    from stereo_runtime.providers.nvidia.tensorrt_native import DistillAnyDepthBaseNativeTensorRt
    from stereo_runtime.providers.nvidia.tensorrt_ort import DistillAnyDepthBaseTensorRtOrt

    assert DistillAnyDepthBase518 is OldPyTorch
    assert DistillAnyDepthBaseOnnxCuda is OldOnnx
    assert DistillAnyDepthBaseNativeTensorRt is OldNativeTrt
    assert DistillAnyDepthBaseTensorRtOrt is OldTrtOrt


def test_platform_provider_factory_delegates_to_legacy_factory(monkeypatch):
    import stereo_runtime.providers.factory as factory

    sentinel = object()
    monkeypatch.setattr(factory, "create_depth_provider", lambda config=None: sentinel)

    assert factory.create_platform_depth_provider({"backend": "pytorch_cuda"}) is sentinel
