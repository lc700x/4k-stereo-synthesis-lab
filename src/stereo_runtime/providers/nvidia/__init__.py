"""NVIDIA CUDA / ONNX / TensorRT depth providers."""

from .onnx_cuda import DistillAnyDepthBaseOnnxCuda
from .pytorch_cuda import DistillAnyDepthBase518, GenericAutoDepthProvider
from .tensorrt_native import DistillAnyDepthBaseNativeTensorRt, build_native_tensorrt_engine
from .tensorrt_ort import DistillAnyDepthBaseTensorRtOrt

__all__ = [
    "DistillAnyDepthBase518",
    "GenericAutoDepthProvider",
    "DistillAnyDepthBaseOnnxCuda",
    "DistillAnyDepthBaseNativeTensorRt",
    "DistillAnyDepthBaseTensorRtOrt",
    "build_native_tensorrt_engine",
]
