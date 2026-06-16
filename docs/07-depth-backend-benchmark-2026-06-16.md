# Depth Backend Benchmark Report

Date: 2026-06-16  
Primary test image: `4K.jpg` (`3840x2160`)  
Latest 4K report: `outputs/env_depth_backend_compare_4k/env_depth_backend_compare.md`  
720p reference report: `outputs/env_depth_backend_compare_reuse_session/env_depth_backend_compare.md`  
GPU: RTX 2060

## Summary

The previous benchmark result was misleading.

The old `309s` number was TensorRT engine build time, not inference time.  
The old `3.4s` number was repeated ORT TensorRT session creation / engine load time, not inference time.

The benchmark has been fixed to separate:

```text
setup_ms     = provider/session creation + TensorRT engine build/load
warmup_ms    = first calls using the same provider/session
inference_ms = repeated inference using the same provider/session
```

With provider/session reuse, TensorRT is now the fastest backend.

## TensorRT Runtime Fix

The earlier TensorRT failure was caused by missing DLL discovery:

```text
onnxruntime_providers_tensorrt.dll depends on nvinfer_10.dll
```

The DLL exists at:

```text
python3/Lib/site-packages/tensorrt_libs/
```

The project now auto-discovers and prepends this folder to `PATH` before creating the ORT TensorRT session.

Implementation:

```text
src/stereo_lab/depth_trt_provider.py
```

Key helper:

```python
ensure_tensorrt_dll_path()
```

## Backend Priority

`distill_base_nvidia` now targets:

```text
TensorRT EP -> ONNX CUDA IOBinding -> PyTorch CUDA
```

ONNX CUDA defaults to IOBinding.

## Environment Comparison

| Env | Torch | Torch CUDA | ONNX Runtime | TensorRT DLL source |
|---|---|---|---|---|
| `python3` | `2.7.1+cu128` | `12.8` | `1.26.0` | `python3/Lib/site-packages/tensorrt_libs` |
| `python-cu13` | `2.14.0.dev20260615+cu130` | `13.0` | `1.27.0` | reuses `python3/Lib/site-packages/tensorrt_libs` |

Both environments expose:

```text
TensorrtExecutionProvider
CUDAExecutionProvider
CPUExecutionProvider
```

## Corrected 720p Reference Results

Command:

```powershell
.\python3\python.exe -B scripts\compare_python_env_depth_backends.py --rgb outputs\demo\fast_half_sbs.png --out-dir outputs\env_depth_backend_compare_reuse_session --warmup 2 --iters 5
```

Results:

| Env | Backend | Status | Setup ms | Warmup mean ms | Inference mean ms |
|---|---|---:|---:|---:|---:|
| `python3` | TensorRT EP | OK | `4363.944` | `110.387` | `14.417` |
| `python3` | ONNX CUDA IOBinding | OK | `1061.619` | `412.142` | `28.895` |
| `python3` | PyTorch CUDA | OK | `11765.374` | `123.420` | `44.999` |
| `python-cu13` | TensorRT EP | OK | `4221.489` | `68.487` | `14.888` |
| `python-cu13` | ONNX CUDA IOBinding | OK | `1076.401` | `357.801` | `29.321` |
| `python-cu13` | PyTorch CUDA | OK | `13439.888` | `170.471` | `45.095` |

## Real 4K Results

Command:

```powershell
.\python3\python.exe -B scripts\compare_python_env_depth_backends.py --rgb 4K.jpg --out-dir outputs\env_depth_backend_compare_4k --warmup 2 --iters 10
```

Input:

```text
4K.jpg: 3840x2160 RGB
```

Results:

| Env | Backend | Status | Setup ms | Warmup mean ms | Inference mean ms | Inference median ms | Inference p90 ms |
|---|---|---:|---:|---:|---:|---:|---:|
| `python3` | TensorRT EP | OK | `4483.240` | `111.276` | `24.971` | `25.201` | `25.585` |
| `python3` | ONNX CUDA IOBinding | OK | `1068.342` | `394.309` | `37.483` | `37.967` | `40.058` |
| `python3` | PyTorch CUDA | OK | `10739.280` | `121.267` | `45.812` | `44.175` | `45.385` |
| `python-cu13` | TensorRT EP | OK | `4458.005` | `84.599` | `29.725` | `26.432` | `34.610` |
| `python-cu13` | ONNX CUDA IOBinding | OK | `1063.333` | `376.167` | `38.670` | `38.832` | `42.331` |
| `python-cu13` | PyTorch CUDA | OK | `9884.255` | `177.033` | `46.746` | `46.809` | `47.934` |

4K conclusion:

```text
python3 is the recommended main environment.
python-cu13 remains an experimental environment.
TensorRT EP is the fastest depth backend when provider/session is reused.
```

Environment delta on 4K:

| Backend | python3 mean | python-cu13 mean | Current judgment |
|---|---:|---:|---|
| TensorRT EP | `24.971 ms` | `29.725 ms` | `python3` faster; `python-cu13` had timing spikes |
| ONNX CUDA IOBinding | `37.483 ms` | `38.670 ms` | similar, `python3` slightly faster |
| PyTorch CUDA | `45.812 ms` | `46.746 ms` | similar, `python3` slightly faster |

## Interpretation

Provider/session reuse is mandatory.

Correct runtime pattern:

```python
provider = DistillAnyDepthBaseTensorRtOrt(...)
provider.load()  # once at startup

for frame in frames:
    depth = provider.predict(frame)
```

Incorrect runtime pattern:

```python
for frame in frames:
    provider = DistillAnyDepthBaseTensorRtOrt(...)
    depth = provider.predict(frame)
```

The previous benchmark accidentally followed the second pattern by constructing a new provider every iteration.

## Current Stability Judgment

| Backend | Stability | Speed on RTX 2060 | Notes |
|---|---|---:|---|
| TensorRT EP | Usable | ~`25 ms` on 4K input | Fastest after setup; setup/build must happen before realtime loop |
| ONNX CUDA IOBinding | Usable | ~`37-39 ms` on 4K input | Good fallback when TensorRT fails |
| PyTorch CUDA | Most stable | ~`46 ms` on 4K input | Correctness baseline and final fallback |

## Remaining Caveats

- Current timing still includes preprocessing and postprocessing.
- Input path still converts PyTorch tensor to numpy before ORT. This should be optimized later.
- TensorRT cache produced separate engine files for `python3` and `python-cu13`, likely due to ORT version differences:

```text
onnxruntime 1.26.0 -> one engine key
onnxruntime 1.27.0 -> another engine key
```

## Next Steps

1. Split benchmark into `preprocess_ms`, `model_ms`, and `postprocess_ms`.
2. Keep TensorRT provider/session alive in any realtime API.
3. Investigate zero-copy or lower-copy ORT input/output paths.
4. Run the same 4K benchmark on higher-end GPUs such as RTX 3090 / RTX 5070.
5. Decide whether ORT TensorRT EP is enough, or whether native TensorRT engine API is needed later.
