# 4K Stereo Synthesis Lab

`4k-stereo-synthesis-lab` 是面向 Desktop2Stereo / OpenXR host 的 4K 立体生成实验核心库。它不负责桌面捕捉、GUI、OpenXR session 或安装包，而是提供可复用的 depth provider、stereo synthesis、输出打包、preset、benchmark 和视觉回归工具。

当前主线目标是在 4K RGB + depth 输入下，验证 layered / occlusion-aware / symmetric stereo synthesis 是否能超过简单 depth-shift 路线，并为后续 GUI/OpenXR host 提供稳定 API。

## 当前状态

- 4K `fast` baseline、`quality_4k`、`hq_4k` synthesis 已实现。
- Half-SBS、Full-SBS、TAB、mono、depth map、anaglyph、interleaved、Leia 输出已支持。
- Distill-Any-Depth Base/Large @ 518 的 PyTorch、ONNX CUDA、Native TensorRT 路径已接入。
- Native TensorRT 支持 engine dtype 检测、预分配、常驻 provider/session。
- Host preset/API 层已实现：`cinema`、`game_low_latency`、`still_image_hq`、`debug_export`、`auto`。
- OpenXR roll-adaptive per-eye render core 已实现，但完整 OpenXR runtime/swapchain 不在本仓库范围内。

## 硬边界

- 不为了性能降低 depth 推理分辨率。
- 不改变当前 Base 模型 `294x518` / `depth_resolution=518` 路径的语义。
- 不把模型、engine、outputs、Python 便携环境提交到 GitHub。
- 不把模型产物写进 Desktop2Stereo 原项目模型目录。
- GUI、桌面捕捉、OpenXR session、安装包、产品级错误 UI 属于外部 host 项目。

## 目录

| 路径 | 用途 |
|---|---|
| `src/stereo_runtime/` | 推荐对外导入包，面向 Desktop2Stereo/host runtime |
| `src/stereo_lab/` | 兼容入口和当前核心实现代码 |
| `tests/` | 单元测试和 smoke contract 测试 |
| `scripts/benchmark/` | 性能测试和 profile |
| `scripts/tools/` | ONNX 导出、深度图生成、对比、视觉回归工具 |
| `scripts/smoke/` | Host/API smoke 示例 |
| `scripts/examples/` | demo 和 OpenXR preview |
| `scripts/windows/` | 可见窗口 Windows 启动脚本 |
| `docs/` | 当前文档入口 |
| `docs/benchmark/` | benchmark 与优化记录 |
| `docs/archive/` | 早期设计文档归档 |

## 重要文档

- 当前交接入口：[docs/00-api-handoff-progress.md](docs/00-api-handoff-progress.md)
- 项目目标：[docs/00-goals.md](docs/00-goals.md)
- Host API 合同：[docs/15-host-api-contract.md](docs/15-host-api-contract.md)
- Preset 调用示例：[docs/14-host-api-preset-examples.md](docs/14-host-api-preset-examples.md)
- OpenXR 计划：[docs/12-openxr-stereo-runtime-plan.md](docs/12-openxr-stereo-runtime-plan.md)
- Benchmark 汇总：[docs/benchmark/07-depth-backend-benchmark.md](docs/benchmark/07-depth-backend-benchmark.md)

## 快速验证

```powershell
.\python3\python.exe -B -m pytest -q
```

Host API smoke：

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out -
```

4K end-to-end benchmark：

```powershell
.\python3\python.exe -B scripts\benchmark\bench_end_to_end_4k.py --rgb 4K.jpg --backend quality_4k --layers 2 --depth-backend tensorrt_native --output-format half_sbs --output-format full_sbs
```
