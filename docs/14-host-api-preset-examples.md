# Host API 与 Preset 调用示例

本文给 GUI、Desktop2Stereo runtime 或 OpenXR host 作为接入参考。更严格的职责边界见 [15-host-api-contract.md](15-host-api-contract.md)。

## Preset 名称

支持的 preset：

```text
auto
cinema
game_low_latency
still_image_hq
debug_export
```

常用别名：

| 输入 | 解析为 |
|---|---|
| `movie` / `video` | `cinema` |
| `game` / `low_latency` | `game_low_latency` |
| `still` / `image` / `hq` | `still_image_hq` |
| `debug` / `export` | `debug_export` |

## RGB + Depth 调用

```python
from stereo_lab import stereo_config_for_preset, synthesize_stereo

config = stereo_config_for_preset(
    "cinema",
    output_format="half_sbs",
    overrides={"depth_strength": 2.2},
)

result = synthesize_stereo(rgb, depth, config)

left_eye = result.left_eye
right_eye = result.right_eye
packed = result.sbs
```

## 只有 RGB 时的常驻 depth provider

```python
from stereo_lab import (
    StereoLabRuntime,
    StereoLabRuntimeConfig,
)

runtime_config = StereoLabRuntimeConfig(
    model_id="lc700x/Distill-Any-Depth-Base-hf",
    model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
    mode="movie",
    stereo_quality="quality_4k",
    output_format="half_sbs",
    depth_backend="auto",
    depth_upsample="bilinear",
    depth_upsample_edge_strength=0.35,
    depth_strength=2.0,
    convergence=0.0,
    ipd=0.064,
    max_shift_ratio=0.05,
    layers=2,
    occlusion=True,
    symmetric=True,
    hole_fill="edge_aware",
    temporal=True,
    temporal_strength=0.75,
    auto_reset_temporal=True,
    edge_threshold=0.04,
    edge_dilation=2,
    fused=True,
)

runtime = StereoLabRuntime(runtime_config)
runtime.load()

for rgb_frame in frames:
    result = runtime.process_rgb_frame(rgb_frame)
```

不要每帧创建 runtime/provider 或重新加载 engine/session。

Desktop2Stereo 的 `capture.py` / `main.py` 仍然负责画面捕捉和颜色前处理，例如 BGR/BGRA 转 RGB；这里的 `rgb` 表示已经完成捕捉侧颜色前处理的当前 RGB 图像帧：

```python
for rgb_frame in frames:
    result = runtime.process_rgb_frame(rgb_frame)
```

D2S/GUI 不需要声明 TensorRT、ONNX、PyTorch 或 Triton 的输入绑定细节。本仓库从 RGB frame 开始，负责 depth provider 输入预处理、推理和后续立体合成；桌面捕捉、窗口裁剪、DPI 处理和 BGR/BGRA 转 RGB 仍属于 D2S capture 管线。

上采/下采规则：

- depth 默认用 `bilinear` 上采回 RGB 源分辨率；
- 需要边缘质量评估时可切到 `depth_upsample="guided"`；
- Half-SBS / Half-TAB 的 torch fallback 使用 `area` 下采；
- Full-SBS / Full-TAB 保留左右眼原始分辨率，不做下采。

Desktop2Stereo 侧只需要传 `model_id` 和下载后的 `model_dir`。ONNX 和 TensorRT artifact 默认放在同一个 `model_dir` 下：

```text
model_fp16_294x518.onnx
model_fp32_294x518.onnx
model_fp16_294x518.trt
```

## Auto 模式

只有用户选择 `auto` 时才启动异步检测。手动 preset 不需要检测线程。

```python
from stereo_lab import (
    AutoModeRuntime,
    AutoModeSignals,
    auto_detection_required,
    stereo_config_for_auto_mode,
    stereo_config_for_preset,
)

selected = "auto"

if auto_detection_required(selected):
    runtime = AutoModeRuntime()
    signals = AutoModeSignals(
        gpu_3d_util=0.72,
        gpu_video_decode_util=0.03,
        input_activity=0.85,
        idle_seconds=0.2,
        audio_active=True,
        fullscreen=True,
        maximized=True,
        frame_motion_score=0.42,
        latency_pressure=0.8,
        target_fps=120.0,
    )
    decision = runtime.update(signals)
    config = stereo_config_for_auto_mode(decision.mode, output_format="half_sbs")
else:
    config = stereo_config_for_preset(selected, output_format="half_sbs")
```

Host 应在后台线程或低频 timer 中采集系统信号，并把 2-3 秒均值或防抖后的快照传给 `AutoModeRuntime`。

## 推荐 preset 用途

| Preset | 用途 | 主要取向 |
|---|---|---|
| `cinema` | 电影、播放器、稳定视频 | 画质和时序稳定 |
| `game_low_latency` | 游戏、桌面交互、快速运动 | 低延迟、较轻 temporal |
| `still_image_hq` | 静态图、暂停画面、图片浏览 | 高质量，可用更重处理 |
| `debug_export` | 视觉回归、算法检查、导出 | debug 信息完整 |

当前用户侧最终模式建议收敛为三类：

- 电影：映射到 `cinema`；
- 游戏：映射到 `game_low_latency`；
- 图片：静态图片、网页缩略图、普通桌面静止画面，映射到 `still_image_hq` 或其轻量变体。

## OpenXR per-eye core

```python
from stereo_lab import openxr_config_for_preset
from stereo_lab.openxr_render import render_openxr_stereo

config = openxr_config_for_preset(
    "cinema",
    screen_roll=screen_roll,
    overrides={"depth_strength": 2.0},
)

result = render_openxr_stereo(rgb, depth, config)
```

注意：本仓库只提供 per-eye render core。完整 OpenXR session、swapchain 和 projection layer 提交由 host 实现。

## Smoke 命令

Host API：

```powershell
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --preset cinema --output-format half_sbs --out -
.\python3\python.exe -B scripts\smoke\host_api_smoke.py --openxr --preset cinema --screen-roll 0.25 --out -
```

Auto runtime：

```powershell
.\python3\python.exe -B scripts\smoke\auto_mode_runtime_demo.py --selected-preset auto --out -
.\python3\python.exe -B scripts\smoke\auto_mode_runtime_demo.py --selected-preset game_low_latency --out -
```

视觉回归生成工具：

```powershell
.\python3\python.exe -B scripts\tools\generate_visual_regression_set.py --rgb 4K.jpg --auto-depth --depth-backend tensorrt_native --preset cinema --out-dir outputs\visual_regression\preset_cinema
```

固定 preset 视觉回归放在最后阶段执行，用于钉住默认参数。
