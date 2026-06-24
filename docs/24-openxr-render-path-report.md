# OpenXR 渲染路径报告

## 目的

本报告阐明旧版 Desktop2Stereo OpenXR 路径的工作方式、当前 OpenXR 路径的差异，以及立体模式在立体运行时重写后应如何映射到渲染路径。

核心产品目标是：

- 传统立体模式应保留旧版 OpenXR 行为。
- 影院、游戏和静态图像立体模式在需要 OpenXR 质量输出时，应使用新的完整 `stereo_runtime.synthesize_stereo()` 管线。

## 旧版 Desktop2Stereo OpenXR 流程

旧版 `Desktop2Stereo_v2.5.0Beta` OpenXR 路径未使用完整的 `make_sbs()` 立体合成路径。

旧版 OpenXR 流程：

```text
capture
-> predict_depth(rgb)
-> depth_q.put(rgb, depth, timestamp)
-> OpenXRViewer.run(first_rgb, first_depth)
-> viewer 上传 RGB 纹理 + 深度纹理
-> OpenXR 着色器从 RGB + 深度生成每眼视差
```

相关旧版行为：

- `main.py` 的 OpenXR 分支创建 `OpenXRViewer(ipd=IPD, depth_ratio=DEPTH_STRENGTH, ...)`。
- 将 `rgb` 和 `depth` 直接传入 `viewer.run()`。
- OpenXR viewer 在 `_update_frame(rgb, depth)` 中上传 RGB 和深度。
- viewer 着色器使用 `depth_strength * depth_ratio` 创建立体视差。
- 旧版 `make_sbs(...)` 路径用于旧版流媒体 / 非 OpenXR 输出，不用于 OpenXR。

因此，旧版 OpenXR 最佳描述为 RGB+深度着色器路径，而非完整的 SBS 合成路径。

## 当前渲染路径概念

当前代码库包含三个不同的概念，不得等同对待：

1. OpenXR rgb-depth
2. OpenXR prewarp eyes
3. OpenXR full stereo synthesis eyes（尚未完全接入）

### 1. OpenXR rgb-depth

这是当前默认的低延迟 OpenXR 路径。

```text
runtime:
RGB -> 深度模型 -> 深度后处理

viewer:
RGB + 深度 -> OpenXR 着色器 -> 头显
```

该路径消费的参数：

- `depth_strength`
- `convergence`
- `ipd` / `ipd_mm`
- `stereo_scale`
- `max_shift_ratio`
- `foreground_scale`
- `depth_antialias_strength`

该路径不消费的参数：

- `temporal_strength`
- `edge_threshold`
- `edge_dilation`
- `mask_feather_radius`
- `hole_fill_mode`
- `hole_fill_radius`
- `hole_fill_strength`

优势：

- 在 OpenXR 路径中延迟最低。
- 控制器和 GUI 对核心深度参数的变更开销很小。
- 最接近旧版 OpenXR 行为。
- 适合交互式调参和实时头显使用。

劣势：

- 未使用完整的立体合成管线。
- 影院、游戏和静态图像预设仅部分生效。
- 补洞、边缘膨胀、遮罩羽化和立体时序混合不会影响头显输出。

性能影响：

- 深度推理后的额外开销最低。
- 大部分开销来自深度模型推理和相对廉价的 viewer 着色器。

正确用途：

- 传统的 OpenXR 模式。
- 旧版行为兼容。
- 低延迟使用场景。
- 实时控制器深度调节。

### 2. OpenXR prewarp eyes

该路径在将图像传递给 viewer 之前，先在 runtime 中生成左右眼图像。

```text
runtime:
RGB + 深度 -> render_openxr_stereo() -> left_eye + right_eye

viewer:
上传 left_eye + right_eye -> 头显
```

该路径消费的参数：

- `depth_strength`
- `convergence`
- `ipd`
- `stereo_scale`
- `max_shift_ratio`
- `screen_roll`

该路径不消费的参数：

- `foreground_scale`
- `depth_antialias_strength`
- `temporal_strength`
- `edge_threshold`
- `edge_dilation`
- `mask_feather_radius`
- `hole_fill_*`

优势：

- Runtime 拥有 OpenXR 立体 warp 的控制权。
- Viewer 更接近左右眼纹理展示器。
- 可用作实验性或兼容性路径。

劣势：

- 未调用完整的 `synthesize_stereo()`。
- 未使影院/游戏/静态图像的完整合成参数生效。
- 比 rgb-depth 更昂贵，因为需要生成和传输两幅眼图。

性能影响：

- 中等。
- GPU 和内存带宽开销高于 rgb-depth。
- 通常不如 rgb-depth 适合快速的交互式调参。

正确用途：

- 兼容性实验。
- 在 viewer 端 RGB+深度着色器行为不可取的情况下使用。
- 不是全质量立体合成的最终路径。

### 3. OpenXR full stereo synthesis eyes

这是新的影院、游戏和静态图像模式所期望的质量路径。尚未完全接入 OpenXR 输出。

目标流程：

```text
runtime:
RGB + 深度
-> stereo_runtime.synthesize_stereo()
-> left_eye + right_eye，或将 SBS 分割为眼图
-> OpenXR runtime result

viewer:
runtime 直接上传眼图纹理 -> 头显
```

该路径应消费的参数：

- `quality_4k` / `fast` / `fast_plus`
- `depth_strength`
- `convergence`
- `ipd` / `ipd_mm`
- `stereo_scale`
- `max_shift_ratio`
- `foreground_scale`
- `depth_antialias_strength`
- `temporal_strength`
- `edge_threshold`
- `edge_dilation`
- `mask_feather_radius`
- `hole_fill_mode`
- `hole_fill_radius`
- `hole_fill_strength`

优势：

- 使新的立体运行时重写在 OpenXR 中发挥作用。
- 为影院、游戏和静态图像模式启用全质量处理。
- 按设计使用遮挡、补洞、边缘处理和时序平滑。

劣势：

- 运行时开销最高。
- 延迟高于 rgb-depth。
- 参数变更需要重新计算合成的眼图。
- 4K 质量模式对于游戏类工作负载可能开销较大。

性能影响：

- 最高。
- 在深度推理之后增加了完整的合成开销：深度后处理、warp/合成、遮挡 mask、补洞、时序混合以及输出打包/上传。
- 必须按预设和分辨率分别进行基准测试。

正确用途：

- 影院质量模式。
- 静态图像高质量模式。
- 游戏模式仅在使用 `fast` 或 `fast_plus` 等低延迟预设并通过性能验证后使用。

## 当前的接入缺口

当前管线包含一个部分回退，可以调用完整合成但未将其最终结果用于 OpenXR 显示。

当前行为：

```python
if ctx.run_mode == "OpenXR" and ctx.openxr_runtime_direct:
    runtime_result = ctx.stereo_runtime.process_openxr_frame(...)
else:
    runtime_result = ctx.stereo_runtime.process_rgb_frame(...)
```

当 `openxr_runtime_direct` 为 false 时，`process_rgb_frame()` 确实会运行完整的立体合成。然而，OpenXR 队列当前接收的是 RGB+深度回退数据：

```python
ctx.queue_put_latest(ctx.runtime_q, ((frame_rgb, fallback_depth), capture_start_time))
```

这意味着完整的合成输出实际上并未在 OpenXR 中显示。Viewer 回退到了 RGB+深度着色器路径。

需要修复的内容：

```text
process_rgb_frame()
-> 使用 StereoRuntimeResult.left_eye/right_eye 或拆分 StereoRuntimeResult.sbs
-> 打包为 OpenXR runtime result
-> 发送到 viewer 的 runtime 直接眼图纹理路径
```

## 推荐模式映射

### 传统立体模式

使用 OpenXR rgb-depth。

理由：

- 这复现了旧版 OpenXR 行为。
- 延迟低。
- 很好地支持实时深度调节。

暴露或强调以下控件：

- `Depth Strength`
- `Convergence`
- `IPD`
- `Stereo Scale`
- `Max Shift Ratio`
- `Foreground Scale`
- `Depth Antialias Strength`

在 OpenXR rgb-depth 中隐藏、禁用或标记为不适用：

- `Temporal Strength`
- `Edge Threshold`
- `Edge Dilation`
- `Mask Feather Radius`
- `Hole Fill Mode`
- `Hole Fill Radius`
- `Hole Fill Strength`

### 影院模式

接入后使用 OpenXR full stereo synthesis eyes。

理由：

- 影院模式受益于高质量的遮挡和补洞处理。
- 延迟不如视觉质量关键。

推荐后端：

- 性能允许时使用 `quality_4k`。
- 如果运行时开销过高，回退到平衡或更快的预设。

### 游戏模式

仅在使用低延迟预设的情况下使用 OpenXR full stereo synthesis eyes。

理由：

- 游戏模式需要更低的延迟。
- 完整的 `quality_4k` 可能开销过大。

推荐后端：

- `fast`
- `fast_plus`
- 在启用昂贵的补洞或时序设置之前仔细进行基准测试。

### 静态图像模式

接入后使用 OpenXR full stereo synthesis eyes。

理由：

- 延迟不那么重要。
- 高质量的补洞、边缘处理和时序设置更有价值。

推荐后端：

- `quality_4k` 或静态图像高质量预设。

## 汇总表

| 路径 | 兼容旧版 | 完整合成 | 低延迟 | 使用补洞/边缘/时序 | 最佳用途 |
|---|---|---|---|---|---|
| OpenXR rgb-depth | 是 | 否 | 最佳 | 否 | 传统 OpenXR，实时调参 |
| OpenXR prewarp eyes | 否 | 否 | 中等 | 否 | 兼容性 / 实验 |
| OpenXR full synthesis eyes | 否 | 是 | 最差 | 是 | 影院、静态图像，重质量模式 |

## 最终建议

保留 OpenXR rgb-depth 作为兼容旧版的传统模式。

增加或修复 OpenXR full stereo synthesis eyes，作为影院、游戏和静态图像模式的独立质量路径。不要在 OpenXR rgb-depth 中将纯 SBS 合成控件呈现为可用状态，除非它们确实被当前渲染路径所消费。
