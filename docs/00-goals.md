# 项目目标

## 核心目标

构建一套面向 4K 实时画面的高质量 stereo synthesis 实验核心库，验证它是否能在普通到高端 NVIDIA 显卡上超过简单 depth-shift 路线，并为 Desktop2Stereo / GUI / OpenXR host 提供稳定可接入的算法 API。

## 成功标准

| 目标 | 标准 |
|---|---|
| 4K 输入输出 | 支持 3840x2160 RGB 输入，输出 Half-SBS 与 Full-SBS |
| 入门硬件 | RTX 2060 12GB 可运行，用作最低 4K 验证基线 |
| 高端硬件 | RTX 3090 / RTX 5070 用于 Quality/HQ 模式目标测试 |
| 画质收益 | 在遮挡、边缘撕裂、空洞、前景漂浮和时序抖动上优于简单 depth-shift |
| 可集成性 | 外部 host 可通过稳定 preset/API 调用，不需要了解内部实现细节 |

## 当前主线

- `fast`：基础 depth-shift baseline。
- `quality_4k`：2-layer occlusion-aware synthesis，当前实时主线。
- `hq_4k`：3 层以上合成、局部修复和静态图高质量路线。
- `auto` preset：由外部 host 异步采集系统信号后切换到 cinema / game / still image 等模式。

## 非目标

- 本仓库不实现桌面捕捉、播放器捕捉或窗口管理。
- 本仓库不实现完整 GUI。
- 本仓库不实现完整 OpenXR session / swapchain runtime。
- 第一阶段不训练新的 stereo ML 模型。
- 性能优化不能牺牲 depth 推理分辨率和深度质量语义。

## 质量边界

任何优化都必须满足：

- 不降低 `294x518` / depth resolution 518 的当前模型输入路径。
- 不改变 RGB resize、antialias、ImageNet normalize 的语义，除非有单独画质评估。
- 不把模型产物写入 Desktop2Stereo 原项目目录。
- Half-SBS 与 Full-SBS 必须来自同一组左右眼，只允许输出打包方式不同。

## 最终锁定步骤

视觉回归基准放在最后执行。需要用固定输入分别生成并检查：

- `cinema`
- `game_low_latency`
- `still_image_hq`
- `debug_export`

这些结果用于钉住默认参数，防止后续算法或性能优化造成画质退化。
