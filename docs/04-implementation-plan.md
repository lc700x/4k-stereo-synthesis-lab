# 高端卡 4K 超越 iw3 实施计划书

## 摘要

本计划目标是实现一条独立的 `Quality 4K` stereo synthesis 实验链路，在 4K 输入输出下，比当前 `Desktop2Stereo` depth-shift 路线和 `iw3 row_flow_v3_sym` 更好地处理遮挡、边缘、空洞、前景漂浮和时序稳定。

第一阶段不训练大模型，不直接改主项目。先在本仓库实现可复现 benchmark 和算法原型，再决定是否接回 `Desktop2Stereo`。

## 阶段 1: Baseline 与评测框架

目标：

- 复现当前 `Desktop2Stereo` depth-shift 合成逻辑。
- 支持 RGB + depth 输入，输出左右眼和 SBS。
- 建立 4K 性能计时、显存记录、图像输出和对比脚本。

交付：

- `baseline_shift` backend。
- `bench_4k.py` 性能脚本。
- `compare_methods.py` 对比脚本。
- 第一版测试场景集说明。

验收：

- 能对单帧 4K RGB + depth 输出 Half-SBS。
- 能记录每阶段耗时和峰值显存。
- 能输出与当前 `Desktop2Stereo` 接近的 baseline 结果。

## 阶段 2: Quality 4K 原型

目标：

- 实现 2-layer depth synthesis。
- 加入 occlusion mask。
- 加入左右眼对称约束。
- 加入轻量 edge-aware hole fill。

算法默认：

- 深度图按 foreground/background 分成 2 层。
- 每层独立生成左右 eye warp。
- 遮挡区由 forward/backward consistency 和 depth discontinuity 共同标记。
- hole fill 只作用在 mask 区域，避免全帧高成本修复。

验收：

- 在复杂边缘场景中，明显少于 baseline 的撕裂和空洞。
- 在 RTX 3090 / RTX 5070 档位目标为 4K 60-90 FPS。
- 在 RTX 2060 12GB 上至少可用，允许降级到 30-60 FPS。

## 阶段 3: HQ 4K 原型

目标：

- 扩展到 3-4 layer synthesis。
- 加强 hole fill 和 temporal stabilization。
- 增加可选局部 refinement，但不做全帧大模型。

算法默认：

- 层数可配置为 2 / 3 / 4。
- 只对 occlusion 和 high-gradient depth 区域做增强修复。
- 时序稳定使用前帧 depth/mask/warp 的轻量缓存。

验收：

- 比 Quality 4K 在遮挡复杂场景更稳定。
- RTX 3090 目标 4K 30-60 FPS。
- RTX 5070 可运行，但允许低于 60 FPS。

## 阶段 4: Desktop2Stereo 集成评估

目标：

- 将实验 backend 设计成后续可接入 `Desktop2Stereo` 的接口。
- 明确 `Fast / Quality / HQ` 三档配置。
- 输出接入主项目的最小改动方案。

建议接口：

```python
def synthesize_stereo(rgb, depth, config):
    """Return left_eye, right_eye, sbs, debug_info."""
```

配置项：

| 参数 | 默认 | 说明 |
|---|---:|---|
| `backend` | `quality_4k` | `fast`, `quality_4k`, `hq_4k` |
| `layers` | `2` | Quality 默认 2，HQ 可 3-4 |
| `occlusion` | `true` | 是否启用遮挡 mask |
| `symmetric` | `true` | 是否强制左右对称 |
| `hole_fill` | `edge_aware` | `none`, `fast`, `edge_aware` |
| `temporal` | `true` | 是否启用时序稳定 |

## 风险与降级

| 风险 | 降级策略 |
|---|---|
| 4K 下新增耗时过高 | 从 4 层降到 2 层，关闭 HQ fill |
| 2060 显存压力 | 只启用 Fast 或 2-layer lightweight |
| 边缘修复引入糊边 | fill 仅限 mask 区域，保留原图边缘 |
| 时序缓存导致拖影 | 提供 temporal strength 和快速关闭开关 |
| 与主项目 viewer 路径不匹配 | 先输出 SBS 文件/张量，再做集成 |

## 测试计划

- 单帧 correctness：输入 RGB + depth，检查左右眼尺寸、范围、SBS 布局。
- 性能 benchmark：4K 单帧、多帧循环、显存峰值。
- 画质对比：baseline、iw3、Quality 4K 同场景输出。
- 退化测试：低显存、低层数、关闭 occlusion、关闭 temporal。
- 时序测试：横向移动、窗口拖动、游戏镜头移动。

## 默认假设

- 4K 指 3840x2160 输入输出。
- 深度推理分辨率以 518 为主，不追求 4K 深度模型直接推理。
- RTX 2060 12GB 是最低可运行线，不保证 HQ。
- RTX 3090 / RTX 5070 是 Quality/HQ 主要目标。
- 第一阶段不训练模型，只做几何和局部修复算法。

