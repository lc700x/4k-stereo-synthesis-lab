# 4K Stereo Synthesis Lab

面向 `Desktop2Stereo` 的 4K 实时立体生成算法实验仓库。

本仓库目标不是直接替代主项目，而是独立验证比 `iw3 row_flow_v3_sym` 更适合 4K 实时桌面/游戏/视频的 stereo synthesis 路线。

## Goals

- 以 4K 输入输出为核心目标，而不是只满足 1080p。
- 以 RTX 2060 12GB 作为最低可运行基线。
- 以 RTX 3090 / RTX 5070 等高端卡作为 Quality/HQ 模式目标硬件。
- 优先探索 layered / occlusion-aware / symmetric stereo synthesis。
- 与 `iw3 row_flow_v3_sym` 和当前 `Desktop2Stereo` depth-shift 路线做可复现实验对比。

## First Milestone

第一阶段只做工程可验证原型：

- `Fast 4K`: 当前 depth-shift 基线复现。
- `Quality 4K`: 2-layer occlusion-aware synthesis。
- `HQ 4K`: 3-4 layer synthesis + 更强 hole fill。

详细计划见：

- [Implementation Plan](docs/04-implementation-plan.md)
- [4K Performance Budget](docs/02-4k-performance-budget.md)
- [iw3 Comparison](docs/03-iw3-comparison.md)

