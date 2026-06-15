# 项目目标

## 核心目标

构建一套面向 4K 实时画面的高质量 stereo synthesis 实验链路，验证是否能在普通到高端 NVIDIA 显卡上超过 `iw3 row_flow_v3_sym` 的实际观看体验。

## 成功标准

| 目标 | 标准 |
|---|---|
| 4K 输入输出 | 保持 3840x2160 画面输入输出路径 |
| 最低硬件 | RTX 2060 12GB 可运行 |
| 高端硬件 | RTX 3090 / RTX 5070 上达到 Quality 4K 实时体验 |
| 画质收益 | 边缘、遮挡、空洞、前景漂浮明显优于当前 depth-shift |
| 可集成性 | 后续能作为 `Desktop2Stereo` 的可选 stereo synthesis backend |

## 非目标

- 第一阶段不训练大型生成模型。
- 第一阶段不做 diffusion / transformer 级全帧生成。
- 第一阶段不直接修改 `Desktop2Stereo` 主项目。

