# 4K 性能预算

## 硬件档位

| 档位 | 代表显卡 | 目标 |
|---|---|---|
| Minimum | RTX 2060 12GB | 4K 可运行，Fast/轻量 Quality |
| Quality | RTX 3090 / RTX 5070 | 4K 60-90 FPS Quality |
| HQ | RTX 3090+ / 更高端卡 | 4K 30-60 FPS HQ |

## 每帧预算

| 目标 FPS | 每帧总预算 |
|---:|---:|
| 90 FPS | 11.1 ms |
| 60 FPS | 16.7 ms |
| 30 FPS | 33.3 ms |

## 新增 stereo synthesis 预算

| 模式 | 新增预算 | 说明 |
|---|---:|---|
| Fast 4K | 0-2 ms | 当前路径 + 轻量对称和时序 |
| Quality 4K | 2-6 ms | 2-layer + occlusion + edge fill |
| HQ 4K | 6-12 ms | 3-4 layer + 更强局部修复 |
| Experimental | 12 ms+ | 小型 learned refinement 或离线验证 |

## 原则

- 4K 主画面不做全帧大模型 refine。
- 深度推理仍可保持 518 级别输入，再映射到 4K 合成。
- 所有新增算法必须支持性能降级开关。
- 先以 CUDA tensor path 原型验证，再考虑 shader 化或 TensorRT 化。

