# 算法路线概览

## 当前路线

`Desktop2Stereo` 当前主路径属于 depth-driven shift：

1. 从 RGB 推理深度图。
2. 基于 `Depth Strength / Convergence / IPD` 计算像素位移。
3. 用 shader 或 tensor path 输出 SBS。

优点是快，适合实时。缺点是遮挡、边缘、空洞和前景漂浮处理能力有限。

## iw3 路线

`iw3 row_flow_v3_sym` 属于 ML backward warp：

1. 用深度、divergence、convergence 构造输入。
2. 模型预测 warp delta。
3. 左眼用 `delta`，右眼用 `-delta`，形成对称生成。

优点是比朴素 depth shift 更自然，也更快于普通 `row_flow_v3`。缺点是仍依赖单层深度和训练分布。

## 超越路线

本仓库优先验证 layered / occlusion-aware synthesis：

1. 从深度图构造 2-4 个 depth layers。
2. 每层独立计算左右眼 warp。
3. 显式生成 occlusion mask。
4. 对遮挡区和边缘做局部 hole fill。
5. 加入左右眼对称约束和时序稳定。

这条路线比直接照搬 `row_flow_v3_sym` 更适合 4K 实时工程化，因为它可以按层数和修复强度做性能分档。

