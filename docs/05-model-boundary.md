# 模型边界

## 当前实现使用哪个模型

当前实现不使用任何新的 ML 模型。

它不加载：

- `iw3 row_flow_v3_sym`
- `iw3 mlbw_l2 / mlbw_l4`
- `Distill-Any-Depth-Base`
- `Distill-Any-Depth-Large`
- 任何 learned refinement net

当前实现只接收外部已经生成好的 `RGB + depth`，然后验证 stereo synthesis 算法。

## 推荐评估基线

为了避免误判，评估时必须固定 depth 来源。

| 组别 | Depth 来源 | 用途 |
|---|---|---|
| A | `Distill-Any-Depth-Base @ 518` | 当前 Desktop2Stereo 默认级别基线 |
| B | `Distill-Any-Depth-Large @ 518` | 高端卡目标基线 |
| C | 同一份 depth 输入 + 不同 synthesis backend | 只比较左右眼生成算法 |

## 为什么这样拆

如果同时更换深度模型和 stereo synthesis 算法，就无法判断画质提升来自哪里。

第一阶段只回答一个问题：

> 在同一份 depth 输入下，`quality_4k / hq_4k` 是否比当前 depth-shift 更好？

只有这个问题成立后，才进入第二阶段：

> 在 `Distill-Any-Depth-Large @ 518` 上，高端卡是否能以 4K 实时方式稳定运行更高质量 synthesis？

