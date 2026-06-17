# 与 iw3 的对比基准

## 对比对象

| 方法 | 类型 | 用途 |
|---|---|---|
| Desktop2Stereo current | depth-driven shift | 当前项目基线 |
| iw3 row_flow_v3_sym | ML symmetric backward warp | 主要外部对比 |
| iw3 mlbw_l2 / mlbw_l4 | multi-layer backward warp | 高质量参考 |
| proposed Quality 4K | layered occlusion-aware synthesis | 本项目主目标 |

## 对比维度

| 维度 | 观察点 |
|---|---|
| 边缘质量 | 前景边缘是否撕裂、拉丝、重复 |
| 遮挡关系 | 新显露区域是否自然 |
| 左右一致性 | 左右眼几何是否稳定 |
| 前景漂浮 | 物体是否脱离背景太假 |
| UI 舒适度 | 文字、窗口、桌面元素是否刺眼 |
| 时序稳定 | 快速移动时是否抖动 |
| 性能 | 4K 输入输出下 FPS、显存、延迟 |

## 场景集

第一阶段使用 5 类测试画面：

- 桌面 UI 和浏览器文本。
- 游戏场景，含角色和复杂背景。
- 视频人物前景。
- 细边缘物体，如栏杆、头发、树枝。
- 快速横向移动画面。

