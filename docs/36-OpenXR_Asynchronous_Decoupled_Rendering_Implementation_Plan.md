# OpenXR 异步解耦渲染重构实施计划书

## 1. 目标与边界

本文以 `docs/35-OpenXR_Asynchronous_Decoupled_Rendering_Architecture_Report.md` 为技术目标，面向 Desktop2Stereo 当前工程制定 OpenXR 异步重构实施计划。

核心目标不是在现有 OpenXR 路径上做局部补丁，而是把 OpenXR 输出重构为“显示器硬实时 + 背景/光效软实时”的解耦架构：

- 虚拟显示器必须稳定跟随目标刷新率，优先消费最新 runtime 画面。
- 房间背景、Glow、墙面反射不得阻塞显示器更新和 `xrEndFrame` 提交。
- 所有高成本效果只消费已经完成的旧结果，不等待当前帧结果。
- OpenXR submit/present、viewer upload、runtime inference、capture 必须分段统计，不混淆性能归因。
- 现有实现只作为迁移参考，不以当前限制倒推目标架构。

非目标：

- 不在第一阶段重写 depth/stereo synthesis 算法。
- 不把本地窗口、RTMP、OpenXR 三种输出合并为一个 frame loop。
- 不把 OpenXR Quad Layer 作为可选装饰功能，而是作为虚拟显示器分层的主路径目标。
- OpenXR view pose 必须按 runtime 提供的真实位置与朝向参与背景、交互和层合成。

## 2. 当前工程锚点

计划实施时主要涉及以下模块：

| 领域 | 现有模块 | 重构定位 |
|---|---|---|
| OpenXR viewer | `src/xr_viewer/*` | OpenXR frame loop、projection layer、quad layer、环境渲染、controller/overlay |
| OpenXR Quad 层 | `src/xr_viewer/core_quad_layer.py` | 改为显示器硬实时主路径，而不是调试/候选路径 |
| OpenXR runtime eye 上传 | `src/xr_viewer/core_runtime_eye.py` | 提供最新显示器纹理，后续接入独立 screen swapchain |
| OpenXR 环境 | `src/xr_viewer/environment_renderer.py`, `environment_model.py`, `environment_profiles.py` | 由每帧复杂房间渲染迁移到 panorama/cubemap 背景路径 |
| Glow/屏幕光 | `src/xr_viewer/core_screen_quality.py`, `src/xr_viewer/environment_effects.py`, `src/xr_viewer/environment_renderer.py`, `docs/20-openxr-gpu-glow-guide.md` | 以现有 GPU glow/downsample/shader 采样为基础，迁移到异步效果结果池；禁止 CPU 采样 |
| Runtime pipeline | `src/stereo_runtime/pipeline.py`, `runtime.py` | 继续产出 latest runtime result，不被 viewer present 反压拖慢 |
| OpenXR 状态 | `src/stereo_runtime/openxr_state.py` | 维护 render/source gate 和 hot config，不承载显示层同步细节 |
| 队列语义 | `src/utils/queue_utils.py` | 保留 latest/overwrite/drop 模型，扩展统计而不是改成阻塞队列 |
| 诊断 | `src/utils/breakdown.py`, runtime debug info | 新增 screen/background/effects/submit/present 分段指标 |

## 3. 目标架构

### 3.1 线程与职责

目标运行时拆成四条职责明确的路径：

| 路径 | 实时等级 | 职责 | 阻塞策略 |
|---|---:|---|---|
| Capture + StereoRuntime | 生产者 | 捕获、depth、stereo synthesis 或 RGB+D | 只写 latest result，队列满则覆盖旧帧 |
| OpenXR Screen Presenter | 硬实时 | 更新虚拟显示器 Quad 层并提交 OpenXR frame | 不等待背景和效果；拿不到新帧则复用上一帧 |
| Background Renderer | 软实时 | 加载或更新外部导出的 panorama/cubemap/projection layer | 可低频、可延迟、可暂停 |
| Effects Worker | 软实时 | Glow、屏幕平均色、墙面反射纹理 | 只发布 completed result；主线程消费旧 safe result |

### 3.2 OpenXR 层结构

目标 `xrEndFrame` 提交层顺序：

1. 背景层：优先 equirect/cubemap composition layer；不支持时退化为 projection layer 内天空球。
2. 环境/overlay projection layer：只包含必要 3D UI、controller、laser、OSD，不再承载显示器主体。
3. 虚拟显示器 Quad layer：每帧更新，始终在前景，无遮挡。
4. 可选辅助 Quad/overlay layer：调试 OSD、键盘、面板等，按交互需求叠加。

显示器画面不再依赖房间 depth test，因此房间复杂度不能影响屏幕刷新。

### 3.3 数据流

```text
Capture
  -> RuntimePipelineLoop
  -> latest OpenXRRuntimeResult
  -> ScreenFrameBridge
  -> ScreenSwapchain/QuadLayer
  -> xrEndFrame

ScreenFrameBridge
  -> AsyncEffectsScheduler
  -> glow_result_pool / light_probe_pool / wall_reflection_pool
  -> Background/Projection shader consumes latest safe result

External Unity/Blender bake or packaged panorama asset
  -> environment profile
  -> mono/SBS HDR panorama or cubemap cache
  -> Background layer or sky sphere
```

关键约束：

- `ScreenFrameBridge` 可以覆盖旧帧，但不得阻塞 runtime producer。
- `AsyncEffectsScheduler` 不向主 OpenXR frame loop 返回 future/promise 等待点。
- 主 OpenXR frame loop 只读 `safe_index`，不等待 GPU fence。
- 初始帧所有 effect result 使用黑色/透明默认纹理。

## 4. 分阶段实施路线

### 阶段 0：基线和诊断

目标：先建立可对比、可归因的诊断框架，不再以旧 projection screen 路径作为普通运行目标。

任务：

- 异步 present 是固定主路径；运行参数只允许限制预算或关闭软效果，不允许把旧同步显示器路径作为常规回退目标：
  - `D2S_OPENXR_SCREEN_UPLOAD_BUDGET_MS`：screen texture 上传预算，超预算下一帧复用上一张 screen texture。
  - `D2S_OPENXR_EFFECT_SUBMIT_BUDGET_MS`：effect source submit + downsample prewarm 预算，超预算下一帧复用 latest safe effect texture。
  - `D2S_OPENXR_BACKGROUND_UPLOAD_BUDGET_MS`：native panorama/cubemap 背景层上传预算，超预算下一帧复用旧背景或 projection fallback。
- GUI/OpenXR config 不再暴露“回到旧显示器路径”的普通开关。
- 在 `FPSBreakdown` 中新增指标：
  - `openxr_upload`
  - `openxr_quad_update_ms`
  - `openxr_background_ms`
  - `openxr_effect_submit_ms`
  - `openxr_effect_ready_age_frames`
  - `openxr_screen_frame_age_frames`
  - `openxr_wait_frame`
  - `openxr_end_frame`
  - `openxr_layer_count`
- 记录每帧使用的是 `new_screen_frame` 还是 `reused_screen_frame`。

验收：

- 默认启动即进入异步 screen/Quad 主路径；预算参数只影响复用策略，不改变架构路径。
- 打开诊断后能分清 runtime FPS、viewer FPS、screen present FPS、submit/present 节奏。

### 阶段 1：ScreenFrameBridge 与 Quad Layer 主路径

目标：把虚拟显示器从 projection 内 3D quad 迁移为 OpenXR Quad Layer 主路径。

任务：

- 新增 `ScreenFrameBridge`：
  - 从 `runtime_q` 非阻塞 drain latest。
  - 维护 `latest_frame`, `last_presented_frame`, `frame_id`, `source_timestamp`。
  - 未收到新 runtime result 时复用上一帧，不阻塞 `xrEndFrame`。
- 重构 `core_quad_layer.py`：
  - Quad 层作为显示器唯一主路径；projection 不再绘制显示器主体。
  - Quad 层支持 stereo eye source、mono fallback 和 full synthesis eye source。
  - Quad layer pose/size 由现有 screen placement 配置驱动。
- 解耦 controller hit/UI：
  - controller raycast 仍使用同一个 screen plane 几何模型。
  - 交互命中不依赖 projection path 是否绘制屏幕。
  - 屏幕边框/焦点/laser hit 可保留在 projection layer 或独立 overlay layer。
- projection layer 只承载背景 fallback、controller、laser、OSD、边框等辅助内容，不承载显示器主体。

验收：

- 房间 mesh depth 不遮挡显示器。
- 同一 screen pose 下，鼠标/laser 命中区域与视觉 Quad 一致。
- 环境模型复杂度提高时，显示器 present FPS 不下降到环境 FPS。
- Quad layer 不可用时进入明确失败/空屏状态并记录原因，不静默回到旧 projection screen 主体。

### 阶段 2：OpenXR frame loop 硬实时化

目标：主 OpenXR frame loop 不等待 runtime、背景、效果。

任务：

- 将 OpenXR frame loop 的每帧流程固定为：
  1. `xrWaitFrame/xrBeginFrame`
  2. 非阻塞读取最新 screen frame
  3. 更新 screen quad swapchain 或复用上一张
  4. 采样 latest safe background/effects result
  5. 构造 layers
  6. `xrEndFrame`
- 对 runtime result 的处理拆分：
  - screen upload 必须有预算上限。
  - 超预算时可跳过本帧 upload，复用上一帧 screen texture。
  - effect source submit 使用独立预算；超预算时只能跳过 effect submit，不能影响 screen upload 或 `xrEndFrame`。
- 当前落地：screen upload 和 effect submit 已拆成独立预算和独立计时，effect source submit 不再计入 `openxr_upload` 或 `openxr_submit_frame`，并延后到 `xrEndFrame` 后 flush；`screen_age` / `fx_age` 使用帧数 value 统计，不再伪装成 time metric。
- 当前落地：effect source pending 为 latest-only 单槽；flush 跟不上时覆盖旧 pending 并记录 `openxr_effect_submit_overwrite` / `fx_overwrite`。
- 避免在 present 路径中执行：
  - 每帧 CPU tensor readback
  - 每帧模型/环境资源加载
  - 当前帧 Glow 生成
  - 当前帧墙面反射计算
- 保留 latest/overwrite 队列模型，不引入阻塞 backpressure。

验收：

- 人为让 runtime 降到 30 FPS 时，OpenXR frame loop 仍按头显节奏提交，屏幕复用旧帧但头动/层合成不阻塞。
- 人为让环境渲染耗时升高时，screen quad 仍可按最新可用帧刷新。
- 日志能显示 reused frame 比例、effect age 和 swapchain image wait 耗时。

### 阶段 3：静态房间背景 panorama/cubemap

目标：把复杂房间背景从每帧 mesh 渲染迁移为外部工具预生成的 panorama/cubemap 背景资产。项目不再实现从 GLB 房间自动 bake 出 panorama/cubemap；GLB 仍可作为预览/兼容输入，但目标背景资产由 Unity、Blender 或其它离线工具导出。

任务：

- 支持外部导出的背景资产：
  - mono equirectangular HDR/LDR panorama。
  - SBS equirectangular HDR panorama，左右眼按 profile 声明的 stereo layout 分区采样。
  - 后续可选 cubemap/cubemap array 输入，但不要求项目内从 GLB 生成。
- `BackgroundBakeService` 只保留轻量辅助职责：
  - 生成/缓存 wall light mask、profile 派生资源等低成本资产。
  - 不负责 GLB PBR 渲染、不负责 cubemap/equirect panorama bake。
- 新增 `BackgroundLayerRenderer`：
  - 首选 OpenXR equirect/cubemap layer（若 runtime 支持）。
  - fallback 为 projection layer 内 sky sphere shader。
- 将动态 controller、laser、overlay 与静态背景分离。
- 当前落地：panorama shader 支持 `stereo_layout: "sbs"`，左右眼分别采样左/右半幅；`.hdr` panorama 优先上传为半浮点纹理，controller panorama IBL 也按同一 eye 分区采样。

验收：

- 外部导出的 mono/SBS panorama 背景按 OpenXR view pose 正确采样和合成，无明显方向错误。
- SBS panorama 在左右眼采样正确，不出现左右眼串图或半幅拉伸。
- environment profile 改变后可重新加载对应 panorama/cubemap 和 mask，并热切换。
- GLB 自动 bake 不是验收项；复杂 GLB 房间若未提供 panorama 资产，只走兼容/预览路径，不作为最终异步背景目标。

### 阶段 4：异步 GPU Glow 结果池

目标：在 `docs/20-openxr-gpu-glow-guide.md` 已实现的 GPU glow 技术基础上，把屏幕光采样、downsample、blur 和混合迁移为可延迟消费的 GPU 结果池。这里不是重新引入 CPU 采样；屏幕颜色来源必须来自 GL texture、低分辨率 glow texture、shader/compute pass 或后续 D3D/Vulkan GPU pass。

任务：

- 新增 `AsyncEffectResultPool`：
  - 至少 triple buffer。
  - 每个 slot 状态：`idle`, `writing`, `ready`, `safe`。
  - 主线程只读 `safe_slot`。
- 新增 `GlowWorker` / `GpuGlowScheduler`：
  - 复用现有 `_prepare_glow_downsample_texture()`、glow shader、frosted/surround shader 的 GPU 采样思路。
  - 输入必须是 screen/runtime eye GL texture、shared GPU texture 或低分辨率 GPU 中间纹理。
  - 降采样、多 pass blur、区域采样、边缘采样、高亮提取都在 GPU 内完成。
  - 完成后只更新 ready/safe index，不唤醒主线程等待。
  - 禁止通过 `.cpu()`、`.numpy()`、`glReadPixels()`、`tex.read()` 等 CPU readback 做实时屏幕采样。
- 更新 environment/screen shader：
  - 继续沿用 `20-openxr-gpu-glow-guide.md` 的低分辨率 GPU glow texture、`textureLod()`、边缘采样、区域化采样和 frosted/surround shader 方案。
  - 使用 `safe_glow_texture` 叠加 screen glow / surround glow / frosted glow。
  - 初始或 worker 失败时使用透明纹理。
- 先以 OpenGL FBO/低频 pass 实现逻辑正确性；后续按平台升级到 D3D11 compute、D3D12/Vulkan async compute queue。
- 当前落地：runtime effect source 已抽出轻量 result pool，具备 staging/ready/safe/spare texture 语义和 safe publish 诊断；ready 被 latest-only 新结果覆盖时旧 ready 会转为 spare 复用或释放，避免异步 submit 跟不上时泄漏纹理。
- 当前落地：`EffectWorker` 已从 submitter/source state 中拆出，`xrEndFrame` 后按 `D2S_OPENXR_EFFECT_WORKER_INTERVAL` 低频预热 glow/light downsample，并只向 scheduler 发布 safe downsample；panorama/controller/glow 消费者只读取 safe result，不在 render 消费路径生成 downsample。
- 当前落地：GPU glow/downsample 复用已有 cache key，新增 `fx_ds_render` / `fx_ds_reuse` 诊断，用于确认 soft effect 是否复用旧结果；no-room 与 environment glow pass 都隔离异常并恢复 GL 状态，effect 失败不污染 screen submit 后续帧。
- 当前落地：screen light source 同帧缓存已接入，环境光、panorama、controller 反射等多消费者复用同一个 prepared light texture，并记录 `light_reuse`。
- 当前落地：screen effect source 同帧缓存已接入，Glow/veil/shell 多 pass 复用同一个 safe source lookup，并记录 `fx_source_reuse`。
- 当前落地：`fx_age` 按每帧每个 safe effect texture 去重记录，避免 Glow、screen light、panorama 等多消费者重复放大 age 样本。

验收：

- Glow worker 降频到 30/45 Hz 时，OpenXR screen present 不受影响。
- Glow 纹理 age 可见但不卡顿。
- GPU Glow Off fast path 不触发 downsample，不触发 CPU 采样。
- worker 故障只关闭 Glow，不影响 screen quad 和 OpenXR submit。

### 阶段 5：墙面反射和屏幕光斑

目标：实现 `35` 中的“预计算掩码 + 动态光斑颜色”路径。

任务：

- 为 environment profile 增加可选资源：
  - `wall_light_mask`：与 panorama UV 对齐的单通道或 RGB 权重纹理。
  - `screen_light_layout`：屏幕到墙面的投影区域描述。
- 新增 `LightProbeWorker` / `GpuLightProbeScheduler`：
  - 从 screen/runtime eye GPU texture 生成 1x1、3x3、8x5 或 16x9 低频区域色。
  - 必须复用 GPU Glow 降采样链或专用 GPU pass，不允许 CPU 采样。
  - 输出 `screen_light_color_grid` 或低分辨率 GPU light texture。
- background shader 合成：
  - `final = panorama + mask * delayed_screen_light_color * intensity`
  - 所有输入都来自 safe result。
- 对简单房间提供数学 mask fallback；对复杂房间使用外部工具或 profile 提供的 `wall_light_mask`，项目内只提供 `wall_light_mask: "auto"` 的基础 UV mask 生成。
- 当前落地：panorama shader 已可用数学 mask fallback 或 profile `wall_light_mask` 图片消费 latest safe screen texture，并使用 3x3 GPU light probe 采样低频屏幕光；复杂房间 mask 由外部工具或 profile 资产提供，项目内不实现复杂房间 bake。
- 当前落地：`BackgroundBakeService` 已支持 profile `wall_light_mask: "auto"`，在配置期生成并缓存与 panorama UV 对齐的灰度 mask PNG；实时渲染路径只加载缓存纹理，不做 CPU 屏幕采样。它不是 GLB panorama/cubemap bake 服务。

验收：

- 屏幕画面颜色变化能延迟一两帧反映到墙面，不影响 screen FPS。
- 没有 mask 时自动退化为现有 glow-only 视觉。
- mask/profile 资源缺失有明确日志，不报错退出。

### 阶段 6：GPU 后端升级路径

目标：在不打断 Python/OpenXR 主流程的情况下，引入真正多队列能力。

分层策略：

1. OpenGL baseline：FBO + low-frequency pass + no-wait safe result，先验证架构语义。
2. D3D11 native：复用现有 D3D11 backend，减少 GL/D3D 互操作成本，支持更稳定的 Windows OpenXR 路径。
3. D3D12/Vulkan 后端：为 async compute/copy queue 建立接口，不把具体 API 绑死在 viewer 业务代码里。

新增接口建议：

```text
EffectScheduler
  submit_screen_frame(frame_handle, frame_id)
  poll_completed()
  latest_safe_glow()
  latest_safe_light_probe()
  release()

ScreenLayerPresenter
  update_or_reuse(frame)
  make_quad_layers(predicted_display_time)
  release()

BackgroundPresenter
  make_background_layer_or_projection()
  update_safe_effect_inputs(effect_state)
  release()
```

验收：

- 后端替换不改变 OpenXR 主 loop 的 no-wait 规则。
- GL/D3D11/D3D12/Vulkan 任一后端失败，都能回退到透明 effect + projection fallback。
- 旧 EXT memory / `render.py` / `d3d11_backend.py` 路径不再参与 OpenXR 热路径；D3D11 projection 不回退 PBO readback，runtime texture upload 的 PBO 仅作为明确告警的 GPU fallback。

## 5. 关键设计约束

### 5.1 主线程禁止等待 effect

禁止在 OpenXR frame loop 中出现以下等待：

- 等待 Glow 完成。
- 等待墙面反射完成。
- 等待外部 panorama/cubemap 资产或派生 mask 准备完成。
- 等待当前 runtime result 必须可用。

可接受行为：

- 复用上一帧 screen texture。
- 使用黑色/透明 effect texture。
- 使用旧 panorama。
- 暂停背景更新但继续提交 screen quad。

### 5.2 Runtime 与 viewer 只通过 latest result 连接

`RuntimePipelineLoop` 不应被 OpenXR present 速度反压。OpenXR viewer 也不应等待 runtime。两边通过 latest result 桥接：

- producer 快：覆盖旧 result，统计 overwrite/drop。
- producer 慢：viewer 复用旧 screen texture，统计 reuse。
- producer 停：viewer 保持 last good frame 或进入明确 wait/idle 状态。

### 5.3 屏幕交互与屏幕显示分离

显示器视觉进入 Quad layer 后，交互仍需要一个逻辑 screen plane：

- raycast 使用 screen pose/size。
- UI hover/click 使用 logical UV。
- projection layer 可绘制边框、hover highlight、controller laser。
- 不允许因为 Quad layer 不参与 depth test 而丢失输入命中。

### 5.4 主路径与降级边界

| 功能 | 主路径 | 降级边界 |
|---|---|---|
| 显示器 | OpenXR Quad layer | 明确失败/空屏并记录原因，不回到 projection screen 主体 |
| 背景 | equirect/cubemap layer | projection sky sphere |
| Glow | async safe texture | transparent/no glow |
| 墙面反射 | mask + delayed light texture | disabled |
| GPU 后端 | D3D11/D3D12/Vulkan async | OpenGL low-frequency pass |

## 6. 测试计划

### 6.1 单元测试

- `ScreenFrameBridge`：latest drain、reuse、frame id、timestamp、drop 统计。
- `AsyncEffectResultPool`：slot 状态转换、safe index 不读 writing slot。
- GPU Glow 采样约束：测试文档/代码不得出现实时 `.cpu()`、`.numpy()`、`glReadPixels()`、`tex.read()` 作为屏幕光颜色来源。
- `QuadLayerPresenter`：layer 构造、pose/size、失败原因。
- `BackgroundBakeService`：wall light mask cache key、profile invalidation、missing asset fallback；不测试 GLB panorama/cubemap bake。

### 6.2 集成测试

- OpenXR 无 headset preview 模式不崩溃。
- Quad layer 创建失败时不绘制 projection screen 主体，并记录明确失败原因。
- runtime producer 低 FPS 时 viewer submit 不阻塞。
- effect worker 人为 sleep 时 screen FPS 不下降。
- GPU Glow Off fast path 不触发 downsample，不触发 CPU 采样。
- 外部 panorama 背景或复杂 GLB 兼容预览路径启用时，screen quad 刷新不被背景路径拖慢。

### 6.3 性能验收

至少记录：

| 指标 | 目标 |
|---|---:|
| screen present FPS | 接近头显目标刷新率 |
| runtime FPS | 独立记录，不作为 present FPS |
| screen frame reuse rate | 可解释、可接受 |
| effect age | 通常 1-3 帧 |
| OpenXR submit/present stall | 明确归因到 runtime/headset，不归因到 StereoRuntime |
| CPU fallback count | OpenXR 实时路径必须红色告警；屏幕光/Glow 实时采样不得 CPU fallback |

## 7. 推荐提交顺序

1. `feat(openxr): add async rendering flags and diagnostics`
2. `feat(openxr): introduce screen frame bridge`
3. `feat(openxr): promote quad layer screen presenter`
4. `feat(openxr): decouple frame loop from runtime producer`
5. `feat(openxr): add panorama background presenter`
6. `feat(openxr): add async glow result pool`
7. `feat(openxr): add wall reflection light probe path`
8. `refactor(openxr): split backend effect scheduler interfaces`
9. `docs(openxr): update runtime architecture and degradation matrix`

每个提交都必须保持 OpenXR 可诊断、可启动；显示器主路径不再以旧 projection screen 作为兼容目标。

## 8. 风险与处理

| 风险 | 影响 | 处理 |
|---|---|---|
| Quad layer 与 projection screen 视觉/交互不一致 | 点击位置偏移 | screen pose/size 单一来源，raycast 与 layer 共用参数 |
| OpenXR runtime 对 Quad/equirect 支持差异 | 某些设备黑屏或层失败 | 能力探测 + 明确失败原因；背景可 projection fallback，显示器不回退到 projection screen 主体 |
| Python/OpenGL 难以实现真正 GPU 多队列 | 无法达到最终并行度 | 先实现 no-wait 语义，再引入 D3D11/D3D12/Vulkan 后端 |
| effect 结果读写冲突 | 闪烁或 GPU hazard | triple buffer + safe index，不读 writing slot |
| 诊断口径混乱 | 性能问题误判 | 强制分开 capture/runtime/viewer/screen/effect/submit/present |

## 9. 第一轮落地建议

第一轮不要同时做 panorama、Glow、墙面反射。建议先完成硬实时显示器链路：

1. 加 flags 和分段统计。
2. 做 `ScreenFrameBridge`。
3. 启用 Quad layer screen presenter。
4. 保持背景仍走现有 projection/environment 渲染。
5. 保持 Glow Off / GPU Glow fast path，不引入 CPU 屏幕采样。
6. 验证复杂环境下 screen FPS 与环境成本解耦。

只有当“显示器层不被环境拖慢”成立后，再进入异步 Glow 和 panorama 背景，否则会把多个问题混在一起，无法判断收益来源。

## 10. 完成定义

重构完成的判定不是“代码里有 Quad layer 或 async worker”，而是满足以下行为：

- 显示器画面作为 OpenXR 前景层稳定提交。
- 背景复杂度变化不改变显示器硬实时路径。
- Glow 和墙面反射都可延迟、可降频、可失败关闭，但屏幕采样必须走 GPU glow/downsample/shader 路径，不能阻塞 screen submit，不能回退到 CPU 实时采样。
- Runtime producer 与 OpenXR presenter 之间没有阻塞等待。
- 日志能清楚解释每一帧的 runtime、viewer、screen、effect、submit/present 状态。
