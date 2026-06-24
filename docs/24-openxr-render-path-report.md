# OpenXR Render Path Report

## Purpose

This report clarifies how the legacy Desktop2Stereo OpenXR path worked, how the current OpenXR paths differ, and how stereo modes should map to render paths after the stereo runtime rewrite.

The key product goal is:

- Traditional stereo mode should preserve the legacy OpenXR behavior.
- Cinema, game, and still-image stereo modes should use the new full `stereo_runtime.synthesize_stereo()` pipeline when OpenXR quality output is desired.

## Legacy Desktop2Stereo OpenXR Flow

The legacy `Desktop2Stereo_v2.5.0Beta` OpenXR path did not use the full `make_sbs()` stereo synthesis path.

Legacy OpenXR flow:

```text
capture
-> predict_depth(rgb)
-> depth_q.put(rgb, depth, timestamp)
-> OpenXRViewer.run(first_rgb, first_depth)
-> viewer uploads RGB texture + depth texture
-> OpenXR shader generates per-eye parallax from RGB + depth
```

Relevant legacy behavior:

- `main.py` OpenXR branch creates `OpenXRViewer(ipd=IPD, depth_ratio=DEPTH_STRENGTH, ...)`.
- It passes `rgb` and `depth` directly into `viewer.run()`.
- The OpenXR viewer uploads RGB and depth in `_update_frame(rgb, depth)`.
- The viewer shader uses `depth_strength * depth_ratio` to create stereo parallax.
- The legacy `make_sbs(...)` path is used by legacy streaming / non-OpenXR output, not by OpenXR.

Therefore, legacy OpenXR is best described as an RGB+depth shader path, not a full SBS synthesis path.

## Current Render Path Concepts

The current codebase has three distinct concepts that must not be treated as equivalent:

1. OpenXR rgb-depth
2. OpenXR prewarp eyes
3. OpenXR full stereo synthesis eyes, not fully wired yet

### 1. OpenXR rgb-depth

This is the current default low-latency OpenXR path.

```text
runtime:
RGB -> depth model -> depth postprocess

viewer:
RGB + depth -> OpenXR shader -> headset
```

Parameters consumed by this path:

- `depth_strength`
- `convergence`
- `ipd` / `ipd_mm`
- `stereo_scale`
- `max_shift_ratio`
- `foreground_scale`
- `depth_antialias_strength`

Parameters not consumed by this path:

- `temporal_strength`
- `edge_threshold`
- `edge_dilation`
- `mask_feather_radius`
- `hole_fill_mode`
- `hole_fill_radius`
- `hole_fill_strength`

Advantages:

- Lowest latency among the OpenXR paths.
- Controller and GUI changes to core depth parameters are lightweight.
- Closest to the legacy OpenXR behavior.
- Good for interactive tuning and real-time headset use.

Disadvantages:

- Does not use the full stereo synthesis pipeline.
- Cinema, game, and still-image presets are only partially meaningful.
- Hole fill, edge dilation, mask feather, and stereo temporal blend do not affect headset output.

Performance impact:

- Lowest additional cost after depth inference.
- Most cost is depth model inference plus a relatively cheap viewer shader.

Correct use:

- Traditional OpenXR mode.
- Legacy behavior compatibility.
- Low-latency usage.
- Realtime controller depth adjustment.

### 2. OpenXR prewarp eyes

This path generates left and right eye images in the runtime before passing them to the viewer.

```text
runtime:
RGB + depth -> render_openxr_stereo() -> left_eye + right_eye

viewer:
upload left_eye + right_eye -> headset
```

Parameters consumed by this path:

- `depth_strength`
- `convergence`
- `ipd`
- `stereo_scale`
- `max_shift_ratio`
- `screen_roll`

Parameters not consumed by this path:

- `foreground_scale`
- `depth_antialias_strength`
- `temporal_strength`
- `edge_threshold`
- `edge_dilation`
- `mask_feather_radius`
- `hole_fill_*`

Advantages:

- Runtime owns the OpenXR stereo warp.
- Viewer becomes closer to a left/right eye texture presenter.
- Useful as an experimental or compatibility path.

Disadvantages:

- It does not call full `synthesize_stereo()`.
- It does not make cinema/game/still-image full synthesis parameters effective.
- More expensive than rgb-depth because two eye images are generated and transferred.

Performance impact:

- Medium.
- Higher GPU and memory bandwidth cost than rgb-depth.
- Usually less suitable for fast interactive tuning than rgb-depth.

Correct use:

- Compatibility experiments.
- Cases where viewer-side RGB+depth shader behavior is undesirable.
- Not the final path for full quality stereo synthesis.

### 3. OpenXR full stereo synthesis eyes

This is the desired quality path for the new cinema, game, and still-image modes. It is not fully wired into OpenXR output yet.

Target flow:

```text
runtime:
RGB + depth
-> stereo_runtime.synthesize_stereo()
-> left_eye + right_eye, or SBS split into eye images
-> OpenXR runtime result

viewer:
runtime-direct eye texture upload -> headset
```

Parameters that should be consumed by this path:

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

Advantages:

- Makes the new stereo runtime rewrite meaningful in OpenXR.
- Enables full quality processing for cinema, game, and still-image modes.
- Uses occlusion, hole fill, edge processing, and temporal smoothing as designed.

Disadvantages:

- Highest runtime cost.
- Higher latency than rgb-depth.
- Parameter changes require recomputing synthesized eye images.
- 4K quality modes may be expensive for game-like workloads.

Performance impact:

- Highest.
- Adds full synthesis cost after depth inference: depth postprocess, warp/composite, occlusion mask, hole fill, temporal blend, and output packing/upload.
- Must be benchmarked separately by preset and resolution.

Correct use:

- Cinema quality mode.
- Still-image high quality mode.
- Game mode only with lower-latency presets such as `fast` or `fast_plus`, after performance validation.

## Current Wiring Gap

The current pipeline contains a partial fallback that can call full synthesis but does not use its final result for OpenXR display.

Current behavior:

```python
if ctx.run_mode == "OpenXR" and ctx.openxr_runtime_direct:
    runtime_result = ctx.stereo_runtime.process_openxr_frame(...)
else:
    runtime_result = ctx.stereo_runtime.process_rgb_frame(...)
```

When `openxr_runtime_direct` is false, `process_rgb_frame()` does run full stereo synthesis. However, the OpenXR queue currently receives RGB+depth fallback data:

```python
ctx.queue_put_latest(ctx.runtime_q, ((frame_rgb, fallback_depth), capture_start_time))
```

That means the full synthesis output is not actually shown in OpenXR. The viewer falls back to the RGB+depth shader path.

Required fix:

```text
process_rgb_frame()
-> use StereoRuntimeResult.left_eye/right_eye or split StereoRuntimeResult.sbs
-> package as an OpenXR runtime result
-> send to viewer runtime-direct eye texture path
```

## Recommended Mode Mapping

### Traditional Stereo Mode

Use OpenXR rgb-depth.

Reason:

- This reproduces the legacy OpenXR behavior.
- It is low latency.
- It supports realtime depth adjustment well.

Expose or emphasize these controls:

- `Depth Strength`
- `Convergence`
- `IPD`
- `Stereo Scale`
- `Max Shift Ratio`
- `Foreground Scale`
- `Depth Antialias Strength`

Hide, disable, or mark as not applicable in OpenXR rgb-depth:

- `Temporal Strength`
- `Edge Threshold`
- `Edge Dilation`
- `Mask Feather Radius`
- `Hole Fill Mode`
- `Hole Fill Radius`
- `Hole Fill Strength`

### Cinema Mode

Use OpenXR full stereo synthesis eyes once wired.

Reason:

- Cinema mode benefits from high-quality occlusion and hole fill.
- Latency is less critical than visual quality.

Recommended backend:

- `quality_4k` where performance allows.
- Fallback to a balanced or faster preset if runtime cost is too high.

### Game Mode

Use OpenXR full stereo synthesis eyes only with a low-latency preset.

Reason:

- Game mode needs lower latency.
- Full `quality_4k` may be too expensive.

Recommended backend:

- `fast`
- `fast_plus`
- Carefully benchmark before enabling expensive hole fill or temporal settings.

### Still Image Mode

Use OpenXR full stereo synthesis eyes once wired.

Reason:

- Latency matters less.
- High-quality hole fill, edge processing, and temporal settings can be more valuable.

Recommended backend:

- `quality_4k` or still-image high quality preset.

## Summary Table

| Path | Legacy-compatible | Full synthesis | Low latency | Uses hole fill / edge / temporal | Best use |
|---|---:|---:|---:|---:|---|
| OpenXR rgb-depth | Yes | No | Best | No | Traditional OpenXR, realtime tuning |
| OpenXR prewarp eyes | No | No | Medium | No | Compatibility / experiments |
| OpenXR full synthesis eyes | No | Yes | Worst | Yes | Cinema, still image, quality-focused modes |

## Final Recommendation

Keep OpenXR rgb-depth as the legacy-compatible traditional mode.

Add or repair OpenXR full stereo synthesis eyes as a separate quality path for cinema, game, and still-image modes. Do not present SBS-only synthesis controls as active in OpenXR rgb-depth unless they are actually consumed by the current render path.
