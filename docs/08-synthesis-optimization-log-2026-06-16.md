# Synthesis Optimization Log - 2026-06-16

## Scope

This document records the 4K stereo synthesis optimizations made after the depth backend work.

Strict boundary:

- Did not change the depth model.
- Did not change inference resolution.
- Did not change RGB resize mode, antialias behavior, or normalization semantics.
- Did not change ONNX / TensorRT export parameters.
- Did not change the generated depth values.

All changes below are limited to stereo synthesis after RGB + depth are already available.

## Baseline Before This Round

Reference profile:

```text
outputs/synthesis_profile_4k/quality_half_profile_no_baseline_warp.json
```

Quality 4K, 2 layers, Half-SBS, 4K input:

| Stage | Mean ms |
|---|---:|
| baseline_shift | 1.238 |
| make_layers | 4.403 |
| warp_layers | 14.543 |
| composite | 9.745 |
| occlusion | 7.679 |
| hole_fill | 20.658 |
| make_sbs | 1.886 |

Main bottlenecks:

1. `hole_fill`
2. `warp_layers`
3. `composite`
4. `occlusion`

## Kept Optimization 1: Layer Center Cache + In-Place Composite

Commit:

```text
49da1fc perf: reduce quality synthesis allocations
```

Files:

- `src/stereo_lab/layers.py`
- `tests/test_synthesis.py`

Changes:

- Cached `torch.linspace` layer centers by `(layers, device, dtype)`.
- Changed `composite_layers` from repeated `out = out + layer * weight` allocation to `out.addcmul_(...)`.

Why this is safe:

- Layer center values are identical for the same layer count, device, and dtype.
- Composite math is unchanged.
- No depth inference path is touched.

Profile after change:

```text
outputs/synthesis_profile_4k/quality_half_profile_composite_inplace.json
```

| Stage | Before ms | After ms |
|---|---:|---:|
| composite | 9.745 | 6.890 |
| synthesis mean | ~59.1 | 58.319 |

## Kept Optimization 2: Separable Box Blur For Hole Fill

Commit:

```text
49da1fc perf: reduce quality synthesis allocations
```

Files:

- `src/stereo_lab/hole_fill.py`
- `tests/test_synthesis.py`

Changes:

- Replaced one `k x k` depthwise box blur with two depthwise passes:
  - horizontal `1 x k`
  - vertical `k x 1`
- Cached horizontal and vertical kernels separately.

Why this is safe:

- A box filter is separable, so the mathematical kernel is the same.
- Added `test_box_blur_matches_2d_kernel`.
- Stereo output semantics are unchanged except for possible floating-point accumulation order noise.
- No depth inference path is touched.

Profile after change:

```text
outputs/synthesis_profile_4k/quality_half_profile_separable_fill.json
```

| Stage | Before ms | After ms |
|---|---:|---:|
| hole_fill | 21.331 | 19.434 |
| synthesis mean | 58.319 | 56.162 |

End-to-end after this optimization:

```text
outputs/end_to_end_4k/quality_native_synthesis_safe_opt.json
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 16.881 | 55.058 | 71.941 | 13.90 |
| Full-SBS | 15.275 | 54.308 | 69.584 | 14.37 |

## Rejected Experiment: Batch Warp

Status:

```text
reverted, not committed
```

Attempt:

- Added a temporary `warp_horizontal_batch`.
- Tried to combine layer/eye warps into one larger batch `grid_sample`.

Reason for rejection:

- 4K memory pressure and input/grid replication outweighed the reduced kernel launch count.
- It did not improve `warp_layers`.

Profile:

```text
outputs/synthesis_profile_4k/quality_half_profile_batch_warp.json
```

| Stage | Previous ms | Batch-warp ms |
|---|---:|---:|
| warp_layers | ~14.25 | 14.483 |
| synthesis mean | 56.162 | 57.335 |

Decision:

- Do not use batch warp in the current PyTorch 4K path.
- Future warp optimization should likely be a fused CUDA/shader/TensorRT-style kernel, not larger PyTorch batch replication.

## Kept Optimization 3: In-Place Depth Edge Accumulation

Commit:

```text
1e2a171 perf: reduce depth edge allocations
```

Files:

- `src/stereo_lab/layers.py`
- `tests/test_synthesis.py`

Changes:

- Replaced padded `dx` + padded `dy` temporary tensors in `depth_edges`.
- New implementation uses one `edges = torch.zeros_like(depth)` and accumulates x/y differences in place.

Why this is safe:

- Added `test_depth_edges_matches_padded_gradient_formula`.
- The test verifies exact equality with the old padded-gradient formula.
- This affects occlusion mask calculation only after depth is already generated.
- No depth inference path is touched.

Profile:

```text
outputs/synthesis_profile_4k/quality_half_profile_depth_edges_inplace_rerun.json
```

| Stage | Before ms | After ms |
|---|---:|---:|
| occlusion | ~7.81 | ~7.09 |
| synthesis mean | 56.162 | 56.166 |

End-to-end after this optimization:

```text
outputs/end_to_end_4k/quality_native_synthesis_edges_opt.json
```

| Output | Depth ms | Synthesis ms | Total ms | FPS |
|---|---:|---:|---:|---:|
| Half-SBS | 16.200 | 54.951 | 71.152 | 14.05 |
| Full-SBS | 15.201 | 53.022 | 68.224 | 14.66 |

## Verification Commands

Syntax:

```powershell
.\python3\python.exe -B -c "import ast, pathlib; files=list(pathlib.Path('src').rglob('*.py'))+list(pathlib.Path('scripts').rglob('*.py'))+list(pathlib.Path('tests').rglob('*.py')); [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('syntax ok', len(files), 'files')"
```

Tests:

```powershell
.\python3\python.exe -B -m pytest -q
```

Latest result:

```text
22 passed
```

Profile:

```powershell
.\python3\python.exe -B scripts\profile_synthesis_4k.py --rgb 4K.jpg --out outputs\synthesis_profile_4k\quality_half_profile_depth_edges_inplace_rerun.json --backend quality_4k --layers 2 --output-format half_sbs --iters 5
```

End-to-end:

```powershell
.\python3\python.exe -B scripts\bench_end_to_end_4k.py --rgb 4K.jpg --out outputs\end_to_end_4k\quality_native_synthesis_edges_opt.json --warmup 2 --iters 5 --backend quality_4k --layers 2 --depth-backend tensorrt_native --output-format half_sbs --output-format full_sbs
```

## Current Bottlenecks

Latest profile indicates the main remaining bottlenecks are:

| Stage | Approx ms | Notes |
|---|---:|---|
| hole_fill | ~20 | Full-frame 4K blur/blend remains bandwidth-heavy |
| warp_layers | ~14-15 | Multiple `grid_sample` calls dominate geometry synthesis |
| occlusion | ~7 | Improved, but still non-trivial |
| composite | ~6-7 | Improved, still bandwidth-bound |

## Next Safe Optimization Candidates

These must still preserve depth model, inference resolution, resize, antialias, and normalization semantics.

1. Profile `hole_fill` at kernel level and evaluate whether mask-limited fill can be implemented without changing visible semantics.
2. Investigate a fused CUDA/Torch extension for warp + composite, with output-equivalence tests against the PyTorch reference.
3. Keep PyTorch batch warp rejected unless a new implementation avoids 4K tensor replication.
4. Add visual regression/contact sheet for baseline vs quality after every synthesis optimization.

