from __future__ import annotations

import queue
from types import SimpleNamespace

from stereo_runtime.pipeline import RuntimePipelineContext, RuntimePipelineLoop


class OneShotShutdown:
    def __init__(self):
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > 2


class FakeRuntime:
    def process_rgb_frame(self, runtime_rgb):
        return SimpleNamespace(
            depth="depth",
            sbs="sbs",
            timing={"total_ms": 1.0},
            debug_info={"backend": "fake"},
        )


def test_runtime_pipeline_processes_one_frame():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    shutdown = OneShotShutdown()
    stats = {}
    breakdown = {}
    latencies = {}
    calls = []

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def breakdown_inc(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    def breakdown_add_time(name, seconds):
        breakdown[f"{name}_count"] = breakdown.get(f"{name}_count", 0) + 1

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        device="cpu",
        use_cudart=False,
        thread_latencies=latencies,
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: calls.append("health"),
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=breakdown_add_time,
        breakdown_add_runtime_timing=lambda result: calls.append(("timing", result.timing)),
        set_preprocess_backend=lambda backend: calls.append(("pre", backend)),
        queue_clear=lambda q: calls.append(("clear", q)),
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: calls.append("mode"),
        apply_stereo_hot_reload_if_needed=lambda: calls.append("hot-reload"),
        warmup_stereo_once_for_frame=lambda frame: calls.append(("warmup", frame)),
        log_fast_plus_fused_runtime_state=lambda result: calls.append(("fused", result.debug_info)),
    )

    RuntimePipelineLoop(context).run()

    runtime_result, capture_start_time = runtime_q.get_nowait()
    assert runtime_result.sbs == "sbs"
    assert capture_start_time == 10.0
    assert stats["raw_get"] == 1
    assert stats["runtime_frames"] == 1
    assert breakdown["raw_get"] == 1
    assert breakdown["runtime"] == 1
    assert latencies["capture"] > 0
    assert latencies["runtime"] >= 0
    assert ("pre", "fake-preprocess") in calls
    assert "mode" in calls
    assert "hot-reload" in calls
    assert ("warmup", "runtime-rgb") in calls
