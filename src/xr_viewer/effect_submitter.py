import time

from .effect_worker import EffectWorker


class EffectSubmitter:
    def __init__(self, viewer):
        self.viewer = viewer
        self.worker = EffectWorker(viewer)

    def _breakdown_inc(self, name, amount=1):
        callback = getattr(self.viewer, "_breakdown_inc", None)
        if callable(callback):
            callback(name, amount)

    def flush_after_submit(self, *, should_render, screen_frame_uploaded):
        if not should_render:
            return False
        viewer = self.viewer
        scheduler = viewer._runtime_effect_submit_scheduler()
        if scheduler.pending_source is None:
            return False
        if not screen_frame_uploaded:
            scheduler.clear_pending_source()
            self._breakdown_inc("openxr_effect_source_reused_safe")
            return False
        should_submit = getattr(viewer, "_should_submit_runtime_effect_source", None)
        if callable(should_submit) and not should_submit():
            scheduler.clear_pending_source()
            self._breakdown_inc("openxr_effect_source_interval_skip")
            self._breakdown_inc("openxr_effect_source_reused_safe")
            return False
        if bool(getattr(viewer, "_openxr_effect_submit_budget_skip_armed", False)):
            viewer._openxr_effect_submit_budget_skip_armed = False
            scheduler.clear_pending_source()
            self._breakdown_inc("openxr_effect_submit_budget_skip")
            self._breakdown_inc("openxr_effect_source_reused_safe")
            return False

        def _promote_ready():
            try:
                result = scheduler.promote_ready_once(getattr(viewer, "_frame_count", 0))
            except Exception as exc:
                print(f"[OpenXRViewer] Runtime effect source promote failed: {type(exc).__name__}: {exc}")
                self._breakdown_inc("openxr_effect_source_promote_failed")
                return
            if result == "reused":
                self._breakdown_inc("openxr_effect_source_promote_reuse")
            elif result == "promoted":
                self._breakdown_inc("openxr_effect_source_safe_publish")

        submit_start = time.perf_counter()
        try:
            status = scheduler.flush_pending_source(
                viewer._submit_runtime_effect_source_texture,
                _promote_ready,
            )
        except Exception as exc:
            scheduler.clear_pending_source()
            print(f"[OpenXRViewer] Runtime effect submit failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_effect_submit_failed")
            return True
        if status == "skipped":
            self._breakdown_inc("openxr_effect_downsample_prewarm_skip")
        elif status == "submitted":
            self.worker.prewarm_after_submit()
        budget_ms = float(getattr(viewer, "_openxr_effect_submit_budget_ms", 0.0) or 0.0)
        if budget_ms > 0.0:
            viewer._openxr_effect_submit_budget_skip_armed = ((time.perf_counter() - submit_start) * 1000.0) > budget_ms
        return True
