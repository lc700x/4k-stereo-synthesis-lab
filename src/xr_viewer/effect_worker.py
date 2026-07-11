import time


class EffectWorker:
    def __init__(self, viewer):
        self.viewer = viewer

    def _interval(self):
        raw = getattr(self.viewer, "_openxr_effect_worker_interval", None)
        if raw is None:
            import os
            raw = os.environ.get("D2S_OPENXR_EFFECT_WORKER_INTERVAL", "1")
        try:
            return max(1, int(raw))
        except (TypeError, ValueError):
            return 1

    def prewarm_after_submit(self):
        viewer = self.viewer
        interval = self._interval()
        frame_id = int(getattr(viewer, "_frame_count", 0) or 0)
        if interval > 1 and frame_id > 0 and (frame_id % interval) != 0:
            viewer._breakdown_inc("openxr_effect_worker_interval_skip")
            return
        scheduler = viewer._runtime_effect_submit_scheduler()
        source_tex, source_size, source_frame_id = scheduler.latest_safe()
        if source_tex is None or source_size is None:
            return
        mode = str(getattr(viewer, "_glow_mode", "") or "").strip().lower()
        glow_needs_downsample = (
            mode in ("screen", "surround")
            and (
                float(getattr(viewer, "_glow_intensity_multiplier", 0.0) or 0.0) > 0.0
                or float(getattr(viewer, "_glow_shell_intensity_multiplier", 0.0) or 0.0) > 0.0
            )
        )
        light_needs_downsample = float(getattr(viewer, "_screen_light_intensity", 0.0) or 0.0) > 0.0 and (
            getattr(viewer, "_panorama_background_path", None)
            or bool(getattr(viewer, "_env_model_visible", False) and getattr(viewer, "_env_model_prims", []))
        )
        if not (glow_needs_downsample or light_needs_downsample):
            return
        source_key = (id(source_tex), tuple(source_size))
        if getattr(viewer, "_runtime_effect_downsample_failed_key", None) == source_key:
            viewer._breakdown_inc("openxr_effect_downsample_prewarm_suppressed")
            return
        prepare = getattr(viewer, "_prepare_glow_downsample_texture", None)
        if not callable(prepare):
            return
        start = time.perf_counter()
        try:
            glow_tex = prepare(source_tex, source_size) if glow_needs_downsample else None
            light_tex = prepare(source_tex, source_size, target_size=(3, 3)) if light_needs_downsample else None
        except Exception as exc:
            viewer._breakdown_add_time("openxr_effect_downsample_prewarm", time.perf_counter() - start)
            print(f"[OpenXRViewer] Runtime effect downsample prewarm failed: {type(exc).__name__}: {exc}")
            viewer._runtime_effect_downsample_failed_key = source_key
            viewer._breakdown_inc("openxr_effect_downsample_prewarm_failed")
            return
        viewer._breakdown_add_time("openxr_effect_downsample_prewarm", time.perf_counter() - start)
        viewer._runtime_effect_downsample_failed_key = None
        if glow_tex is not None:
            scheduler.publish_downsample(glow_tex, getattr(glow_tex, 'size', None), source_frame_id)
        if light_tex is not None:
            scheduler.publish_light_probe(light_tex, getattr(light_tex, 'size', None), source_frame_id)
        if glow_tex is not None or light_tex is not None:
            viewer._breakdown_inc("openxr_effect_downsample_prewarm")
