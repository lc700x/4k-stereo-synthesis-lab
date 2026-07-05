import time


class EffectWorker:
    def __init__(self, viewer):
        self.viewer = viewer

    def prewarm_after_submit(self):
        viewer = self.viewer
        scheduler = viewer._runtime_effect_submit_scheduler()
        source_tex, source_size, _source_frame_id = scheduler.latest_safe()
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
            tex = prepare(source_tex, source_size)
        except Exception as exc:
            viewer._breakdown_add_time("openxr_effect_downsample_prewarm", time.perf_counter() - start)
            print(f"[OpenXRViewer] Runtime effect downsample prewarm failed: {type(exc).__name__}: {exc}")
            viewer._runtime_effect_downsample_failed_key = source_key
            viewer._breakdown_inc("openxr_effect_downsample_prewarm_failed")
            return
        viewer._breakdown_add_time("openxr_effect_downsample_prewarm", time.perf_counter() - start)
        viewer._runtime_effect_downsample_failed_key = None
        if tex is not None:
            viewer._breakdown_inc("openxr_effect_downsample_prewarm")
