import time

try:
    import xr
except ImportError:
    xr = None


class FrameSubmitter:
    def __init__(self, viewer):
        self.viewer = viewer

    def submit(self, layers, *, display_time, submit_start=0.0):
        viewer = self.viewer
        breakdown_enabled = callable(getattr(viewer, "_fps_breakdown_add_time", None))
        viewer._breakdown_inc("openxr_layer_count", len(layers))
        end_frame_start = time.perf_counter() if breakdown_enabled else 0.0
        try:
            xr.end_frame(
                viewer._xr_session,
                xr.FrameEndInfo(
                    display_time=display_time,
                    environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                    layers=layers,
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] xr.end_frame failed: layers={len(layers)} {type(exc).__name__}: {exc}")
            raise
        if breakdown_enabled:
            now = time.perf_counter()
            viewer._breakdown_add_time("openxr_end_frame", now - end_frame_start)
            if submit_start:
                viewer._breakdown_add_time("openxr_submit_frame", now - submit_start)
