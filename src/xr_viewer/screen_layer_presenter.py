import ctypes
import time

from .background_layer_renderer import BackgroundLayerRenderer
from .background_presenter import BackgroundPresenter

try:
    import xr
except ImportError:
    xr = None


class ScreenLayerPresenter:
    def __init__(self, viewer):
        self.viewer = viewer
        self._frame_background_layers = []
        self._frame_projection_layer = None
        self._frame_quad_layers = []
        self._frame_background_projection_fallback = False

    def poll_screen_frame(self):
        viewer = self.viewer
        poll_start = time.perf_counter()
        bridge = viewer._screen_frame_bridge()
        poll = bridge.drain_latest()
        latest = poll.frame
        dequeued = poll.dequeued

        if dequeued:
            viewer._breakdown_inc("viewer_get", dequeued)
            if poll.dropped:
                viewer._breakdown_inc("viewer_drop", poll.dropped)

        if latest is not None:
            viewer._mark_source_frame_received()

        if not bridge.has_unpresented_frame():
            reuse = bridge.reuse_presented()
            if reuse.frame is not None:
                viewer._breakdown_inc("openxr_reused_screen_frame")
                viewer._record_screen_frame_bridge_age(bridge)
                viewer._record_screen_frame_source_latency(reuse.source_timestamp)
            viewer._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
            return False

        budget_ms = float(getattr(viewer, "_openxr_screen_upload_budget_ms", 0.0) or 0.0)
        skip_armed = bool(getattr(viewer, "_openxr_screen_upload_budget_skip_armed", False))
        if budget_ms > 0.0 and skip_armed:
            reuse = bridge.reuse_presented()
            if reuse.frame is not None:
                viewer._openxr_screen_upload_budget_skip_armed = False
                viewer._breakdown_inc("openxr_reused_screen_frame")
                viewer._breakdown_inc("openxr_screen_upload_budget_skip")
                viewer._record_screen_frame_bridge_age(bridge)
                viewer._record_screen_frame_source_latency(reuse.source_timestamp)
                viewer._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
                return False

        pending_frame = bridge.latest_frame
        source_frame, frame_ts = viewer._normalize_source_frame(pending_frame)

        upload_start = time.perf_counter()
        effect_source_rgb = None
        if viewer._is_runtime_result(source_frame):
            effect_source_rgb = viewer._update_runtime_frame(source_frame)
        else:
            rgb, depth = source_frame
            viewer._update_frame(rgb, depth)
        upload_elapsed = time.perf_counter() - upload_start
        if budget_ms > 0.0:
            viewer._openxr_screen_upload_budget_skip_armed = (upload_elapsed * 1000.0) > budget_ms
        if not viewer._has_renderable_source_frame():
            viewer._breakdown_inc("openxr_screen_upload_not_renderable")
            viewer._breakdown_add_time("openxr_upload", upload_elapsed)
            viewer._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
            return False
        if getattr(viewer, '_runtime_eye_reused_previous_frame', False):
            viewer._breakdown_add_time("openxr_upload", upload_elapsed)
            viewer._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
            return False

        presented = bridge.mark_presented(pending_frame)
        viewer._record_screen_frame_bridge_age(bridge)
        viewer._record_screen_frame_source_latency(presented.source_timestamp)
        viewer._breakdown_inc("openxr_new_screen_frame")
        viewer._breakdown_add_time("openxr_upload", upload_elapsed)
        viewer._queue_runtime_effect_submit(effect_source_rgb)
        if frame_ts is not None:
            viewer.total_latency = (time.perf_counter() - frame_ts) * 1000.0
        sbs_now = time.perf_counter()
        viewer._sbs_ts_ring.append(sbs_now)
        m = len(viewer._sbs_ts_ring)
        if m >= 2:
            sbs_span = sbs_now - viewer._sbs_ts_ring[0]
            if sbs_span > 0:
                viewer.sbs_fps = (m - 1) / sbs_span
        viewer._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
        return True

    def update_or_reuse(self, *, screen_frame_uploaded=False):
        return self.viewer._update_quad_layer_swapchains(force=screen_frame_uploaded)

    def quad_screen_unavailable_reason(self):
        reason = getattr(self.viewer, '_quad_layer_unavailable_reason', None)
        return reason() if callable(reason) else None

    def render_quad_screen_overlay(self, *, mgl_fbo, vp_mat, mark_perf=None):
        viewer = self.viewer
        viewer._breakdown_inc("openxr_quad_screen_overlay")
        if mark_perf:
            mark_perf('screen_quad_layer')
        if not viewer._screen_curved:
            viewer._render_border(mgl_fbo, vp_mat)
        if mark_perf:
            mark_perf('border')

    def projection_layer_reason(self, *, background_projection_fallback=None):
        viewer = self.viewer
        if background_projection_fallback is True:
            return "background_projection_fallback"
        if background_projection_fallback is None:
            background_renderer = getattr(viewer, '_background_layer_renderer', None)
            if background_renderer is None:
                background_renderer = BackgroundLayerRenderer(viewer)
                viewer._background_layer_renderer = background_renderer
            try:
                panorama_ready = background_renderer.panorama_ready()
                if panorama_ready and not background_renderer.native_background_available(panorama_ready=panorama_ready):
                    return "panorama_projection_fallback"
            except Exception as exc:
                print(f"[OpenXRViewer] Background projection gate failed: {type(exc).__name__}: {exc}")
                viewer._breakdown_inc('openxr_background_layer_failed')
                return "background_gate_failed"
        if viewer._keyboard_visible and viewer._keyboard_tex is not None:
            return "keyboard"
        if viewer._aim_mat_l is not None or viewer._aim_mat_r is not None:
            return "controller_aim"
        if viewer._grip_mat_l is not None or viewer._grip_mat_r is not None:
            return "controller_grip"
        if float(getattr(viewer, '_border_alpha', 0.0) or 0.0) > 0.0:
            return "screen_border"
        if any(getattr(viewer, name, None) is not None for name in (
            '_depth_osd_tex', '_screen_osd_tex', '_preset_osd_tex', '_seat_adjust_osd_tex'
        )):
            return "osd"
        if viewer._brand_osd_tex is not None and viewer._grip_mat_r is not None:
            return "brand_osd"
        if viewer._hand_fps_visible and viewer._overlay_tex is not None:
            return "hand_fps"
        if viewer._team_fps_visible and viewer._team_status_tex is not None:
            return "team_fps"
        if viewer._calibration_mode:
            return "calibration"
        if viewer._fps_overlay_visible and viewer._help_tex is not None:
            return "help"
        if viewer._team_status_visible and viewer._team_help_visible and viewer._team_help_tex is not None:
            return "team_help"
        return "scene"

    def projection_layer_needed(self, *, background_projection_fallback=None):
        return True

    def prepare_projection_frame_state(self):
        self.viewer._openxr_quad_screen_unavailable_reason = self.quad_screen_unavailable_reason()

    def prepare_frame_layers(self, *, screen_frame_uploaded=False):
        self._frame_background_layers = []
        self._frame_projection_layer = None
        self._frame_quad_layers = []
        self._frame_background_projection_fallback = False
        background_renderer = getattr(self.viewer, '_background_layer_renderer', None)
        if background_renderer is None:
            background_renderer = BackgroundLayerRenderer(self.viewer)
            self.viewer._background_layer_renderer = background_renderer
        updated_quad_eyes = self.update_or_reuse(screen_frame_uploaded=screen_frame_uploaded)
        quad_layers, quad_layer_headers, updated_quad_eyes = self.make_quad_layers(updated_quad_eyes)
        self._frame_quad_layers = quad_layers
        try:
            background_layer_headers, background_projection_fallback = background_renderer.make_background_layers()
            self._frame_background_layers = list(getattr(background_renderer, '_frame_background_layers', []))
        except Exception as exc:
            print(f"[OpenXRViewer] Background layer build failed: {type(exc).__name__}: {exc}")
            self.viewer._breakdown_inc('openxr_background_layer_failed')
            background_layer_headers, background_projection_fallback = [], True
            self._frame_background_layers = []
        self._frame_background_projection_fallback = bool(background_projection_fallback)
        self.prepare_projection_frame_state()
        projection_reason = self.projection_layer_reason(
            background_projection_fallback=background_projection_fallback
        )
        render_projection_layer = self.projection_layer_needed(
            background_projection_fallback=background_projection_fallback
        )
        if projection_reason != getattr(self.viewer, '_last_projection_layer_reason', None):
            self.viewer._last_projection_layer_reason = projection_reason
            print(f"[OpenXRViewer] Projection layer active: reason={projection_reason}")
        return quad_layers, quad_layer_headers, updated_quad_eyes, render_projection_layer, background_layer_headers

    def append_frame_layers(self, composition_layers, *, projection_views=(), projection_space=None, quad_layer_headers=(), background_layer_headers=()):
        composition_layers.extend(background_layer_headers)
        if projection_views:
            try:
                projection_layer = xr.CompositionLayerProjection(
                    space=projection_space,
                    views=projection_views,
                )
                self._frame_projection_layer = projection_layer
                composition_layers.append(
                    ctypes.cast(
                        ctypes.pointer(projection_layer),
                        ctypes.POINTER(xr.CompositionLayerBaseHeader),
                    )
                )
            except Exception as exc:
                print(f"[OpenXRViewer] Projection layer append failed: {type(exc).__name__}: {exc}")
                self.viewer._breakdown_inc('openxr_projection_render_failed')
                self._frame_projection_layer = None
        composition_layers.extend(quad_layer_headers)
        return composition_layers

    def make_quad_layers(self, updated_quad_eyes):
        viewer = self.viewer
        quad_layers = []
        quad_layer_headers = []
        for quad_eye_index in updated_quad_eyes:
            try:
                quad_layer = viewer._make_quad_layer(quad_eye_index)
                if quad_layer is None:
                    raise RuntimeError(f"missing quad layer for eye {quad_eye_index}")
                quad_layers.append(quad_layer)
                quad_layer_headers.append(
                    ctypes.cast(
                        ctypes.pointer(quad_layer),
                        ctypes.POINTER(xr.CompositionLayerBaseHeader),
                    )
                )
            except Exception as exc:
                set_failed = getattr(viewer, '_set_quad_layer_failed', None)
                if callable(set_failed):
                    set_failed(f"layer_build_failed_{type(exc).__name__}")
                else:
                    viewer._xr_quad_layer_active = False
                    viewer._xr_quad_layer_failed = True
                    viewer._xr_quad_layer_failure_reason = f"layer_build_failed_{type(exc).__name__}"
                viewer._breakdown_inc('openxr_quad_layer_failed')
                print(f"[OpenXRViewer] Quad layer build failed: {type(exc).__name__}: {exc}")
                return [], [], []
        return quad_layers, quad_layer_headers, list(updated_quad_eyes)
