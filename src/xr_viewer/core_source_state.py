# Desktop2Stereo OpenXR viewer: source-frame and idle/render state helpers.

import queue as _queue
import time
from dataclasses import dataclass
from typing import Any, Optional

from OpenGL.GL import glDeleteBuffers


@dataclass
class ScreenFramePoll:
    frame: Optional[Any]
    frame_id: int
    source_timestamp: Optional[float] = None
    dequeued: int = 0
    dropped: int = 0
    is_new: bool = False
    reused: bool = False


class ScreenFrameBridge:
    """Non-blocking latest-frame bridge for OpenXR screen presentation."""

    def __init__(self, source_q=None):
        self.source_q = source_q
        self.latest_frame = None
        self.last_presented_frame = None
        self.frame_id = 0
        self.latest_frame_id = 0
        self.last_presented_frame_id = 0
        self.source_timestamp = None
        self.last_presented_source_timestamp = None

    @staticmethod
    def _source_timestamp(frame):
        if isinstance(frame, tuple):
            if len(frame) == 2 and hasattr(frame[0], "left_eye"):
                return frame[1]
            if len(frame) == 3:
                return frame[2]
        return None

    def drain_latest(self):
        latest = None
        dequeued = 0
        if self.source_q is None:
            return ScreenFramePoll(None, self.latest_frame_id)
        try:
            while True:
                latest = self.source_q.get_nowait()
                dequeued += 1
        except _queue.Empty:
            pass

        if latest is None:
            return ScreenFramePoll(None, self.latest_frame_id, dequeued=dequeued)

        self.latest_frame = latest
        self.frame_id += 1
        self.latest_frame_id = self.frame_id
        self.source_timestamp = self._source_timestamp(latest)
        return ScreenFramePoll(
            latest,
            self.latest_frame_id,
            source_timestamp=self.source_timestamp,
            dequeued=dequeued,
            dropped=max(0, dequeued - 1),
            is_new=True,
        )

    def mark_presented(self, frame=None):
        if frame is None:
            frame = self.latest_frame
        if frame is None:
            return ScreenFramePoll(None, self.last_presented_frame_id)
        is_new = self.last_presented_frame_id != self.latest_frame_id
        self.last_presented_frame = frame
        self.last_presented_frame_id = self.latest_frame_id
        self.last_presented_source_timestamp = self.source_timestamp
        return ScreenFramePoll(
            frame,
            self.last_presented_frame_id,
            source_timestamp=self.last_presented_source_timestamp,
            is_new=is_new,
            reused=not is_new,
        )

    def reuse_presented(self):
        if self.last_presented_frame is None:
            return ScreenFramePoll(None, self.last_presented_frame_id)
        return ScreenFramePoll(
            self.last_presented_frame,
            self.last_presented_frame_id,
            source_timestamp=self.last_presented_source_timestamp,
            reused=True,
        )


class CoreSourceStateMixin:
    """Source frame polling plus idle/render state transitions."""

    def _set_source_active(self, active: bool):
        if self._source_active_event is None:
            return
        try:
            if active:
                self._source_active_event.set()
            else:
                self._source_active_event.clear()
        except Exception:
            pass

    def _set_idle_active(self, active: bool):
        if self._idle_active_event is None:
            return
        try:
            if active:
                self._idle_active_event.set()
            else:
                self._idle_active_event.clear()
        except Exception:
            pass

    def _release_idle_gpu_resources(self):
        self._release_runtime_eye_pbos()
        if self._pbo_color is not None and self._cuda_gl:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass
        self._pbo_color = None
        self._pbo_depth = None
        self._cuda_res_color = None
        self._cuda_res_depth = None
        self._pbo_texture_size = None
        if self._cuda_gl is not False:
            self._cuda_gl = None

        for tex_name in ("color_tex", "depth_tex"):
            tex = getattr(self, tex_name, None)
            if tex is not None:
                try:
                    tex.release()
                except Exception:
                    pass
                setattr(self, tex_name, None)
        self._texture_size = None

    def _arm_headset_wait_inference_timeout(self, now=None):
        now = time.perf_counter() if now is None else now
        timeout = self._headset_wait_inference_timeout
        if timeout <= 0.0:
            self._headset_wait_inference_deadline = now
            self._headset_wait_inference_paused = True
            self._set_source_active(False)
            return
        self._headset_wait_inference_deadline = now + timeout
        self._headset_wait_inference_paused = False
        self._set_source_active(True)

    def _refresh_headset_wait_inference_timeout(self, now=None):
        if self._headset_wait_inference_paused:
            return
        if not self._waiting_for_headset:
            return
        if self._headset_wait_inference_deadline <= 0.0:
            return
        now = time.perf_counter() if now is None else now
        if now < self._headset_wait_inference_deadline:
            return
        self._headset_wait_inference_paused = True
        self._set_source_active(False)
        self._enter_hard_idle_wait()
        print(
            f"[OpenXRViewer] No headset detected for {self._headset_wait_inference_timeout:.0f}s; "
            "stopping source inference (source_active_event cleared)"
        )
        print(
            f"[OpenXRViewer] Waiting for VR headset connect... "
            f"(retry in {self._openxr_no_headset_retry_interval:.1f}s)"
        )

    def _resume_source_inference(self):
        self._headset_wait_inference_deadline = 0.0
        self._headset_wait_inference_paused = False
        self._leave_hard_idle_wait()
        self._set_source_active(True)
        print("[OpenXRViewer] Source inference enabled")

    def _enter_hard_idle_wait(self):
        if self._hard_idle_active:
            return
        self._hard_idle_active = True
        self._set_render_active(False)
        self._set_idle_active(True)
        self._release_idle_gpu_resources()
        if self._preview_active and self._show_preview_window:
            try:
                glfw.hide_window(self.window)
                self._preview_active = False
            except Exception:
                pass
        print("[OpenXRViewer] Enter hard idle: source/render paused")

    def _leave_hard_idle_wait(self):
        if not self._hard_idle_active:
            return
        self._hard_idle_active = False
        self._set_idle_active(False)
        if (not self._preview_active) and self._show_preview_window:
            try:
                glfw.show_window(self.window)
                self._preview_active = True
            except Exception:
                pass
        print("[OpenXRViewer] Exit hard idle")

    def _try_restore_openxr(self, now=None):
        if not self._preview_only_mode:
            return False
        now = time.perf_counter() if now is None else now
        if now < self._openxr_retry_cooldown_until:
            return False
        if (now - self._last_openxr_retry_time) < self._openxr_retry_interval:
            return False
        self._last_openxr_retry_time = now
        try:
            self._debug_openxr_trace("_try_restore_openxr attempt", now)
            if self._xr_backend == 'd3d11':
                # Re-probe from a clean instance so restore attempts still try
                # OpenGL first before falling back to D3D11 again.
                self._cleanup_partial_openxr(destroy_instance=True)
                self._ensure_preview_swapchain_size()
            self._debug_openxr_trace(
                f"_try_restore_openxr attempt probes={self._openxr_restore_opengl_probe_attempts}x{self._openxr_restore_opengl_probe_interval:.1f}s",
                now,
            )
            self._init_openxr(
                quiet=True,
                opengl_probe_attempts=self._openxr_restore_opengl_probe_attempts,
                opengl_probe_interval=self._openxr_restore_opengl_probe_interval,
            )
            self._preview_only_mode = False
            self._debug_openxr_trace("_try_restore_openxr success", now)
            print("[OpenXRViewer] OpenXR session created, waiting for render confirmation")
            return True
        except Exception as exc:
            self._cleanup_partial_openxr(destroy_instance=False)
            self._ensure_preview_swapchain_size()
            if self._is_no_headset_error(exc):
                self._defer_openxr_retry(self._openxr_no_headset_retry_interval)
            self._debug_openxr_trace(f"_try_restore_openxr failed: {type(exc).__name__}", now)
            return False

    def _mark_source_frame_received(self, now=None):
        self._last_source_frame_time = time.perf_counter() if now is None else now
        self._source_resume_grace_until = 0.0
        self._source_stalled = False
        self._source_stall_count = 0
        if self._session_running and not self._session_ready_pending:
            self._set_render_active(True)

    def _has_fresh_source_frame(self, now=None):
        now = time.perf_counter() if now is None else now
        if now <= self._source_resume_grace_until:
            return True
        if self._last_source_frame_time <= 0.0:
            return False
        return (now - self._last_source_frame_time) <= self._source_frame_timeout

    def _pause_xr_output_for_source_stall(self):
        now = time.perf_counter()
        last_notice = getattr(self, "_last_source_stall_notice_time", 0.0)
        self._source_stall_count += 1
        if not self._source_stalled or (now - last_notice) >= 5.0:
            age = -1.0
            if self._last_source_frame_time > 0.0:
                age = now - self._last_source_frame_time
            try:
                queued = self.depth_q.qsize()
            except Exception:
                queued = -1
            if getattr(self, "_openxr_debug", False):
                print(
                    "[OpenXRViewer][debug] Source stale: "
                    f"age={age:.2f}s timeout={self._source_frame_timeout:.2f}s "
                    f"q={queued} count={self._source_stall_count}; keeping previous frame"
                )
            self._last_source_stall_notice_time = now
        self._source_stalled = True

    def _set_render_active(self, active: bool):
        if self._render_active_event is None:
            return
        try:
            if active:
                self._render_active_event.set()
            else:
                self._render_active_event.clear()
        except Exception:
            pass

    def _breakdown_inc(self, name, amount=1):
        callback = getattr(self, "_fps_breakdown_inc", None)
        if not callable(callback):
            return
        try:
            callback(name, amount)
        except Exception:
            pass

    def _breakdown_add_time(self, name, seconds):
        callback = getattr(self, "_fps_breakdown_add_time", None)
        if not callable(callback):
            return
        try:
            callback(name, seconds)
        except Exception:
            pass

    def _breakdown_add_value(self, name, value):
        callback = getattr(self, "_fps_breakdown_add_value", None)
        if not callable(callback):
            return
        try:
            callback(name, value)
        except Exception:
            pass

    def _screen_frame_bridge(self):
        bridge = getattr(self, "_openxr_screen_frame_bridge", None)
        if bridge is None or bridge.source_q is not self.depth_q:
            bridge = ScreenFrameBridge(self.depth_q)
            self._openxr_screen_frame_bridge = bridge
        return bridge

    def _record_screen_frame_bridge_age(self, bridge):
        age = max(0, int(bridge.latest_frame_id) - int(bridge.last_presented_frame_id))
        self._breakdown_add_value("openxr_screen_frame_age_frames", float(age))

    def _record_screen_frame_source_latency(self, source_timestamp):
        if source_timestamp is None:
            return
        self._breakdown_add_time("openxr_source_latency", time.perf_counter() - float(source_timestamp))

    def _queue_runtime_effect_submit(self, effect_source_rgb):
        if effect_source_rgb is None:
            return
        needs_effect_source = getattr(self, "_runtime_effects_need_source_texture", None)
        if callable(needs_effect_source):
            try:
                if not needs_effect_source():
                    release_effect_source = getattr(self, "_release_runtime_effect_source_texture", None)
                    if callable(release_effect_source):
                        release_effect_source()
                    return
            except Exception:
                pass
        if getattr(self, "_pending_runtime_effect_source", None) is not None:
            self._breakdown_inc("openxr_effect_submit_overwrite")
        self._pending_runtime_effect_source = effect_source_rgb

    def _flush_runtime_effect_submit(self):
        effect_source_rgb = getattr(self, "_pending_runtime_effect_source", None)
        if effect_source_rgb is None:
            return
        self._pending_runtime_effect_source = None
        try:
            submitted = self._submit_runtime_effect_source_texture(effect_source_rgb)
        except Exception as exc:
            print(f"[OpenXRViewer] Runtime effect submit failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_effect_submit_failed")
            return
        if submitted is False:
            self._breakdown_inc("openxr_effect_downsample_prewarm_skip")
            return
        try:
            self._prewarm_runtime_effect_downsample()
        except Exception as exc:
            print(f"[OpenXRViewer] Runtime effect downsample prewarm failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_effect_downsample_prewarm_failed")

    def _prewarm_runtime_effect_downsample(self):
        source_tex = getattr(self, "_runtime_effect_safe_source_tex", None)
        source_size = getattr(self, "_runtime_effect_safe_source_size", None)
        promote_ready = getattr(self, "_promote_runtime_effect_ready_texture", None)
        if callable(promote_ready):
            source_tex = promote_ready()
            source_size = getattr(self, "_runtime_effect_safe_source_size", None)
        if source_tex is None or source_size is None:
            return
        mode = str(getattr(self, "_glow_mode", "") or "").strip().lower()
        glow_needs_downsample = (
            mode in ("screen", "surround")
            and (
                float(getattr(self, "_glow_intensity_multiplier", 0.0) or 0.0) > 0.0
                or float(getattr(self, "_glow_shell_intensity_multiplier", 0.0) or 0.0) > 0.0
            )
        )
        light_needs_downsample = float(getattr(self, "_screen_light_intensity", 0.0) or 0.0) > 0.0 and (
            getattr(self, "_panorama_background_path", None)
            or bool(getattr(self, "_env_model_visible", False) and getattr(self, "_env_model_prims", []))
        )
        if not (glow_needs_downsample or light_needs_downsample):
            return
        prepare = getattr(self, "_prepare_glow_downsample_texture", None)
        if not callable(prepare):
            return
        if prepare(source_tex, source_size) is not None:
            self._breakdown_inc("openxr_effect_downsample_prewarm")

    def _poll_source_frame(self, upload=False):
        poll_start = time.perf_counter()
        bridge = self._screen_frame_bridge()
        poll = bridge.drain_latest()
        latest = poll.frame
        dequeued = poll.dequeued

        if dequeued:
            self._breakdown_inc("viewer_get", dequeued)
            if poll.dropped:
                self._breakdown_inc("viewer_drop", poll.dropped)

        if latest is not None:
            self._pending_source_frame = latest
            self._mark_source_frame_received()

        if not upload:
            self._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
            return latest is not None

        if self._pending_source_frame is None:
            reuse = bridge.reuse_presented()
            if reuse.frame is not None:
                self._breakdown_inc("openxr_reused_screen_frame")
                self._record_screen_frame_bridge_age(bridge)
                self._record_screen_frame_source_latency(reuse.source_timestamp)
            self._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
            return False

        budget_ms = float(getattr(self, "_openxr_screen_upload_budget_ms", 0.0) or 0.0)
        skip_armed = bool(getattr(self, "_openxr_screen_upload_budget_skip_armed", False))
        if budget_ms > 0.0 and skip_armed:
            reuse = bridge.reuse_presented()
            if reuse.frame is not None:
                self._openxr_screen_upload_budget_skip_armed = False
                self._breakdown_inc("openxr_reused_screen_frame")
                self._breakdown_inc("openxr_screen_upload_budget_skip")
                self._record_screen_frame_bridge_age(bridge)
                self._record_screen_frame_source_latency(reuse.source_timestamp)
                self._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
                return False

        pending_frame = self._pending_source_frame
        source_frame, frame_ts = self._normalize_source_frame(pending_frame)
        self._pending_source_frame = None

        upload_start = time.perf_counter()
        effect_source_rgb = None
        if self._is_runtime_result(source_frame):
            effect_source_rgb = self._update_runtime_frame(source_frame)
        else:
            rgb, depth = source_frame
            self._update_frame(rgb, depth)
        upload_elapsed = time.perf_counter() - upload_start
        upload_elapsed_ms = upload_elapsed * 1000.0
        if budget_ms > 0.0:
            self._openxr_screen_upload_budget_skip_armed = upload_elapsed_ms > budget_ms
        presented = bridge.mark_presented(pending_frame)
        self._record_screen_frame_bridge_age(bridge)
        self._record_screen_frame_source_latency(presented.source_timestamp)
        self._breakdown_inc("openxr_new_screen_frame")
        self._breakdown_add_time("openxr_upload", upload_elapsed)
        self._queue_runtime_effect_submit(effect_source_rgb)
        if frame_ts is not None:
            self.total_latency = (time.perf_counter() - frame_ts) * 1000.0
        sbs_now = time.perf_counter()
        self._sbs_ts_ring.append(sbs_now)
        m = len(self._sbs_ts_ring)
        if m >= 2:
            sbs_span = sbs_now - self._sbs_ts_ring[0]
            if sbs_span > 0:
                self.sbs_fps = (m - 1) / sbs_span
        self._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
        return True

    def _normalize_source_frame(self, item):
        if self._is_runtime_result(item):
            return item, None
        if isinstance(item, tuple):
            if len(item) == 2 and self._is_runtime_result(item[0]):
                return item[0], item[1]
            if len(item) == 3:
                rgb, depth, frame_ts = item
                return (rgb, depth), frame_ts
        raise RuntimeError(f"Unsupported OpenXR source frame: {type(item).__name__}")

    def _is_runtime_result(self, item):
        return (
            hasattr(item, "left_eye")
            and hasattr(item, "right_eye")
            and hasattr(item, "depth")
        )

    def _publish_runtime_config(self, *, include_stereo=False):
        callback = self._runtime_config_callback
        if not callable(callback):
            return
        try:
            payload = {"screen_roll": self.screen_roll}
            if include_stereo:
                depth_strength = float(self.depth_strength)
                if self._quad_layer_can_replace_projection_screen():
                    depth_strength *= float(getattr(self, '_xr_quad_layer_stereo_boost', 1.0))
                payload.update(
                    depth_strength=depth_strength,
                    convergence=self.convergence,
                )
            callback(**payload)
        except Exception:
            pass

    def _has_renderable_source_frame(self):
        if self._runtime_direct_source:
            if getattr(self, '_use_d3d11', False):
                renderer = getattr(self, '_d3d11_native_renderer', None)
                if renderer is not None and renderer.has_frame:
                    return True
            return all(self._runtime_eye_textures)
        return self.color_tex is not None and self.depth_tex is not None

    def _should_show_source_border(self, now=None):
        if getattr(self, "_hard_idle_active", False):
            return False
        source_active_event = getattr(self, "_source_active_event", None)
        if source_active_event is not None:
            try:
                if not source_active_event.is_set():
                    return False
            except Exception:
                return False
        if not self._has_renderable_source_frame():
            return False
        return self._has_fresh_source_frame(now)
