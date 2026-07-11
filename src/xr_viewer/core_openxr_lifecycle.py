# Desktop2Stereo OpenXR viewer: OpenXR lifecycle, retry, and wait-state helpers.

import sys
import time


class CoreOpenXRLifecycleMixin:
    """Backend selection, OpenXR retry, preview-only, and wait-state helpers."""

    def _init_openxr(self, quiet=False, opengl_probe_attempts=None, opengl_probe_interval=None):
        """Try OpenGL first; fall back to D3D11 on Windows if OpenGL fails."""
        if self._forced_xr_backend == 'd3d11' or self._xr_backend == 'd3d11':
            self._init_openxr_d3d11(quiet=quiet)
            self._use_d3d11 = True
            return
        try:
            self._init_openxr_opengl_with_retry(
                quiet=quiet,
                attempts=opengl_probe_attempts,
                interval=opengl_probe_interval,
            )
            return
        except Exception as e:
            if self._is_no_headset_error(e):
                raise
            if sys.platform != "win32":
                raise
            if not quiet:
                print(f"[OpenXRViewer] OpenGL init failed, fallback to D3D11: {type(e).__name__}: {e}")
            self._cleanup_partial_openxr(destroy_instance=True)

        self._init_openxr_d3d11(quiet=quiet)
        self._use_d3d11 = True
        if not quiet:
            print("[OpenXRViewer] D3D11 fallback active after OpenGL init failure")

    def _init_openxr_opengl_with_retry(self, quiet=False, attempts=None, interval=None):
        attempts = self._openxr_opengl_probe_attempts if attempts is None else max(1, int(attempts))
        interval = self._openxr_opengl_probe_interval if interval is None else max(0.0, float(interval))
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                self._init_openxr_opengl(quiet=quiet)
                return
            except Exception as exc:
                last_exc = exc
                self._cleanup_partial_openxr(destroy_instance=True)
                if self._is_no_headset_error(exc):
                    break
                if attempt >= attempts:
                    break
                if not quiet:
                    print(
                        f"[OpenXRViewer] OpenGL probe retry {attempt}/{attempts} "
                        f"in {interval:.1f}s after {type(exc).__name__}: {exc}"
                    )
                time.sleep(interval)
        raise last_exc

    def _ensure_preview_swapchain_size(self):
        if 0 in self._swapchain_sizes:
            return
        pw = max(640, min(1920, int(self.frame_size[0]) if self.frame_size else 1280))
        ph = max(360, min(1080, int(self.frame_size[1]) if self.frame_size else 720))
        if pw % 2:
            pw += 1
        if ph % 2:
            ph += 1
        self._swapchain_sizes[0] = (pw, ph)

    def _enter_preview_only_wait(self):
        was_waiting = self._waiting_for_headset
        self._preview_only_mode = True
        self._session_running = False
        self._session_ready_pending = False
        self._source_resume_grace_until = 0.0
        self._set_render_active(False)
        self._ensure_preview_swapchain_size()
        if not self._waiting_for_headset:
            if not self._suppress_next_wait_notice:
                print(
                    f"[OpenXRViewer] Waiting for VR headset connect... "
                    f"(stop inference after {self._headset_wait_inference_timeout:.0f}s)"
                )
            self._suppress_next_wait_notice = False
        self._waiting_for_headset = True
        if not was_waiting:
            self._waiting_retry_notice_pending = True
            self._arm_headset_wait_inference_timeout()

    def _defer_openxr_retry(self, delay_s):
        now = time.perf_counter()
        self._openxr_retry_cooldown_until = max(
            self._openxr_retry_cooldown_until,
            now + max(0.0, float(delay_s)),
        )

    def _compute_standby_retry_delay(self, now=None):
        now = time.perf_counter() if now is None else now
        runtime_s = 0.0
        if self._openxr_ready_since > 0.0:
            runtime_s = max(0.0, now - self._openxr_ready_since)
        if runtime_s >= self._openxr_standby_stable_seconds:
            self._openxr_standby_retry_count = 0
            return self._openxr_standby_retry_interval
        self._openxr_standby_retry_count += 1
        return min(
            self._openxr_standby_retry_max_interval,
            self._openxr_standby_retry_interval * (2 ** self._openxr_standby_retry_count),
        )

    def _track_session_idle_render(self, should_render, now=None):
        now = time.perf_counter() if now is None else now
        if should_render:
            if self._session_idle_since > 0.0 and self._session_idle_notice_emitted:
                print("[OpenXRViewer] Headset online : render resumed")
                if self._hard_idle_active or self._headset_wait_inference_paused:
                    self._resume_source_inference()
                    self._source_resume_grace_until = now + self._source_resume_grace
            self._session_idle_since = 0.0
            self._session_idle_notice_emitted = False
            return False
        if self._session_idle_since <= 0.0:
            self._session_idle_since = now
            self._session_idle_notice_emitted = True
            print("[OpenXRViewer] Headset offline: render paused")
        timeout = self._session_idle_render_timeout
        return timeout > 0.0 and (now - self._session_idle_since) >= timeout

    def _debug_openxr_trace(self, message, now=None):
        if not getattr(self, "_openxr_debug", False):
            return
        print(f"[OpenXRViewer][debug] {message}")

    @staticmethod
    def _is_no_headset_error(exc):
        return type(exc).__name__ == "FormFactorUnavailableError"
