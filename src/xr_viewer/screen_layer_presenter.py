import ctypes
import time

import moderngl
from OpenGL.GL import GL_CCW, glFrontFace

from .background_presenter import BackgroundPresenter

try:
    import xr
except ImportError:
    xr = None


class ScreenLayerPresenter:
    def __init__(self, viewer):
        self.viewer = viewer
        self._frame_projection_layer = None
        self._frame_quad_layers = []

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

    def projection_screen_needed(self):
        return not self.viewer._quad_layer_screen_presentable()

    def projection_screen_unavailable_reason(self):
        if not self.projection_screen_needed():
            return None
        return self.viewer._quad_layer_unavailable_reason()

    def projection_screen_source_ready(self, eye_index):
        viewer = self.viewer
        if not self.projection_screen_needed():
            return True
        if viewer._runtime_direct_source:
            return viewer._runtime_eye_textures[eye_index] is not None
        return getattr(viewer, 'color_tex', None) is not None and getattr(viewer, 'depth_tex', None) is not None

    def projection_screen_effects_enabled(self):
        if not self.projection_screen_needed():
            return False
        viewer = self.viewer
        if not getattr(viewer, '_screen_effects_enabled', True):
            return False
        if getattr(viewer, 'screen_height', None) is None:
            return False
        should_render = getattr(viewer, '_should_render_source_screen_effects', None)
        return bool(should_render()) if callable(should_render) else True

    def render_projection_screen(self, *, eye_index, mgl_fbo, vp_mat, swapchain_size, mark_perf=None):
        viewer = self.viewer
        sc_w, sc_h = swapchain_size
        draw_projection_screen_effects = bool(viewer._openxr_projection_screen_effects_enabled)
        quad_unavailable_reason = viewer._openxr_projection_screen_unavailable_reason or 'unknown'
        viewer._breakdown_inc(f"openxr_quad_unavailable_{quad_unavailable_reason}")
        if draw_projection_screen_effects:
            viewer._render_screen_background_effects(mgl_fbo, vp_mat)
        if mark_perf:
            mark_perf('bgfx')

        if viewer._screen_curved:
            viewer._render_border(mgl_fbo, vp_mat)
        if mark_perf:
            mark_perf('pre_border')

        mgl_fbo.use()
        eye_sign = -1.0 if eye_index == 0 else 1.0
        runtime_rgb_depth_max_disparity_px = (
            0.0 if viewer._runtime_direct_source else float(getattr(viewer, '_runtime_rgb_depth_max_disparity_px', 0.0))
        )
        runtime_rgb_depth_render_width = (
            0 if viewer._runtime_direct_source else int(getattr(viewer, '_runtime_rgb_depth_render_width', 0) or 0)
        )
        if runtime_rgb_depth_render_width <= 0:
            source_size = viewer._texture_size or (0, 0)
            runtime_rgb_depth_render_width = int(source_size[0] or 0)
        screen_disparity_uv = 0.0
        if not viewer._runtime_direct_source and runtime_rgb_depth_render_width > 0:
            screen_disparity_uv = max(0.0, runtime_rgb_depth_max_disparity_px) / float(runtime_rgb_depth_render_width)
        screen_depth_strength = (
            0.0
            if viewer._runtime_direct_source
            else max(0.0, float(getattr(viewer, '_runtime_rgb_depth_depth_strength', viewer.depth_strength) or 0.0))
        )
        screen_eye_offset = 0.0 if viewer._runtime_direct_source else eye_sign * screen_disparity_uv / 2.0
        model = viewer._build_model_mat4()
        mvp = vp_mat @ model
        if viewer._runtime_direct_source:
            viewer._log_screen_footprint_once(eye_index, mvp, (sc_w, sc_h))
            source_tex = viewer._runtime_eye_textures[eye_index]
            screen_tex = viewer._prepare_screen_quality_texture(
                source_tex,
                viewer._runtime_eye_texture_size or viewer._texture_size,
                mvp,
                (sc_w, sc_h),
                f'runtime_eye_{eye_index}',
            ) or source_tex
            screen_depth_tex = viewer._runtime_depth_texture
        else:
            screen_tex = viewer._prepare_screen_quality_texture(
                viewer.color_tex,
                viewer._texture_size,
                mvp,
                (sc_w, sc_h),
                'color',
            ) or viewer.color_tex
            screen_depth_tex = viewer.depth_tex
        if mark_perf:
            mark_perf('quality')

        mgl_fbo.use()
        viewer.ctx.viewport = (0, 0, sc_w, sc_h)
        viewer.ctx.enable(moderngl.DEPTH_TEST)
        viewer.ctx.depth_mask = True
        viewer.ctx.disable(moderngl.BLEND)
        viewer.ctx.disable(moderngl.CULL_FACE)
        glFrontFace(GL_CCW)

        screen_source_size = (
            (viewer._runtime_eye_texture_size or viewer._texture_size)
            if viewer._runtime_direct_source else viewer._texture_size
        )
        screen_source_size = screen_source_size or (sc_w, sc_h)
        shader_resolution_mode = str(getattr(viewer, '_openxr_rgb_depth_shader_resolution', 'source') or 'source')
        if viewer._runtime_direct_source or shader_resolution_mode == 'source':
            shader_resolution = (float(screen_source_size[0]), float(screen_source_size[1]))
        elif shader_resolution_mode == 'swapchain':
            shader_resolution = (float(sc_w), float(sc_h))
        else:
            shader_resolution = None
        if not viewer._runtime_direct_source and not getattr(viewer, '_openxr_rgb_depth_shader_resolution_logged', False):
            print(
                "[OpenXRViewer] rgb_depth shader:"
                f" resolution_mode={shader_resolution_mode}"
                f" resolution={shader_resolution if shader_resolution is not None else 'unset'}"
                f" feather={int(bool(getattr(viewer, '_openxr_rgb_depth_feather', False)))}",
                f" max_disparity_px={runtime_rgb_depth_max_disparity_px:.3f}"
                f" render_width={runtime_rgb_depth_render_width}"
                f" disparity_uv={screen_disparity_uv:.6f}"
                f" eye_offset_abs={abs(screen_eye_offset):.6f}"
                f" depth_strength={screen_depth_strength:.6f}"
                f" convergence={float(viewer.convergence):.6f}",
                flush=True,
            )
            viewer._openxr_rgb_depth_shader_resolution_logged = True

        if viewer._screen_curved and viewer._curved_prog is not None:
            if screen_depth_tex is None:
                viewer.ctx.screen.use()
                return False
            screen_tex.use(location=0)
            screen_depth_tex.use(location=1)
            params = (
                viewer.screen_width,
                viewer.screen_height,
                viewer.screen_distance,
                viewer.screen_pan_x,
                viewer.screen_pan_y,
                viewer.screen_yaw,
                viewer.screen_pitch,
                viewer.screen_roll,
            )
            if viewer._curved_verts_params != params:
                arc_verts = viewer._build_curved_screen_verts()
                viewer._curved_vbo.write(arc_verts.tobytes())
                viewer._curved_verts_params = params
            viewer._curved_prog['u_mvp'].write(vp_mat.T.tobytes())
            runtime_rgb_depth = not viewer._runtime_direct_source
            feather_enabled = bool(runtime_rgb_depth and viewer._openxr_rgb_depth_feather)
            if shader_resolution is not None:
                viewer._curved_prog['u_resolution'].value = shader_resolution
            viewer._curved_prog['u_feather_enabled'].value = 1 if feather_enabled else 0
            viewer._curved_prog['u_feather_width'].value = 0.02 if feather_enabled else 0.0
            viewer._curved_prog['u_viewport'].value = (0.0, 0.0, float(sc_w), float(sc_h))
            viewer._curved_prog['u_roll'].value = 0.0 if viewer._runtime_direct_source else viewer.screen_roll
            viewer._curved_prog['u_eye_offset'].value = screen_eye_offset
            viewer._curved_prog['u_depth_strength'].value = screen_depth_strength
            viewer._curved_prog['u_convergence'].value = float(viewer.convergence)
            viewer._curved_prog['u_corner_radius'].value = viewer._corner_radius
            viewer._curved_vao.render(moderngl.TRIANGLE_STRIP, vertices=(48 + 1) * 2)
        else:
            if screen_depth_tex is None:
                viewer.ctx.screen.use()
                return False
            screen_tex.use(location=0)
            screen_depth_tex.use(location=1)
            viewer.prog['u_mvp'].write(mvp.T.tobytes())
            runtime_rgb_depth = not viewer._runtime_direct_source
            feather_enabled = bool(runtime_rgb_depth and viewer._openxr_rgb_depth_feather)
            if shader_resolution is not None:
                viewer.prog['u_resolution'].value = shader_resolution
            viewer.prog['u_feather_enabled'].value = 1 if feather_enabled else 0
            viewer.prog['u_feather_width'].value = 0.02 if feather_enabled else 0.0
            viewer.prog['u_viewport'].value = (0.0, 0.0, float(sc_w), float(sc_h))
            viewer.prog['u_roll'].value = 0.0 if viewer._runtime_direct_source else viewer.screen_roll
            viewer.prog['u_eye_offset'].value = screen_eye_offset
            viewer.prog['u_depth_strength'].value = screen_depth_strength
            viewer.prog['u_convergence'].value = float(viewer.convergence)
            viewer.prog['u_corner_radius'].value = viewer._corner_radius
            viewer.quad_vao.render(moderngl.TRIANGLE_STRIP)
        if mark_perf:
            mark_perf('screen')

        if not viewer._screen_curved:
            viewer._render_border(mgl_fbo, vp_mat)
        if mark_perf:
            mark_perf('border')

        if draw_projection_screen_effects:
            viewer._render_screen_foreground_effects(mgl_fbo, vp_mat)
        if mark_perf:
            mark_perf('fgfx')
        return True

    def render_quad_screen_overlay(self, *, mgl_fbo, vp_mat, mark_perf=None):
        viewer = self.viewer
        viewer._breakdown_inc("openxr_projection_screen_skipped")
        if mark_perf:
            mark_perf('screen_quad_layer')
        if not viewer._screen_curved:
            viewer._render_border(mgl_fbo, vp_mat)
        if mark_perf:
            mark_perf('border')

    def projection_layer_needed(self):
        viewer = self.viewer
        if self.projection_screen_needed():
            return True
        background_presenter = getattr(viewer, '_background_presenter', None)
        if background_presenter is None:
            background_presenter = BackgroundPresenter(viewer)
            viewer._background_presenter = background_presenter
        if background_presenter.projection_fallback_needed():
            return True
        if viewer._keyboard_visible and viewer._keyboard_tex is not None:
            return True
        if viewer._aim_mat_l is not None or viewer._aim_mat_r is not None:
            return True
        if viewer._grip_mat_l is not None or viewer._grip_mat_r is not None:
            return True
        if float(getattr(viewer, '_border_alpha', 0.0) or 0.0) > 0.0:
            return True
        if any(getattr(viewer, name, None) is not None for name in (
            '_depth_osd_tex', '_screen_osd_tex', '_preset_osd_tex', '_seat_adjust_osd_tex'
        )):
            return True
        if viewer._brand_osd_tex is not None and viewer._grip_mat_r is not None:
            return True
        if viewer._hand_fps_visible and viewer._overlay_tex is not None:
            return True
        if viewer._team_fps_visible and viewer._team_status_tex is not None:
            return True
        if viewer._calibration_mode:
            return True
        if viewer._fps_overlay_visible and viewer._help_tex is not None:
            return True
        if viewer._team_status_visible and viewer._team_help_visible and viewer._team_help_tex is not None:
            return True
        return False

    def prepare_projection_frame_state(self):
        self.viewer._openxr_draw_projection_screen = self.projection_screen_needed()
        self.viewer._openxr_projection_screen_unavailable_reason = self.projection_screen_unavailable_reason()
        self.viewer._openxr_projection_screen_source_ready = tuple(
            self.projection_screen_source_ready(eye_index) for eye_index in range(2)
        )
        self.viewer._openxr_projection_screen_effects_enabled = self.projection_screen_effects_enabled()

    def prepare_frame_layers(self, *, screen_frame_uploaded=False):
        self._frame_projection_layer = None
        self._frame_quad_layers = []
        updated_quad_eyes = self.update_or_reuse(screen_frame_uploaded=screen_frame_uploaded)
        quad_layers, quad_layer_headers, updated_quad_eyes = self.make_quad_layers(updated_quad_eyes)
        self._frame_quad_layers = quad_layers
        self.prepare_projection_frame_state()
        render_projection_layer = self.projection_layer_needed()
        if not render_projection_layer:
            self.viewer._breakdown_inc('openxr_projection_layer_skipped')
        return quad_layers, quad_layer_headers, updated_quad_eyes, render_projection_layer

    def append_frame_layers(self, composition_layers, *, projection_views=(), projection_space=None, quad_layer_headers=()):
        if projection_views:
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
                viewer._xr_quad_layer_active = False
                viewer._xr_quad_layer_failed = True
                viewer._breakdown_inc('openxr_quad_layer_failed')
                print(f"[OpenXRViewer] Quad layer build failed: {type(exc).__name__}: {exc}")
                return [], [], []
        return quad_layers, quad_layer_headers, list(updated_quad_eyes)
