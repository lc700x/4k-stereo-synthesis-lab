import moderngl
from OpenGL.GL import GL_CCW, glFrontFace


class OverlayLayerPresenter:
    def __init__(self, viewer):
        self.viewer = viewer

    def render_projection_overlays(self, *, eye_index, mgl_fbo, vp_mat, view_mat, swapchain_size, mark_perf=None):
        viewer = self.viewer
        sc_w, sc_h = swapchain_size

        def _try_aux_render(metric, label, callback):
            try:
                callback()
                return True
            except Exception as exc:
                viewer._breakdown_inc(metric)
                print(f"[OpenXRViewer] {label} render failed: {type(exc).__name__}: {exc}")
                mgl_fbo.use()
                viewer.ctx.viewport = (0, 0, sc_w, sc_h)
                viewer.ctx.disable(moderngl.BLEND)
                viewer.ctx.depth_mask = True
                viewer.ctx.enable(moderngl.DEPTH_TEST)
                return False

        if viewer._keyboard_visible and viewer._keyboard_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'keyboard', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_keyboard(mgl_fbo, vp_mat),
            ))
        if mark_perf:
            mark_perf('keyboard')

        if viewer._depth_osd_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'depth OSD', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_depth_osd(eye_index, mgl_fbo, vp_mat),
            ))
        if viewer._screen_osd_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'screen OSD', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_screen_osd(eye_index, mgl_fbo, vp_mat),
            ))
        if viewer._preset_osd_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'preset OSD', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_preset_osd(eye_index, mgl_fbo, vp_mat),
            ))
        if viewer._brand_osd_tex is not None and viewer._grip_mat_r is not None:
            _try_aux_render('openxr_overlay_render_failed', 'brand OSD', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_brand_osd(eye_index, mgl_fbo, vp_mat),
            ))
        if viewer._seat_adjust_osd_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'seat OSD', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_seat_adjust_osd(eye_index, mgl_fbo, vp_mat),
            ))

        _try_aux_render('openxr_laser_render_failed', 'laser beam', lambda: (
            setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
            viewer._render_lasers(mgl_fbo, vp_mat, blend=False),
        ))

        if viewer._ctrl_prims_l or viewer._ctrl_prims_r:
            _try_aux_render('openxr_controller_render_failed', 'controller', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer.ctx.disable(moderngl.BLEND),
                setattr(viewer.ctx, 'depth_mask', True),
                glFrontFace(GL_CCW),
                viewer._render_controllers(mgl_fbo, vp_mat, view_mat),
            ))

        _try_aux_render('openxr_laser_render_failed', 'laser hit circle', lambda: (
            setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
            viewer.ctx.disable(moderngl.DEPTH_TEST),
            setattr(viewer.ctx, 'depth_mask', False),
            viewer.ctx.enable(moderngl.BLEND),
            setattr(viewer.ctx, 'blend_func', (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)),
            viewer._render_lasers(mgl_fbo, vp_mat, blend=True),
        ))
        viewer.ctx.disable(moderngl.BLEND)
        viewer.ctx.depth_mask = True
        viewer.ctx.enable(moderngl.DEPTH_TEST)

        if viewer._hand_fps_visible and viewer._overlay_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'FPS overlay', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_fps_overlay(eye_index, mgl_fbo, vp_mat),
            ))
        if viewer._team_fps_visible and viewer._team_status_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'team status overlay', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_team_status_overlay(eye_index, mgl_fbo, vp_mat),
            ))
        if viewer._calibration_mode:
            _try_aux_render('openxr_overlay_render_failed', 'calibration panel', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_calibration_panel(mgl_fbo, vp_mat),
            ))
        if viewer._fps_overlay_visible and viewer._help_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'help panel', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_help_panel(mgl_fbo, vp_mat),
            ))
        if viewer._team_status_visible and viewer._team_help_visible and viewer._team_help_tex is not None:
            _try_aux_render('openxr_overlay_render_failed', 'team help panel', lambda: (
                setattr(viewer.ctx, 'viewport', (0, 0, sc_w, sc_h)),
                viewer._render_team_help_panel(mgl_fbo, vp_mat),
            ))
        if mark_perf:
            mark_perf('osd')
