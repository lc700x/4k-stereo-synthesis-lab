# Desktop2Stereo OpenXR viewer: screen glow and background effect helpers.

from .implementation import *


class EnvironmentEffectsMixin:
    """Screen-space room effects such as glow and background lighting."""

    def _screen_effect_model(self, width, height, z_offset=0.0, y_offset=0.0):
        sx = width / 2.0
        sy = height / 2.0
        cy = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch)
        sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll)
        sr = math.sin(self.screen_roll)
        rot_y = np.array([[cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
        rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
        rot_z = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
        scale = np.diag([sx, sy, 1.0, 1.0]).astype('f4')
        trans = np.eye(4, dtype='f4')
        trans[0, 3] = self.screen_pan_x
        trans[1, 3] = self.screen_pan_y + y_offset
        trans[2, 3] = -self.screen_distance + z_offset
        return trans @ rot_y @ rot_x @ rot_z @ scale


    def _default_blank_fast_path(self):
        env_name = str(getattr(self, '_environment_model', '') or '').strip().lower()
        if env_name not in ('default', 'none'):
            return False
        if getattr(self, '_active_environment', None) is not None:
            return False
        if getattr(self, '_env_model_visible', False) and getattr(self, '_env_model_prims', []):
            return False
        return float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0


    def _render_glow(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_glow_intensity', 0.0)) * float(getattr(self, '_glow_intensity_multiplier', 0.0))
        if intensity <= 0.0 or self.screen_height is None:
            return
        if getattr(self, '_glow_prog', None) is None or getattr(self, '_glow_vao', None) is None:
            return

        self._advance_glow_color()
        screen_long = max(self.screen_width, self.screen_height)
        glow_scale = screen_long / max(float(getattr(self, '_glow_ref_screen', 2.4)), 1e-6)
        glow_width = float(getattr(self, '_glow_width_m', 0.50)) * glow_scale
        glow_extent = glow_width * 2.4
        glow_margin = glow_extent
        glow_w = self.screen_width + 2.0 * glow_margin
        glow_h = self.screen_height + 2.0 * glow_margin
        inner_w = self.screen_width / glow_w
        inner_h = self.screen_height / glow_h
        uv_scale = max(glow_w, glow_h, 1e-6)
        uv_glow_width = glow_width / uv_scale
        uv_glow_extent = glow_extent / uv_scale

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        model = self._screen_effect_model(glow_w, glow_h, z_offset=-self._screen_back_offset(1.0))
        mvp = vp_mat @ model
        self._glow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._glow_prog['u_screen_half'].value = (inner_w * 0.5, inner_h * 0.5)
        self._glow_prog['u_glow_color'].value = tuple(getattr(self, '_glow_color', (0.30, 0.55, 1.0)))
        self._glow_prog['u_glow_width'].value = max(uv_glow_width, 1e-6)
        self._glow_prog['u_glow_extent'].value = max(uv_glow_extent, 1e-6)
        self._glow_prog['u_glow_intensity'].value = intensity
        self._glow_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_screen_background_effects(self, mgl_fbo, vp_mat):
        if self._default_blank_fast_path():
            return
        env_active = bool(getattr(self, '_env_model_visible', False) and getattr(self, '_env_model_prims', []))
        passthrough_active = getattr(self, '_bg_color_idx', 0) == 1
        if not env_active and not passthrough_active:
            self._render_glow(mgl_fbo, vp_mat)

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat):
        return None