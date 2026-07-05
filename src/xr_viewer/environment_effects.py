# Desktop2Stereo OpenXR viewer: screen glow and background effect helpers.

from .implementation import *


class EnvironmentEffectsMixin:
    """Screen-space room effects such as glow and background lighting."""

    _FLAT_FROST_N = 8
    _FLAT_FROST_M = 8
    _FLAT_FROST_STRIDE = 8 * 2 * (8 + 1) + (8 - 1) * 2

    def _screen_effect_basis(self):
        cy = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch)
        sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll)
        sr = math.sin(self.screen_roll)
        rot = np.array([
            [ cy * cr + sy_ * sp * sr, -cy * sr + sy_ * sp * cr,  sy_ * cp, 0],
            [ cp * sr,                  cp * cr,                 -sp,      0],
            [-sy_ * cr + cy * sp * sr,  sy_ * sr + cy * sp * cr,  cy * cp, 0],
            [ 0,                        0,                        0,       1],
        ], dtype=np.float32)
        center = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance], dtype=np.float32)
        return rot, center

    def _screen_effect_model(self, width, height, z_offset=0.0, y_offset=0.0, z_scale=1.0):
        sx = width / 2.0
        sy = height / 2.0
        rot, center = self._screen_effect_basis()
        scale = np.diag([sx, sy, z_scale, 1.0]).astype(np.float32)
        trans = np.eye(4, dtype=np.float32)
        trans[:3, 3] = center
        trans[1, 3] += y_offset
        if z_offset:
            normal = rot[:3, 2]
            trans[0, 3] += float(normal[0]) * z_offset
            trans[1, 3] += float(normal[1]) * z_offset
            trans[2, 3] += float(normal[2]) * z_offset
        return trans @ rot @ scale

    def _frost_front_layout(self):
        head = getattr(self, '_head_pos_w', None)
        if head is None:
            head_key = None
        else:
            head_key = (round(float(head[0]), 2), round(float(head[1]), 2), round(float(head[2]), 2))
        key = (
            round(float(self.screen_width), 6),
            round(float(self.screen_height), 6),
            round(float(self.screen_distance), 6),
            round(float(self.screen_pan_x), 6),
            round(float(self.screen_pan_y), 6),
            round(float(self.screen_yaw), 6),
            round(float(self.screen_pitch), 6),
            round(float(self.screen_roll), 6),
            head_key,
        )
        if key == getattr(self, '_frost_layout_cache_key', None):
            return self._frost_layout_cache_val

        R, center = self._screen_effect_basis()
        head_w = np.array([0.0, 0.0, 0.0], dtype=np.float32) if head is None else np.asarray(head, dtype=np.float32)
        head_local = R[:3, :3].T @ (head_w - center)
        front_depth = max(float(head_local[2]) + 0.55, float(self.screen_distance) + 0.35, 0.75)
        front_half_w = max(float(self.screen_width) * 0.5, abs(float(head_local[0])) + 0.65, 0.65)
        front_half_h = max(float(self.screen_height) * 0.5, abs(float(head_local[1])) + 0.65, 0.65)
        self._frost_layout_cache_key = key
        self._frost_layout_cache_val = (front_depth, front_half_w, front_half_h)
        return self._frost_layout_cache_val

    def _build_flat_frost_verts(self, front_half_w, front_half_h, N=8, M=8):
        sx = max(float(self.screen_width) * 0.5, 1e-6)
        sy = max(float(self.screen_height) * 0.5, 1e-6)
        fx = front_half_w / sx
        fy = front_half_h / sy
        verts = []

        def wall(ax, ay, uva, bx, by, uvb):
            def vtx(s, t):
                px = ax + s * (bx - ax)
                py = ay + s * (by - ay)
                return [
                    px + t * (px * fx - px),
                    py + t * (py * fy - py),
                    t,
                    uva[0] + s * (uvb[0] - uva[0]),
                    uva[1] + s * (uvb[1] - uva[1]),
                ]

            prev_last = None
            for i in range(N):
                t0, t1 = i / N, (i + 1) / N
                row = []
                for j in range(M + 1):
                    s = j / M
                    row.append(vtx(s, t0))
                    row.append(vtx(s, t1))
                if prev_last is not None:
                    verts.extend(prev_last)
                    verts.extend(row[0])
                for vv in row:
                    verts.extend(vv)
                prev_last = row[-1]

        wall(-1, 1, (0, 0), 1, 1, (1, 0))
        wall(-1, -1, (0, 1), 1, -1, (1, 1))
        wall(-1, 1, (0, 0), -1, -1, (0, 1))
        wall(1, 1, (1, 0), 1, -1, (1, 1))
        return np.array(verts, dtype='f4')


    def _default_blank_fast_path(self):
        env_name = str(getattr(self, '_environment_model', '') or '').strip().lower()
        if env_name not in ('default', 'none'):
            return False
        if getattr(self, '_active_environment', None) is not None:
            return False
        if getattr(self, '_env_model_visible', False) and getattr(self, '_env_model_prims', []):
            return False
        return (
            float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0
            and float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)) <= 0.0
        )


    def _should_render_source_screen_effects(self):
        should_show_source_border = getattr(self, '_should_show_source_border', None)
        if not callable(should_show_source_border):
            return True
        if not hasattr(self, '_runtime_direct_source'):
            return True
        return should_show_source_border()

    def _record_screen_effect_safe_age(self, source_tex, safe_frame_id=None):
        if source_tex is None:
            return
        if safe_frame_id is None:
            _tex, _size, safe_frame_id = self._runtime_effect_submit_scheduler().latest_safe()
        safe_frame_id = int(safe_frame_id or 0)
        current_frame = int(getattr(self, '_frame_count', 0) or 0)
        if safe_frame_id > 0 and current_frame >= safe_frame_id:
            age_key = (current_frame, safe_frame_id, int(getattr(source_tex, 'glo', 0) or 0))
            if getattr(self, '_screen_effect_age_record_key', None) == age_key:
                return
            self._screen_effect_age_record_key = age_key
            try:
                self._breakdown_add_value('openxr_effect_ready_age_frames', float(current_frame - safe_frame_id))
            except Exception:
                self._breakdown_inc("openxr_effect_ready_age_record_failed")

    def _screen_effect_source_texture(self, *, allow_runtime_eye=True):
        frame_id = int(getattr(self, '_frame_count', 0) or 0)
        if getattr(self, '_runtime_direct_source', False):
            source_tex, source_size, source_frame_id = self._runtime_effect_submit_scheduler().latest_safe_glow()
            value = (source_tex, source_size)
            cache_key = (
                frame_id,
                int(source_frame_id or 0),
                int(getattr(source_tex, 'glo', 0) or 0) if source_tex is not None else 0,
                tuple(value[1]) if value[1] is not None else None,
            )
            if getattr(self, '_screen_effect_source_cache_key', None) == cache_key:
                self._breakdown_inc("openxr_screen_effect_source_reuse")
                return getattr(self, '_screen_effect_source_cache_value', value)
            self._record_screen_effect_safe_age(source_tex, source_frame_id)
            self._screen_effect_source_cache_key = cache_key
            self._screen_effect_source_cache_frame = frame_id
            self._screen_effect_source_cache_value = value
            return value
        source_tex = getattr(self, 'color_tex', None)
        source_size = getattr(self, '_texture_size', None)
        cache_key = (
            frame_id,
            int(getattr(source_tex, 'glo', 0) or 0) if source_tex is not None else 0,
            tuple(source_size) if source_size is not None else None,
        )
        if getattr(self, '_screen_effect_source_cache_key', None) == cache_key:
            self._breakdown_inc("openxr_screen_effect_source_reuse")
            return getattr(self, '_screen_effect_source_cache_value', (source_tex, source_size))
        value = (source_tex, source_size)
        self._screen_effect_source_cache_key = cache_key
        self._screen_effect_source_cache_frame = frame_id
        self._screen_effect_source_cache_value = value
        return value


    def _render_glow(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_glow_intensity', 0.0)) * float(getattr(self, '_glow_intensity_multiplier', 0.0))
        if intensity <= 0.0 or self.screen_height is None:
            return
        if getattr(self, '_glow_prog', None) is None or getattr(self, '_glow_vao', None) is None:
            return

        source_tex, source_size = self._screen_effect_source_texture(allow_runtime_eye=False)
        screen_long = max(self.screen_width, self.screen_height)
        glow_scale = screen_long / max(float(getattr(self, '_glow_ref_screen', 2.4)), 1e-6)
        glow_width = float(getattr(self, '_glow_width_m', 0.50)) * glow_scale
        glow_extent = glow_width * 2.4
        glow_margin = glow_extent
        if (
            str(getattr(self, '_environment_model', '') or '').strip().lower() == 'default'
            and getattr(self, '_active_environment', None) is None
        ):
            surround_margin = max(
                float(getattr(self, '_glow_surround_margin_m', 14.0)),
                max(self.screen_width, self.screen_height) * 1.15,
            )
            glow_margin = max(glow_margin, surround_margin)
            glow_extent = max(glow_extent, glow_margin * 0.92)
        glow_w = self.screen_width + 2.0 * glow_margin
        glow_h = self.screen_height + 2.0 * glow_margin
        inner_w = self.screen_width / glow_w
        inner_h = self.screen_height / glow_h
        uv_scale = max(glow_w, glow_h, 1e-6)
        uv_glow_width = glow_width / uv_scale
        uv_glow_extent = glow_extent / uv_scale

        previous_depth_mask = self.ctx.depth_mask
        try:
            self.ctx.depth_mask = False
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE
            model = self._screen_effect_model(glow_w, glow_h, z_offset=self._screen_back_offset(0.08))
            mvp = vp_mat @ model
            self._glow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
            self._glow_prog['u_screen_half'].value = (inner_w * 0.5, inner_h * 0.5)
            glow_color = tuple(getattr(self, '_glow_color', (0.30, 0.55, 1.0)))
            self._glow_prog['u_glow_color'].value = glow_color
            use_source_tex = 1 if source_tex is not None else 0
            self._glow_prog['u_glow_use_tex'].value = use_source_tex
            if source_tex is not None:
                source_tex.use(location=0)
            self._glow_prog['u_glow_width'].value = max(uv_glow_width, 1e-6)
            self._glow_prog['u_glow_extent'].value = max(uv_glow_extent, 1e-6)
            self._glow_prog['u_glow_intensity'].value = intensity
            self._glow_vao.render(moderngl.TRIANGLE_STRIP)
        except Exception as exc:
            print(f"[OpenXRViewer] Screen glow render failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_screen_glow_failed")
        finally:
            self.ctx.disable(moderngl.BLEND)
            self.ctx.depth_mask = previous_depth_mask
            self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_frosted_glow(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_frosted_glow_intensity', 0.0))
        intensity *= max(
            float(getattr(self, '_glow_intensity_multiplier', 0.0)),
            float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)),
        )
        if intensity <= 0.0 or self.screen_height is None:
            return
        if getattr(self, '_frosted_glow_prog', None) is None or getattr(self, '_frosted_glow_vao', None) is None:
            return
        source_tex, _source_size = self._screen_effect_source_texture(allow_runtime_eye=False)
        if source_tex is None:
            return

        front_depth, front_half_w, front_half_h = self._frost_front_layout()
        params = (
            round(float(self.screen_width), 6),
            round(float(self.screen_height), 6),
            round(float(front_half_w), 2),
            round(float(front_half_h), 2),
        )
        if params != getattr(self, '_frosted_glow_verts_params', None):
            self._frosted_veil_vbo.write(self._build_flat_frost_verts(front_half_w, front_half_h).tobytes())
            self._frosted_glow_verts_params = params

        previous_depth_mask = self.ctx.depth_mask
        try:
            self.ctx.depth_mask = False
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            model = self._screen_effect_model(self.screen_width, self.screen_height, z_scale=front_depth)
            source_tex.use(location=0)
            self._frosted_glow_prog['u_model'].write(model.T.astype('f4').tobytes())
            self._frosted_glow_prog['u_vp'].write(vp_mat.T.astype('f4').tobytes())
            self._frosted_glow_prog['u_edge_inset'].value = float(getattr(self, '_frosted_glow_inset', 0.045))
            self._frosted_glow_prog['u_lod'].value = float(getattr(self, '_frosted_glow_lod', 5.4))
            self._frosted_glow_prog['u_threshold'].value = float(getattr(self, '_frosted_glow_threshold', 0.46))
            self._frosted_glow_prog['u_intensity'].value = intensity
            self._frosted_glow_prog['u_frost_alpha'].value = float(getattr(self, '_frosted_glow_alpha', 0.42))
            self._frosted_glow_prog['u_noise_scale'].value = 54.0
            self._frosted_glow_prog['u_beam_softness'].value = 0.34
            self._frosted_glow_prog['u_frost_blend'].value = float(getattr(self, '_frosted_glow_blend', 1.35))
            self._frosted_glow_prog['u_beam_thickness'].value = float(getattr(self, '_frosted_glow_thickness', 1.6))
            self._frosted_glow_prog['u_diffuse_scatter'].value = float(getattr(self, '_frosted_glow_diffuse', 0.85))
            self._frosted_glow_prog['u_time'].value = float(time.time())
            self._frosted_glow_prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)
            stride = int(getattr(self, '_FLAT_FROST_STRIDE', 158))
            for wall_idx in range(4):
                self._frosted_glow_vao.render(moderngl.TRIANGLE_STRIP, vertices=stride, first=wall_idx * stride)
        except Exception as exc:
            print(f"[OpenXRViewer] Frosted glow render failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_frosted_glow_failed")
        finally:
            self.ctx.disable(moderngl.BLEND)
            self.ctx.depth_mask = previous_depth_mask
            self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_frosted_veil(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_frosted_veil_intensity', 1.0))
        intensity *= max(
            float(getattr(self, '_glow_intensity_multiplier', 0.0)),
            float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)),
        )
        if intensity <= 0.0 or self.screen_height is None:
            return
        if (
            getattr(self, '_frosted_veil_prog', None) is None
            or getattr(self, '_frosted_veil_vao', None) is None
            or getattr(self, '_frosted_veil_vbo', None) is None
        ):
            return

        source_tex, _source_size = self._screen_effect_source_texture(allow_runtime_eye=False)
        if source_tex is None:
            return
        if mgl_fbo is not None:
            mgl_fbo.use()

        front_depth, front_half_w, front_half_h = self._frost_front_layout()
        params = (
            round(float(self.screen_width), 6),
            round(float(self.screen_height), 6),
            round(float(front_half_w), 2),
            round(float(front_half_h), 2),
        )
        if params != getattr(self, '_frosted_veil_verts_params', None):
            self._frosted_veil_vbo.write(self._build_flat_frost_verts(front_half_w, front_half_h).tobytes())
            self._frosted_veil_verts_params = params

        previous_depth_mask = self.ctx.depth_mask
        try:
            self.ctx.depth_mask = False
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            model = self._screen_effect_model(self.screen_width, self.screen_height, z_scale=front_depth)
            source_tex.use(location=0)
            self._frosted_veil_prog['u_model'].write(model.T.astype('f4').tobytes())
            self._frosted_veil_prog['u_vp'].write(vp_mat.T.astype('f4').tobytes())
            self._frosted_veil_prog['u_edge_inset'].value = 0.02
            self._frosted_veil_prog['u_intensity'].value = intensity
            self._frosted_veil_prog['u_frost_alpha'].value = float(getattr(self, '_frosted_veil_alpha', 0.58))
            self._frosted_veil_prog['u_beam_softness'].value = 0.34
            self._frosted_veil_prog['u_beam_thickness'].value = 3.0
            self._frosted_veil_prog['u_source_crop'].value = (0.0, 0.0, 1.0, 1.0)
            stride = int(getattr(self, '_FLAT_FROST_STRIDE', 158))
            for wall_idx in range(4):
                self._frosted_veil_vao.render(moderngl.TRIANGLE_STRIP, vertices=stride, first=wall_idx * stride)
        except Exception as exc:
            print(f"[OpenXRViewer] Frosted veil render failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_frosted_veil_failed")
        finally:
            self.ctx.disable(moderngl.BLEND)
            self.ctx.depth_mask = previous_depth_mask
            self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_glow_shell(self, mgl_fbo, vp_mat, intensity_multiplier=None):
        if intensity_multiplier is None:
            intensity_multiplier = float(getattr(self, '_glow_shell_intensity_multiplier', 0.0))
        intensity = float(getattr(self, '_glow_intensity', 0.0)) * float(intensity_multiplier)
        if intensity <= 0.0:
            return
        if getattr(self, '_glow_shell_prog', None) is None or getattr(self, '_glow_shell_vao', None) is None:
            return

        source_tex, source_size = self._screen_effect_source_texture(allow_runtime_eye=False)
        if getattr(self, '_runtime_direct_source', False):
            glow_tex, _glow_size, _glow_frame_id = self._runtime_effect_submit_scheduler().latest_safe_downsample(
                cached_downsample=getattr(self, '_cached_glow_downsample_texture', None)
            )
        else:
            glow_tex = self._prepare_glow_downsample_texture(source_tex, source_size)
        if mgl_fbo is not None:
            mgl_fbo.use()
        center = getattr(self, '_head_pos_w', None)
        if center is None:
            center = (self.screen_pan_x, self.screen_pan_y, -self.screen_distance * 0.55)
        radius = max(float(getattr(self, '_glow_shell_radius_m', 18.0)), max(self.screen_width, self.screen_height, 1.0) * 0.85)
        height = max(float(getattr(self, '_glow_shell_height_m', 8.5)), self.screen_height * 1.8)

        previous_depth_mask = self.ctx.depth_mask
        try:
            self.ctx.depth_mask = False
            self.ctx.disable(moderngl.DEPTH_TEST)
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
            self._glow_shell_prog['u_vp'].write(vp_mat.T.astype('f4').tobytes())
            self._glow_shell_prog['u_center'].value = tuple(float(v) for v in center)
            self._glow_shell_prog['u_shell_scale'].value = (radius, height * 0.5, radius)
            self._glow_shell_prog['u_glow_color'].value = tuple(getattr(self, '_glow_color', (0.30, 0.55, 1.0)))
            use_source_tex = 1 if glow_tex is not None else 0
            self._glow_shell_prog['u_glow_use_tex'].value = use_source_tex
            if glow_tex is not None:
                glow_tex.use(location=0)
            self._glow_shell_prog['u_glow_intensity'].value = intensity
            self._glow_shell_vao.render(moderngl.TRIANGLE_STRIP)
        except Exception as exc:
            print(f"[OpenXRViewer] Glow shell render failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_glow_shell_failed")
        finally:
            self.ctx.disable(moderngl.BLEND)
            self.ctx.depth_mask = previous_depth_mask
            self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_screen_background_effects(self, mgl_fbo, vp_mat):
        try:
            if not self._should_render_source_screen_effects():
                return
            if self._default_blank_fast_path():
                return
            mode = str(getattr(self, '_glow_mode', 'screen') or 'screen').strip().lower()
            if mode == 'off':
                return
            if (
                float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0
                and float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)) <= 0.0
            ):
                return
            env_active = bool(getattr(self, '_env_model_visible', False) and getattr(self, '_env_model_prims', []))
            passthrough_active = getattr(self, '_bg_color_idx', 0) == 1
            if not env_active and not passthrough_active:
                if mode == 'surround':
                    self._render_glow_shell(mgl_fbo, vp_mat)
                elif mode == 'screen':
                    screen_mult = float(getattr(self, '_glow_intensity_multiplier', 0.0))
                    self._render_glow_shell(mgl_fbo, vp_mat, intensity_multiplier=screen_mult * 0.72)
        except Exception as exc:
            print(f"[OpenXRViewer] Screen background effect failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_screen_background_effect_failed")

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat):
        try:
            if not self._should_render_source_screen_effects():
                return
            mode = str(getattr(self, '_glow_mode', 'screen') or 'screen').strip().lower()
            if mode == 'off':
                return
            if (
                float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0
                and float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)) <= 0.0
            ):
                return
            if mode == 'veil':
                self._render_frosted_veil(mgl_fbo, vp_mat)
            elif mode == 'frosted':
                self._render_frosted_glow(mgl_fbo, vp_mat)
        except Exception as exc:
            print(f"[OpenXRViewer] Screen foreground effect failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_screen_foreground_effect_failed")
