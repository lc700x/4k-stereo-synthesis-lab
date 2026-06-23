# Desktop2Stereo OpenXR viewer: virtual keyboard texture, placement, and rendering.

import math

import moderngl
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .keyboard_layout import _KB_ROWS, _KB_TEX_H, _KB_TEX_W, _KB_UNITS_WIDE, _KeyEntry


class CoreKeyboardMixin:
    """Virtual keyboard layout texture, sizing, world transform, and rendering."""

    def _build_keyboard_texture(self):
        """(Re)build the virtual keyboard texture with the current shift state.

        When Shift or Caps Lock is active the number/symbol keys show their
        shifted glyph (e.g. '!' instead of '1').  Modifier key backgrounds
        are highlighted as before.
        """
        TW, TH   = _KB_TEX_W, _KB_TEX_H
        ROW_H    = TH / len(_KB_ROWS)
        UNIT_W   = TW / float(_KB_UNITS_WIDE)
        UNIT_M   = self._keyboard_width / float(_KB_UNITS_WIDE)
        PAD      = 3
        show_s   = self._kb_show_shifted   # whether to render shifted labels

        img  = Image.new('RGBA', (TW, TH), (30, 30, 35, 230))
        draw = ImageDraw.Draw(img)

        fnt = None
        for candidate in (r"C:\Windows\Fonts\seguisym.ttf",
                        r"C:\Windows\Fonts\segoeui.ttf",
                        self.font_type):
            if not candidate:
                continue
            try:
                fnt = ImageFont.truetype(candidate, 16)
                break
            except Exception:
                continue

        self._keyboard_keys = []
        kw_half  = self._keyboard_width  / 2.0
        kh_half  = self._keyboard_height / 2.0
        row_h_m  = self._keyboard_height / len(_KB_ROWS)

        for row_i, row in enumerate(_KB_ROWS):
            py0 = int(row_i * ROW_H)
            py1 = int((row_i + 1) * ROW_H)
            ly1 = kh_half - row_i * row_h_m
            ly0 = ly1 - row_h_m
            px  = 0.0
            lx  = -kw_half
            for (label, vk_normal, shifted_label, vk_shifted, width_units) in row:
                px_end  = px + width_units * UNIT_W
                lx_end  = lx + width_units * UNIT_M

                if vk_normal == -1:
                    px = px_end
                    lx = lx_end
                    continue

                # Key background
                draw.rectangle([px + PAD, py0 + PAD, px_end - PAD, py1 - PAD],
                            fill=(60, 62, 70, 255), outline=(130, 132, 140, 255))

                # Pick label: shifted version if available and shift is active
                display_label = label
                if show_s and shifted_label is not None:
                    display_label = shifted_label

                if fnt:
                    tx = (px + px_end) / 2
                    ty = (py0 + py1) / 2
                    draw.text((tx, ty), display_label, font=fnt,
                            fill=(220, 220, 225, 255), anchor='mm')
                else:
                    draw.text((int(px + PAD + 2), int(py0 + PAD + 2)),
                            display_label, fill=(220, 220, 225, 255))

                uv_rect   = (px / TW, py0 / TH, px_end / TW, py1 / TH)
                loc_rect  = (lx, ly0, lx_end, ly1)

                self._keyboard_keys.append(_KeyEntry(
                    label=label,
                    shifted_label=shifted_label,
                    vk=vk_normal,
                    shifted_vk=vk_shifted if vk_shifted is not None else vk_normal,
                    rect_uv=uv_rect,
                    rect_local=loc_rect,
                ))

                px  = px_end
                lx  = lx_end

        # Upload
        tex_data = np.flipud(np.array(img, dtype=np.uint8))
        if self._keyboard_tex is not None:
            self._keyboard_tex.release()
        self._keyboard_tex = self.ctx.texture((TW, TH), 4, dtype='f1')
        self._keyboard_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._keyboard_tex.write(tex_data.tobytes())

        # VAO (first build only -geometry never changes)
        if self._keyboard_vao is None:
            verts = np.array([-1,-1,0,0, 1,-1,1,0, -1,1,0,1, 1,1,1,1], dtype='f4')
            self._keyboard_vao = self.ctx.vertex_array(
                self._overlay_prog,
                [(self.ctx.buffer(verts.tobytes()), '2f 2f', 'in_position', 'in_uv')],
            )

    def _init_keyboard(self):
        """Initial keyboard build (called once when the user toggles it on)."""
        self._kb_show_shifted = False
        # Set size once: 80% of screen width with original aspect ratio
        if self.screen_width > 0 and self._keyboard_width == 1.6:
            self._keyboard_width = self.screen_width * 0.8
        self._sync_keyboard_size_from_width()
        self._build_keyboard_texture()

    def _sync_keyboard_size_from_width(self):
        """Keep keyboard physical size in sync with its texture aspect ratio."""
        self._keyboard_height = self._keyboard_width * _KB_TEX_H / float(_KB_TEX_W)
        self._kb_last_build_width = self._keyboard_width

    def _sync_keyboard_size_from_screen_width(self, previous_screen_width=None):
        """Scale the keyboard with the screen while preserving user size ratio."""
        if self.screen_width <= 0.0:
            return
        try:
            prev = float(previous_screen_width) if previous_screen_width is not None else 0.0
        except (TypeError, ValueError):
            prev = 0.0
        if prev > 0.0:
            ratio = float(self._keyboard_width) / prev
        else:
            ratio = 0.8
        ratio = max(0.2, min(1.2, ratio))
        self._keyboard_width = max(0.3, float(self.screen_width) * ratio)
        self._sync_keyboard_size_from_width()
        if self._keyboard_tex is not None:
            self._build_keyboard_texture()

    def _kb_world_mat(self):
        """Build the keyboard's world transform: rot_y(yaw) -rot_x(pitch) then translate.

        The keyboard's local frame has Z = surface normal. Negative pitch tilts the
        face up so a user looking down at it sees the face dead-on (friendly angle,
        like a desk keyboard).
        """
        cp = math.cos(self._keyboard_pitch); sp = math.sin(self._keyboard_pitch)
        cy = math.cos(self._keyboard_yaw);   sy = math.sin(self._keyboard_yaw)
        rot_y = np.array([[ cy, 0, sy, 0],
                        [  0, 1,  0, 0],
                        [-sy, 0, cy, 0],
                        [  0, 0,  0, 1]], dtype=np.float32)
        rot_x = np.array([[1,  0,   0, 0],
                        [0, cp, -sp, 0],
                        [0, sp,  cp, 0],
                        [0,  0,   0, 1]], dtype=np.float32)
        trans = np.eye(4, dtype=np.float32)
        # Translate to (pan_x, pan_y, -distance) -matches the world-anchor convention
        # used by the main screen.
        trans[0, 3] = self._keyboard_pan_x
        trans[1, 3] = self._keyboard_pan_y
        trans[2, 3] = -self._keyboard_distance
        return trans @ rot_y @ rot_x

    def _team_status_panel_metrics(self):
        """Return teammate status panel gap and height in screen-local meters."""
        if self.screen_height is None:
            fw, fh = self.frame_size
            if fh > fw:  # portrait: width becomes height
                sh = self.screen_width
            else:
                sh = self.screen_width * 9.0 / 16.0
        else:
            sh = self.screen_height
        return sh * 0.02, sh / 6.0

    def _anchor_keyboard_below_screen(self):
        """Snap the keyboard below the screen's bottom edge, facing the same direction.

        The keyboard sits below the FPS overlay panel so it doesn't overlap.
        """
        KB_GAP = 0.10
        if self.screen_height is None:
            fw, fh = self.frame_size
            if fh > fw:
                sh = self.screen_width
            else:
                sh = self.screen_width * 9.0 / 16.0
        else:
            sh = self.screen_height
        eff_height = self._keyboard_height
        self._keyboard_pan_x    = self.screen_pan_x
        self._keyboard_pan_y    = (self.screen_pan_y - sh / 2.0
                                - KB_GAP
                                - eff_height / 2.0)
        self._keyboard_distance = self.screen_distance - 0.001  # slightly closer than screen  # tiny Z offset to avoid depth fighting
        # Auto-orient keyboard toward the head (sphere center), same logic as screen.
        head = getattr(self, '_head_pos_w', None)
        if head is not None:
            hx, hy, hz = float(head[0]), float(head[1]), float(head[2])
            dx = self._keyboard_pan_x - hx
            dy = self._keyboard_pan_y - hy
            dz = -self._keyboard_distance - hz
            r = math.sqrt(dx*dx + dy*dy + dz*dz)
            if r > 1e-4:
                nx, ny, nz = dx / r, dy / r, dz / r
                base_yaw   = math.atan2(-nx, -nz)
                base_pitch = math.asin(max(-1.0, min(1.0, ny)))
                self._keyboard_yaw   = base_yaw   + self._kb_yaw_offset
                self._keyboard_pitch = base_pitch + self._kb_pitch_offset
            else:
                self._keyboard_yaw   = self.screen_yaw + self._kb_yaw_offset
                self._keyboard_pitch = self._kb_pitch_offset
        else:
            self._keyboard_yaw      = self.screen_yaw + self._kb_yaw_offset
            self._keyboard_pitch    = self._kb_pitch_offset

    def _render_keyboard(self, mgl_fbo, vp_mat):
        """Render the virtual keyboard quad and highlight hovered keys."""
        if self._keyboard_tex is None or self._keyboard_vao is None:
            return

        kw2 = self._keyboard_width  / 2.0
        kh2 = self._keyboard_height / 2.0
        kb_world = self._kb_world_mat()
        vp_kb = vp_mat @ kb_world   # shared for all key highlights

        # Keyboard quad: vertices are in [-1, +1] in X and Y, so scale by half-extents.
        scale_kb = np.array([[kw2, 0,   0, 0],
                            [0,   kh2, 0, 0],
                            [0,   0,   1, 0],
                            [0,   0,   0, 1]], dtype=np.float32)
        mvp = vp_kb @ scale_kb

        mgl_fbo.use()
        # depth_mask = False  # do not write semi-transparent pixels to depth
        self.ctx.depth_mask = False
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Keyboard border: solid quad slightly larger than the keyboard
        if self._kb_border_alpha > 0.0 and self._border_prog is not None:
            BORDER = 0.008
            bx = kw2 + BORDER
            by = kh2 + BORDER
            border_scale = np.array([[bx, 0, 0, 0],
                                    [0, by, 0, 0],
                                    [0, 0,  1, -0.001],
                                    [0, 0,  0, 1]], dtype=np.float32)
            border_mvp = vp_kb @ border_scale
            self._border_prog['u_mvp'].write(border_mvp.T.tobytes())
            self._border_prog['u_color'].value = (0.3, 0.7, 1.0, self._kb_border_alpha)
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        self._keyboard_tex.use(location=2)
        self._overlay_prog['u_mvp'].write(mvp.T.tobytes())
        self._keyboard_vao.render(moderngl.TRIANGLE_STRIP)

        def _hl_quad(rect_local, color):
            # rect_local is already in metres, expressed in the keyboard's local frame
            # (X right, Y up, Z = surface). Place a unit quad scaled to the key rect at
            # +1 mm in front of the surface to avoid z-fighting.
            x0, y0, x1, y1 = rect_local
            cx = (x0 + x1) / 2.0; cy_ = (y0 + y1) / 2.0
            hw = (x1 - x0) / 2.0; hh  = (y1 - y0) / 2.0
            hl_local = np.array([[hw, 0,  0, cx ],
                                [0,  hh, 0, cy_],
                                [0,  0,  1, 0.001],
                                [0,  0,  0, 1  ]], dtype=np.float32)
            hl_mvp = vp_kb @ hl_local
            self._border_prog['u_mvp'].write(hl_mvp.T.tobytes())
            self._border_prog['u_color'].value = color
            self._border_vao.render(moderngl.TRIANGLE_STRIP)

        # Highlight every key whose VK matches an armed modifier.
        # Locked modifiers get a brighter amber than one-shot to make state legible.
        VK_SHIFT = 0x10; VK_CAPS = 0x14; VK_CTRL = 0x11; VK_ALT = 0x12; VK_WIN = 0x5B
        oneshot_vks = set(); locked_vks = set()
        for name, vk in (('shift', VK_SHIFT), ('ctrl', VK_CTRL),
                        ('alt', VK_ALT), ('win', VK_WIN)):
            active, locked, _ = self._mod_state[name]
            if locked:   locked_vks.add(vk)
            elif active: oneshot_vks.add(vk)
        if self._caps_lock: locked_vks.add(VK_CAPS)
        for key in self._keyboard_keys:
            if key.vk in locked_vks:
                _hl_quad(key.rect_local, (1.0, 0.55, 0.05, 0.65))
            elif key.vk in oneshot_vks:
                _hl_quad(key.rect_local, (1.0, 0.7, 0.15, 0.45))

        # Cyan highlight on keys hovered by either laser (suppressed while gripping)
        if not (self._grip_l_now or self._grip_r_now):
            for hover_idx in set(x for x in [self._kb_hover_l, self._kb_hover_r] if x is not None):
                _hl_quad(self._keyboard_keys[hover_idx].rect_local, (0.2, 0.7, 1.0, 0.35))

        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True

        # Draw keyboard cursor circle at smoothed position (both hands)
        for _sk in ('_kb_smooth_l', '_kb_smooth_r'):
            _smooth_pos = getattr(self, _sk, None)
            if _smooth_pos is None:
                continue
            _cp = math.cos(self._keyboard_pitch); _sp = math.sin(self._keyboard_pitch)
            _cy = math.cos(self._keyboard_yaw);   _sy = math.sin(self._keyboard_yaw)
            _kb_r = np.array([_cy, 0.0, -_sy], dtype='f4')
            _kb_u = np.array([_sy * _sp, _cp, _cy * _sp], dtype='f4')
            _kb_nv = np.array([_sy * _cp, -_sp, _cy * _cp], dtype='f4')
            _kb_pos = np.array([self._keyboard_pan_x, self._keyboard_pan_y, -self._keyboard_distance], dtype='f4')
            _wp = _kb_pos + _kb_r * float(_smooth_pos[0]) + _kb_u * float(_smooth_pos[1])
            # Scale with distance like screen cursor, and hide outside keyboard bounds
            _kw2 = self._keyboard_width * 0.5
            _kh2 = self._keyboard_height * 0.5
            if abs(float(_smooth_pos[0])) <= _kw2 and abs(float(_smooth_pos[1])) <= _kh2:
                _dist_scale = float(np.clip(self._keyboard_distance / 2.0, 1.0, 3.0))
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.enable(moderngl.BLEND)
                for _r, _col in [(_dist_scale * 0.0096, (0.2, 0.6, 1.0, 0.75)), (_dist_scale * 0.0056, (1.0, 1.0, 1.0, 0.75))]:
                    _m = np.eye(4, dtype='f4')
                    _m[:3, 0] = _kb_r * _r
                    _m[:3, 1] = _kb_u * _r
                    _m[:3, 2] = _kb_nv
                    _m[:3, 3] = _wp
                    _mvp = vp_mat @ _m
                    self._border_prog['u_mvp'].write(_mvp.T.tobytes())
                    self._border_prog['u_color'].value = _col
                    self._circle_vao.render(moderngl.TRIANGLE_FAN)
                self.ctx.disable(moderngl.BLEND)
                self.ctx.enable(moderngl.DEPTH_TEST)