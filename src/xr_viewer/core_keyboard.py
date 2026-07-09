# Desktop2Stereo OpenXR viewer: virtual keyboard texture, placement, and rendering.

import math

import numpy as np

from .keyboard_layout import _KB_TEX_H, _KB_TEX_W
from .overlay_textures import build_keyboard_rgba


class CoreKeyboardMixin:
    """Virtual keyboard layout texture, sizing, world transform, and rendering."""

    def _refresh_keyboard_content(self, hover_indices=(), held_indices=()):
        """Refresh shared keyboard RGBA/key hit rects without assuming a renderer."""
        hover_indices = tuple(i for i in hover_indices if i is not None)
        held_indices = tuple(i for i in held_indices if i is not None)
        self._keyboard_rgba, self._keyboard_keys = build_keyboard_rgba(
            bool(self._kb_show_shifted),
            self._keyboard_width,
            self._keyboard_height,
            getattr(self, "font_type", None),
            hover_indices=hover_indices,
            held_indices=held_indices,
        )
        self._keyboard_content_key = (
            "keyboard",
            bool(self._kb_show_shifted),
            round(float(self._keyboard_width), 4),
            round(float(self._keyboard_height), 4),
            hover_indices,
            held_indices,
        )
        return self._keyboard_rgba

    def _init_keyboard(self):
        """Initial keyboard build for Quad overlay rendering."""
        self._kb_show_shifted = False
        # Set size once: 80% of screen width with original aspect ratio
        if self.screen_width > 0 and self._keyboard_width == 1.6:
            self._keyboard_width = self.screen_width * 0.8
        self._sync_keyboard_size_from_width()
        self._refresh_keyboard_content()

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
        self._refresh_keyboard_content()

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

        The keyboard's top edge stays 15% of the screen height below the screen.
        """
        if self.screen_height is None:
            fw, fh = self.frame_size
            if fh > fw:
                sh = self.screen_width
            else:
                sh = self.screen_width * 9.0 / 16.0
        else:
            sh = self.screen_height
        kb_gap = sh * 0.15
        eff_height = self._keyboard_height
        self._keyboard_pan_x    = self.screen_pan_x
        self._keyboard_pan_y    = (self.screen_pan_y - sh / 2.0
                                - kb_gap
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
