# Desktop2Stereo OpenXR viewer: glow color smoothing helpers.


class CoreGlowMixin:
    """Shared glow and screen light color smoothing."""

    def _advance_glow_color(self, lerp=0.03):
        """Advance glow color toward the sampled frame average."""
        c = getattr(self, '_glow_color', (0.30, 0.55, 1.0))
        t = getattr(self, '_glow_target_color', c)
        self._glow_color = (
            float(c[0]) + lerp * (float(t[0]) - float(c[0])),
            float(c[1]) + lerp * (float(t[1]) - float(c[1])),
            float(c[2]) + lerp * (float(t[2]) - float(c[2])),
        )
        colors = getattr(self, '_screen_light_colors', tuple([self._glow_color] * 6))
        targets = getattr(self, '_screen_light_target_colors', colors)
        self._screen_light_colors = tuple(
            (
                float(c0[0]) + lerp * (float(t0[0]) - float(c0[0])),
                float(c0[1]) + lerp * (float(t0[1]) - float(c0[1])),
                float(c0[2]) + lerp * (float(t0[2]) - float(c0[2])),
            )
            for c0, t0 in zip(colors, targets)
        )