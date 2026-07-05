# Desktop2Stereo OpenXR viewer: GLFW setup and debug keyboard controls.

import time

import glfw


class CoreWindowInputMixin:
    """Hidden GLFW context setup plus desktop debug key handling."""

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("[OpenXRViewer] GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # hidden -GL context only
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(1, 1, "D2S-XR", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("[OpenXRViewer] GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        # Keyboard controls -keep a reference so it isn't GC'd
        self._key_callback_ref = self._make_key_callback()
        glfw.set_key_callback(self.window, self._key_callback_ref)
        self._frosted_hotkey_prev = {}

    def _toggle_team_status_panel(self):
        self._team_status_visible = not self._team_status_visible
        if self._team_status_visible:
            self._team_fps_visible = self._team_status_visible

    def _cycle_a_panel(self):
        """Cycle A long-press: hidden ->screen FPS ->screen help ->hidden."""
        self._a_cycle_state = (self._a_cycle_state + 1) % 3
        if self._a_cycle_state == 0:
            self._team_fps_visible = False
            self._team_status_visible = False
            self._team_help_visible = False
        elif self._a_cycle_state == 1:
            self._team_fps_visible = True
            self._team_status_visible = True
            self._team_help_visible = False
        elif self._a_cycle_state == 2:
            self._team_fps_visible = True
            self._team_status_visible = True
            self._team_help_visible = True

    def _cycle_b_panel(self):
        """Cycle B long-press: hidden ->hand FPS ->hand help ->hidden."""
        self._b_cycle_state = (self._b_cycle_state + 1) % 3
        if self._b_cycle_state == 0:
            self._hand_fps_visible = False
            self._fps_overlay_visible = False
        elif self._b_cycle_state == 1:
            self._hand_fps_visible = True
            self._fps_overlay_visible = False
        elif self._b_cycle_state == 2:
            self._hand_fps_visible = True
            self._fps_overlay_visible = True

    def _adjust_frosted_glow_keyboard(self, key, mods=0):
        return False

    def _adjust_frosted_glow_vk(self, vk):
        return False

    def _poll_frosted_glow_hotkeys(self):
        return

    def _make_key_callback(self):
        viewer = self
        def _cb(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            d = 0.1; s = 0.15; p = 0.1; r = 0.05
            screen_locked = viewer._environment_screen_locked()
            if key == glfw.KEY_F:
                viewer._toggle_team_status_panel()
            elif key == glfw.KEY_Z:
                viewer.depth_strength = max(0.0, viewer.depth_strength - 0.01)
            elif key == glfw.KEY_C:
                viewer.depth_strength = min(0.5, viewer.depth_strength + 0.01)
            elif key == glfw.KEY_X:
                viewer.depth_strength = 0.0   # flat mode -no parallax distortion
            elif key == glfw.KEY_V:
                viewer._toggle_quad_layer_compare()
            elif key == glfw.KEY_R:
                viewer._reset_screen_to_default(show_border=True)
            elif screen_locked:
                return
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                viewer._screen_ref_size += s; viewer.screen_height = None
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                viewer._screen_ref_size = max(0.8, viewer._screen_ref_size - s)
                viewer.screen_height = None
            elif key == glfw.KEY_Q: viewer.screen_yaw += r
            elif key == glfw.KEY_E: viewer.screen_yaw -= r
            elif key == glfw.KEY_T: viewer.screen_pitch += r
            elif key == glfw.KEY_G: viewer.screen_pitch -= r
        return _cb

    def _toggle_quad_layer_compare(self):
        if self._xr_quad_layer_failed or not self._quad_swapchains:
            self._xr_quad_layer_active = False
            self._preset_name_overlay = 'Quad Layer unavailable'
        else:
            self._xr_quad_layer_active = True
            self._preset_name_overlay = 'Quad Layer Screen'
        self._preset_osd_show_t = time.perf_counter()
        self._publish_runtime_config()
        print(
            "[OpenXRViewer] Screen layer status: "
            f"{self._preset_name_overlay} "
            f"active={self._xr_quad_layer_active} "
            f"swapchains={len(self._quad_swapchains)} "
            f"array_size={int(self._quad_swapchain_array_size.get(0, 0) or 0)} "
            f"per_eye_layers=True "
            f"stereo_boost={float(getattr(self, '_xr_quad_layer_stereo_boost', 1.0)):.2f} "
            f"failed={self._xr_quad_layer_failed}"
        )
