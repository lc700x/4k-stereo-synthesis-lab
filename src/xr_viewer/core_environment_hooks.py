# Desktop2Stereo OpenXR viewer: base environment/profile hooks for no-room mode.

import math

import numpy as np


class CoreEnvironmentHooksMixin:
    """Default no-room profile helpers and environment extension hooks."""

    def _profile_vec3(self, profile, keys, default):
        for key in keys:
            value = profile.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return [float(value[0]), float(value[1]), float(value[2])]
                except (TypeError, ValueError):
                    pass
        return list(default)

    def _profile_float(self, profile, keys, default):
        for key in keys:
            if key in profile:
                try:
                    return float(profile[key])
                except (TypeError, ValueError):
                    pass
        return float(default)

    def _profile_bool(self, profile, keys, default):
        for key in keys:
            if key in profile:
                value = profile[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in ('1', 'true', 'yes', 'on')
                return bool(value)
        return bool(default)

    def _profile_rotation_rad(self, profile, deg_keys, rad_keys, default):
        for key in deg_keys:
            value = profile.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return [math.radians(float(value[0])),
                            math.radians(float(value[1])),
                            math.radians(float(value[2]))]
                except (TypeError, ValueError):
                    pass
        return self._profile_vec3(profile, rad_keys, default)


    # Environment extension hooks.  The base viewer is intentionally environment-free;
    # xrviewer_env.OpenXRViewer overrides these methods with the room/profile/model code.
    def _discover_environment_models(self):
        return []

    def _reset_environment_profile_defaults(self):
        return None

    def _configure_environment_profile(self):
        self._environment_enabled = False
        self._environment_model = 'None'
        self._env_profile = {}
        self._env_model_path = None
        self._env_model_visible = False
        self._env_model_prims = []
        self._env_model_textures = []
        self._env_model_lights = []
        self._env_current_name = 'None'

    def _configure_profile_view_layout(self):
        # No room profile: keep the default screen placement configured in __init__.
        return None

    def _screen_profile_value(self, key, default=None):
        return default

    def _environment_screen_locked(self):
        return False

    def _head_model_mat4_from_views(self, views):
        return np.eye(4, dtype=np.float32)

    def _level_head_model_mat4(self, head_model):
        return head_model

    def _apply_profile_view_pose_to_xr_space(self, views=None):
        return None

    def _recenter_profile_view_pose(self, views=None):
        return None

    def _auto_view_position_from_screen(self):
        return None

    def _apply_profile_screen_layout(self):
        return None

    def _build_env_model_mat4(self):
        return np.eye(4, dtype=np.float32)

    def _transform_env_point(self, point):
        return np.asarray(point, dtype=np.float32)

    def _transform_env_direction(self, direction):
        d = np.asarray(direction, dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-8)

    def _env_light_range_scale(self):
        return 1.0

    def _load_env_model(self, path):
        return [], [], []

    def _release_env_model_resources(self):
        self._env_model_prims = []
        self._env_model_textures = []
        self._env_model_lights = []

    def _generate_default_room(self):
        return [], [], []

    def _init_env_model(self):
        self._env_model_visible = False
        self._env_model_prims = []
        self._env_model_textures = []
        self._env_model_lights = []
        self._env_available_models = []
        self._env_current_name = 'None'

    def _switch_environment_model(self, direction=1):
        return None

    def _render_env_model(self, mgl_fbo, vp_mat, view_mat):
        return None