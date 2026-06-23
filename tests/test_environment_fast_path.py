from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _make_default_viewer(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.environment import OpenXRViewer

    class _DefaultViewer(OpenXRViewer):
        def __init__(self):
            pass

    return _DefaultViewer()


def _make_no_room_viewer(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.base import ScreenEffectsMixin

    class _NoRoomViewer(ScreenEffectsMixin):
        def __init__(self):
            pass

    return _NoRoomViewer()


def test_no_room_background_effects_skip_shadow_and_ground(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    viewer._screen_effects_enabled = True
    viewer.screen_height = 9.0
    viewer._render_glow_called = False
    viewer._render_shadow_called = False
    viewer._render_ground_light_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_shadow(*_args):
        viewer._render_shadow_called = True

    def _render_ground_light(*_args):
        viewer._render_ground_light_called = True

    viewer._render_glow = _render_glow
    viewer._render_shadow = _render_shadow
    viewer._render_ground_light = _render_ground_light
    viewer._render_screen_background_effects(None, None)

    assert viewer._render_glow_called
    assert not viewer._render_shadow_called
    assert not viewer._render_ground_light_called


def test_default_screen_state_persistence_is_disabled(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer.screen_width = 16.0
    viewer.screen_distance = 16.0
    viewer.screen_pan_x = 0.0
    viewer.screen_pan_y = 0.0
    viewer._preset_index = 5

    assert not viewer._restore_screen_state()
    viewer.screen_width = 2.4
    viewer.screen_distance = 2.0
    assert viewer._persist_screen_state() is None
    assert viewer.screen_width == 2.4
    assert viewer.screen_distance == 2.0


def test_default_glow_off_uses_blank_fast_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._glow_intensity_multiplier = 0.0
    viewer._render_glow_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    viewer._render_glow = _render_glow
    viewer._render_screen_background_effects(None, None)

    assert viewer._default_blank_fast_path()
    assert not viewer._render_glow_called


def test_default_glow_on_keeps_background_effect_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_intensity_multiplier = 1.0
    viewer._render_glow_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    viewer._render_glow = _render_glow
    viewer._render_screen_background_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert viewer._render_glow_called


def test_default_background_effects_skip_dark_room_board(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = [object()]
    viewer._dark_room_background = True
    viewer._bg_color_idx = 0
    viewer._glow_intensity_multiplier = 1.0
    viewer._current_view_mat = object()
    viewer._render_glow_called = False
    viewer._render_env_model_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_env_model(*_args):
        viewer._render_env_model_called = True

    viewer._render_glow = _render_glow
    viewer._render_env_model = _render_env_model
    viewer._render_screen_background_effects(None, None)

    assert viewer._render_glow_called
    assert not viewer._render_env_model_called


def test_default_glow_does_not_skip_curved_screen():
    env_text = (SRC / "xr_viewer" / "environment.py").read_text(encoding="utf-8")

    assert "if getattr(self, '_screen_curved', False):\n            return" not in env_text
