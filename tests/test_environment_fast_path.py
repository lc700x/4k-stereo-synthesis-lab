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


def test_no_room_screen_effects_wait_for_source_ready(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    viewer._screen_effects_enabled = True
    viewer.screen_height = 9.0
    viewer._runtime_direct_source = False
    viewer._should_show_source_border = lambda: False
    viewer._render_glow_called = False
    viewer._render_metallic_border_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_metallic_border(*_args):
        viewer._render_metallic_border_called = True

    viewer._render_glow = _render_glow
    viewer._render_metallic_border = _render_metallic_border
    viewer._render_screen_background_effects(None, None)
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._render_glow_called
    assert not viewer._render_metallic_border_called


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
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_glow_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    viewer._render_glow = _render_glow
    viewer._render_screen_background_effects(None, None)

    assert viewer._default_blank_fast_path()
    assert not viewer._render_glow_called


def test_default_profile_starts_with_surround_glow():
    import json

    profile_path = SRC / "xr_viewer" / "environments" / "Default" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["glow_mode"] == "surround"
    assert profile["controller_hdr_lighting"] is False
    assert profile["glow_intensity_multiplier"] == 0.0
    assert profile["glow_shell_intensity_multiplier"] == 1.85
    assert profile["frosted_glow_blend"] == 2.40
    assert profile["frosted_glow_thickness"] == 2.40
    assert profile["lighting_preset_index"] == 0
    assert profile["lighting_presets"][0]["glow_mode"] == "surround"


def test_panorama_profile_config_resolves_image(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    room = tmp_path / "PanoramaRoom"
    room.mkdir()
    image = room / "background.jpg"
    image.write_bytes(b"not-a-real-image")

    is_panorama, path, cfg = viewer._panorama_profile_config(
        {
            "environment_type": "panorama",
            "background": {"image": "background.jpg", "yaw_offset_deg": 45.0},
        },
        str(room),
    )

    assert is_panorama
    assert path == str(image)
    assert cfg["image"] == "background.jpg"
    assert cfg["yaw_offset_deg"] == 45.0


def test_environment_discovery_includes_panorama_image_folder(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    root = tmp_path / "environments"
    pano = root / "Pano"
    pano.mkdir(parents=True)
    (pano / "panorama.png").write_bytes(b"not-a-real-image")
    viewer._environment_root = str(root)
    viewer._environment_model = "Default"

    assert "Pano" in viewer._discover_environment_models()


def test_environment_discovery_includes_hdr_panorama_folder(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    root = tmp_path / "environments"
    pano = root / "HDR Pano"
    pano.mkdir(parents=True)
    (pano / "background.hdr").write_bytes(b"not-a-real-hdr")
    viewer._environment_root = str(root)
    viewer._environment_model = "Default"

    assert "HDR Pano" in viewer._discover_environment_models()


def test_radiance_hdr_loader_decodes_flat_rgbe(tmp_path):
    from xr_viewer.environment_renderer import _hdr_to_ldr_u8, _read_radiance_hdr

    hdr = tmp_path / "tiny.hdr"
    hdr.write_bytes(
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n"
        + bytes([128, 64, 32, 129, 0, 0, 0, 0])
    )

    arr, size = _read_radiance_hdr(str(hdr))

    assert size == (2, 1)
    assert arr.shape == (1, 2, 3)
    assert arr[0, 0, 0] > arr[0, 0, 1] > arr[0, 0, 2]
    assert arr[0, 1].sum() == 0.0
    ldr = _hdr_to_ldr_u8(arr)
    assert ldr.dtype.name == "uint8"
    assert ldr.shape == arr.shape


def test_official_webxr_hdr_environments_are_packaged():
    import pytest

    names = {
        "WebXR Autumn Forest": "autumn_forest_01_2k.hdr",
        "WebXR Cave Wall": "cave_wall_2k.hdr",
        "WebXR Fireplace": "fireplace_2k.hdr",
        "WebXR Georgentor": "georgentor_2k.hdr",
        "WebXR Snowy Park": "snowy_park_01_2k.hdr",
        "WebXR Studio": "studio_small_03_2k.hdr",
    }
    if not (SRC / "xr_viewer" / "environments" / "WebXR Autumn Forest" / "profile.json").is_file():
        pytest.skip("optional WebXR HDR panorama assets are not packaged")
    for env_name, hdr_name in names.items():
        env_dir = SRC / "xr_viewer" / "environments" / env_name
        profile = (env_dir / "profile.json").read_text(encoding="utf-8")
        assert (env_dir / hdr_name).is_file()
        assert '"environment_type": "panorama"' in profile
        assert f'"image": "{hdr_name}"' in profile
        assert '"controller_hdr_lighting": true' in profile


def test_panorama_profile_survives_viewer_initialization(monkeypatch):
    import pytest

    if not (SRC / "xr_viewer" / "environments" / "WebXR Autumn Forest" / "profile.json").is_file():
        pytest.skip("optional WebXR HDR panorama assets are not packaged")
    monkeypatch.chdir(SRC)
    from xr_viewer.environment import OpenXRViewer

    viewer = OpenXRViewer(environment_model="WebXR Autumn Forest", show_preview_window=False)

    assert viewer._env_model_path is None
    assert viewer._panorama_background_path.endswith("autumn_forest_01_2k.hdr")
    assert viewer._panorama_background_settings["image"] == "autumn_forest_01_2k.hdr"
    assert viewer._controller_hdr_lighting is True


def test_async_panorama_background_disables_default_glb_mesh(monkeypatch, tmp_path):
    import json

    root = tmp_path / "environments"
    default = root / "Default"
    default.mkdir(parents=True)
    (default / "profile.json").write_text(
        json.dumps({"xr_quad_layer_enabled": False}),
        encoding="utf-8",
    )
    (default / "environment.glb").write_bytes(b"glb")
    monkeypatch.chdir(SRC)
    from xr_viewer.environment import OpenXRViewer

    viewer = OpenXRViewer(environment_model="None", show_preview_window=False)
    viewer._environment_root = str(root)
    viewer._environment_model = "Default"
    viewer._openxr_panorama_background_enabled = True
    viewer._xr_quad_layer_enabled = True

    viewer._configure_environment_profile()

    assert viewer._env_model_path is None
    assert viewer._panorama_background_path is None
    assert viewer._xr_quad_layer_enabled is True


def test_panorama_environment_skips_glb_initialization(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_enabled = True
    viewer._panorama_background_path = "background.jpg"
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    viewer._active_environment = "Pano"

    viewer._init_env_model()

    assert not viewer._env_model_visible
    assert viewer._env_model_prims == []
    assert viewer._active_environment == "Pano"


def test_default_glow_on_keeps_background_effect_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_glow_called = False
    viewer._render_glow_shell_call = None

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_glow_shell(*args, **kwargs):
        viewer._render_glow_shell_call = (args, kwargs)

    viewer._render_glow = _render_glow
    viewer._render_glow_shell = _render_glow_shell
    viewer._render_screen_background_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert not viewer._render_glow_called
    assert viewer._render_glow_shell_call is not None
    assert viewer._render_glow_shell_call[1]["intensity_multiplier"] == 0.72


def test_default_surround_glow_uses_shell_render_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "surround"
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 1.0
    viewer._render_glow_called = False
    viewer._render_glow_shell_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_glow_shell(*_args):
        viewer._render_glow_shell_called = True

    viewer._render_glow = _render_glow
    viewer._render_glow_shell = _render_glow_shell
    viewer._render_screen_background_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert not viewer._render_glow_called
    assert viewer._render_glow_shell_called


def test_environment_screen_effects_wait_for_source_ready(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "surround"
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 1.0
    viewer._runtime_direct_source = False
    viewer._should_show_source_border = lambda: False
    viewer._render_glow_shell_called = False
    viewer._render_frosted_glow_called = False

    def _render_glow_shell(*_args):
        viewer._render_glow_shell_called = True

    def _render_frosted_glow(*_args):
        viewer._render_frosted_glow_called = True

    viewer._render_glow_shell = _render_glow_shell
    viewer._render_frosted_glow = _render_frosted_glow
    viewer._render_screen_background_effects(None, None)
    viewer._glow_mode = "frosted"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._render_glow_shell_called
    assert not viewer._render_frosted_glow_called


def test_default_frosted_glow_uses_frosted_render_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "frosted"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_frosted_glow_called = False
    viewer._render_glow_called = False

    def _render_frosted_glow(*_args):
        viewer._render_frosted_glow_called = True

    def _render_glow(*_args):
        viewer._render_glow_called = True

    viewer._render_frosted_glow = _render_frosted_glow
    viewer._render_glow = _render_glow
    viewer._render_screen_background_effects(None, None)
    assert not viewer._render_frosted_glow_called
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert viewer._render_frosted_glow_called
    assert not viewer._render_glow_called


def test_default_veil_glow_uses_veil_render_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "veil"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_frosted_veil_called = False
    viewer._render_frosted_glow_called = False

    def _render_frosted_veil(*_args):
        viewer._render_frosted_veil_called = True

    def _render_frosted_glow(*_args):
        viewer._render_frosted_glow_called = True

    viewer._render_frosted_veil = _render_frosted_veil
    viewer._render_frosted_glow = _render_frosted_glow
    viewer._render_screen_background_effects(None, None)
    assert not viewer._render_frosted_veil_called
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert viewer._render_frosted_veil_called
    assert not viewer._render_frosted_glow_called


def test_default_background_effects_skip_dark_room_board(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = [object()]
    viewer._dark_room_background = True
    viewer._bg_color_idx = 0
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._current_view_mat = object()
    viewer._render_glow_called = False
    viewer._render_glow_shell_called = False
    viewer._render_env_model_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_glow_shell(*_args, **_kwargs):
        viewer._render_glow_shell_called = True

    def _render_env_model(*_args):
        viewer._render_env_model_called = True

    viewer._render_glow = _render_glow
    viewer._render_glow_shell = _render_glow_shell
    viewer._render_env_model = _render_env_model
    viewer._render_screen_background_effects(None, None)

    assert not viewer._render_glow_called
    assert viewer._render_glow_shell_called
    assert not viewer._render_env_model_called


def test_default_glow_does_not_skip_curved_screen():
    env_text = (SRC / "xr_viewer" / "environment.py").read_text(encoding="utf-8")

    assert "if getattr(self, '_screen_curved', False):\n            return" not in env_text


def test_default_glow_mode_cycle_from_y(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._glow_mode = "surround"
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 1.85
    viewer._env_profile = {
        "glow_intensity_multiplier": 0.0,
        "glow_shell_intensity_multiplier": 0.0,
        "lighting_presets": [
            {
                "name": "Surround Glow",
                "glow_mode": "surround",
                "glow_intensity_multiplier": 0.0,
                "glow_shell_intensity_multiplier": 1.85,
            },
            {
                "name": "Screen Glow",
                "glow_mode": "screen",
                "glow_intensity_multiplier": 1.85,
                "glow_shell_intensity_multiplier": 0.0,
            },
            {
                "name": "Glow Off",
                "glow_mode": "off",
                "glow_intensity_multiplier": 0.0,
                "glow_shell_intensity_multiplier": 0.0,
            },
            {
                "name": "Frosted Veil",
                "glow_mode": "veil",
                "glow_intensity_multiplier": 1.85,
                "glow_shell_intensity_multiplier": 0.0,
                "frosted_veil_intensity": 1.35,
            },
            {
                "name": "Frosted Glow",
                "glow_mode": "frosted",
                "glow_intensity_multiplier": 1.85,
                "glow_shell_intensity_multiplier": 0.0,
                "frosted_glow_intensity": 3.0,
            },
        ],
    }
    viewer._save_glow_to_builtin_profile = lambda: None

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "screen"
    assert viewer._glow_intensity_multiplier == 1.85
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "off"
    assert viewer._glow_intensity_multiplier == 0.0
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "veil"
    assert viewer._glow_intensity_multiplier == 1.85
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "frosted"
    assert viewer._glow_intensity_multiplier == 1.85
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "surround"
    assert viewer._glow_intensity_multiplier == 0.0
    assert viewer._glow_shell_intensity_multiplier == 1.85


def test_frosted_glow_shader_uses_flat_grid_source_crop():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    assert "_FROSTED_VEIL_VERT" in glsl_text
    assert "_FROSTED_GLOW_FRAG" in glsl_text
    assert "in vec2 v_uv" in glsl_text
    assert "in vec3 v_local" in glsl_text
    assert "uniform mat4 u_model" in glsl_text
    assert "uniform sampler2D u_screen_tex" in glsl_text
    assert "uniform vec4 u_source_crop" in glsl_text
    assert "u_source_crop.xy + v_uv * u_source_crop.zw" in glsl_text
    assert "textureLod(u_screen_tex" in glsl_text
    assert "smoothstep(u_threshold, 1.0, luma)" in glsl_text
    assert "u_beam_softness" in glsl_text
    assert "u_frost_blend" in glsl_text
    assert "u_beam_thickness" in glsl_text
    assert "u_edge_inset" in glsl_text
    assert "u_diffuse_scatter" in glsl_text
    assert "float depth = clamp(v_local.z, 0.0, 1.0)" in glsl_text
    assert "wall_endpoint_fade(v_uv, v_local)" in glsl_text
    assert "u_debug_wall" not in glsl_text
    assert "hash12" in glsl_text


def test_screen_effects_do_not_sample_runtime_eye_texture():
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    source_func = effects_text.split("def _screen_effect_source_texture", 1)[1].split("def _render_glow", 1)[0]

    assert "def _screen_effect_source_texture(self, *, allow_runtime_eye=True):" in effects_text
    assert effects_text.count("_screen_effect_source_texture(allow_runtime_eye=False)") == 4
    assert "_runtime_effect_safe_source_tex" in source_func
    assert "_runtime_eye_textures" not in source_func
    assert "_current_eye_index" not in source_func


def test_screen_light_uses_effect_source_texture_not_runtime_eye_texture():
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    source_func = render_text.split("def _screen_light_source_texture", 1)[1].split("def _apply_cinema_light_uniforms", 1)[0]

    assert "_runtime_effect_safe_source_tex" in source_func
    assert "_runtime_effect_safe_source_size" in source_func
    assert "_runtime_eye_textures" not in source_func
    assert "_current_eye_index" not in source_func


def test_runtime_effect_source_texture_is_prepared_for_all_glow_modes(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    cases = [
        ("screen", 1.0, 0.0),
        ("surround", 0.0, 1.0),
        ("veil", 1.0, 0.0),
        ("frosted", 1.0, 0.0),
    ]

    for mode, glow_mult, shell_mult in cases:
        viewer._glow_mode = mode
        viewer._glow_intensity_multiplier = glow_mult
        viewer._glow_shell_intensity_multiplier = shell_mult
        assert viewer._runtime_effects_need_source_texture()

    viewer._glow_mode = "off"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 1.0

    assert not viewer._runtime_effects_need_source_texture()


def test_openxr_full_synthesis_preserves_effect_source_before_eye_sampling():
    runtime_text = (SRC / "stereo_runtime" / "runtime.py").read_text(encoding="utf-8")
    pipeline_text = (SRC / "stereo_runtime" / "pipeline.py").read_text(encoding="utf-8")
    core_text = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    uploader_text = (SRC / "viewer" / "gl_texture_uploader.py").read_text(encoding="utf-8")

    assert "source_rgb: torch.Tensor | None = None" in runtime_text
    assert "source_rgb=source_rgb" in runtime_text
    assert "openxr_result_from_stereo_result(runtime_result, source_rgb=runtime_rgb)" in pipeline_text
    assert "def _try_update_runtime_effect_source_texture_gpu" in core_text
    assert "CudaGlTextureUploader" in core_text
    assert "self.ctx.texture((w, h), 4, dtype='f1')" in core_text
    gpu_upload_func = core_text.split("def _try_update_runtime_effect_source_texture_gpu", 1)[1].split("def _update_runtime_effect_source_texture", 1)[0]
    assert "torch.cuda.current_stream(device_index).synchronize()" not in gpu_upload_func
    assert "ready_event.synchronize()" not in core_text
    assert gpu_upload_func.index("source_rgba =") < gpu_upload_func.index("upload_path = uploader.upload_rgba")
    update_func = core_text.split("def _update_runtime_effect_source_texture", 1)[1].split("def _release_runtime_eye_texture_resources", 1)[0]
    assert "_try_update_runtime_effect_source_texture_gpu" in update_func
    assert "_runtime_eye_to_numpy" not in update_func
    assert ".write(rgba.tobytes())" not in update_func
    assert "self._release_runtime_effect_source_texture()" in update_func
    image_upload = uploader_text.split("def _upload_image", 1)[1].split("def _ensure_pbos", 1)[0]
    assert "glFlush()" not in image_upload
    assert "_screen_effect_source_texture(allow_runtime_eye=False)" in effects_text


def test_screen_glow_shader_uses_region_color_grid():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")

    assert "_GLOW_DOWNSAMPLE_FRAG" in glsl_text
    assert "uniform sampler2D u_glow_tex" in glsl_text
    assert "uniform sampler2D u_screen_light_tex" in glsl_text
    assert "textureLod(u_glow_tex" in glsl_text
    assert "textureLod(u_screen_light_tex" in glsl_text
    assert "glow_grid_color" in glsl_text
    assert "def _screen_effect_source_texture" in effects_text
    assert "_runtime_effect_safe_source_tex" in effects_text


def test_surround_glow_shell_uses_screen_border_color():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")

    assert "_GLOW_SHELL_FRAG" in glsl_text
    assert "uniform sampler2D u_glow_tex" in glsl_text
    assert "uniform int u_glow_use_tex" in glsl_text
    assert "sample_border_color" in glsl_text
    assert "sample_region_reflection" in glsl_text
    assert "vec2 grid = vec2(4.0, 3.0)" in glsl_text
    shell_frag = glsl_text.split("_GLOW_SHELL_FRAG", 1)[1]
    assert "top_col" in shell_frag
    assert "bottom_col" in shell_frag
    assert "left_col" in shell_frag
    assert "right_col" in shell_frag
    assert "vertical_edges" in shell_frag
    assert "textureLod(u_glow_tex, q, 0.0)" in shell_frag
    assert "region_mix" in shell_frag
    assert "edge_band_depth" not in shell_frag
    assert "for (int" not in shell_frag


def test_realtime_screen_glow_cpu_sampler_is_removed():
    upload_text = (SRC / "xr_viewer" / "core_frame_upload.py").read_text(encoding="utf-8")
    maybe_sample = upload_text.split("def _maybe_sample_glow_target_color", 1)[1]

    assert "def _sample_glow_target_color" not in upload_text
    assert "np.asarray(rgb" not in maybe_sample
    assert "self._glow_color_counter = 0" in maybe_sample
    assert "return" in maybe_sample


def test_screen_glow_does_not_trigger_cpu_color_sampling(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._glow_intensity = 1.0
    viewer._glow_intensity_multiplier = 1.0
    viewer._screen_light_dynamic = False
    viewer._bg_color_idx = 0
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._glow_color_counter = 0
    viewer._sampled = False

    viewer._maybe_sample_glow_target_color(None, is_tensor=False)

    assert not viewer._sampled
    assert viewer._glow_color_counter == 0


def test_environment_dynamic_light_does_not_trigger_cpu_color_sampling(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._glow_intensity = 1.0
    viewer._glow_intensity_multiplier = 0.0
    viewer._screen_light_dynamic = True
    viewer._screen_light_intensity = 2.0
    viewer._bg_color_idx = 0
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    viewer._dark_room_prims = []
    viewer._glow_color_counter = 0
    viewer._sampled = False

    viewer._maybe_sample_glow_target_color(None, is_tensor=False)

    assert not viewer._sampled
    assert viewer._glow_color_counter == 0


def test_default_dark_room_does_not_trigger_cpu_color_sampling(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._glow_intensity = 1.0
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._screen_light_dynamic = True
    viewer._screen_light_intensity = 3.5
    viewer._bg_color_idx = 0
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = [object()]
    viewer._glow_color_counter = 0
    viewer._sampled = False

    viewer._maybe_sample_glow_target_color(None, is_tensor=False)

    assert not viewer._sampled
    assert viewer._glow_color_counter == 0


def test_frosted_glow_keyboard_adjustment_is_disabled(monkeypatch):
    import glfw

    viewer = _make_default_viewer(monkeypatch)
    viewer._frosted_glow_blend = 2.40
    viewer._frosted_glow_thickness = 2.40
    viewer._preset_name_overlay = ""
    viewer._preset_osd_show_t = 0.0

    assert not viewer._adjust_frosted_glow_keyboard(glfw.KEY_RIGHT)
    assert not viewer._adjust_frosted_glow_keyboard(glfw.KEY_UP, glfw.MOD_SHIFT)
    assert viewer._frosted_glow_blend == 2.40
    assert viewer._frosted_glow_thickness == 2.40
    assert viewer._preset_name_overlay == ""


def test_frosted_glow_virtual_keyboard_adjustment_is_disabled(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._frosted_glow_blend = 2.40
    viewer._frosted_glow_thickness = 2.40
    viewer._preset_name_overlay = ""
    viewer._preset_osd_show_t = 0.0

    assert not viewer._adjust_frosted_glow_vk(0x27)
    assert not viewer._adjust_frosted_glow_vk(0x28)
    assert viewer._frosted_glow_blend == 2.40
    assert viewer._frosted_glow_thickness == 2.40
    assert viewer._preset_name_overlay == ""


def test_controller_shader_uses_panorama_ibl_reflection():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    render_text = (SRC / "xr_viewer" / "core_laser_render.py").read_text(encoding="utf-8")
    profile_text = (SRC / "xr_viewer" / "environment_profiles.py").read_text(encoding="utf-8")
    ctrl_frag = glsl_text.split("_CTRL_FRAG", 1)[1].split("_ENV_VERT", 1)[0]

    assert "uniform sampler2D u_env_tex" in ctrl_frag
    assert "uniform sampler2D u_screen_light_tex" in ctrl_frag
    assert "textureLod(u_env_tex, env_uv(R), 3.0)" in ctrl_frag
    assert "textureLod(u_screen_light_tex" in ctrl_frag
    assert "u_light_color" not in ctrl_frag
    assert "u_ambient_color" not in ctrl_frag
    assert "_get_panorama_texture" in render_text
    assert "_controller_hdr_lighting" in render_text
    assert "if getattr(self, '_controller_hdr_lighting', True):" in render_text
    assert "u_use_env_tex" in render_text
    assert "u_screen_light_enabled" in render_text
    assert "controller_hdr_lighting" in profile_text
    assert "controller_hdr_reflection" in profile_text
    assert "== '.hdr'" in profile_text


def test_controller_touch_bindings_share_profile_suggestion_call():
    source = (SRC / "xr_viewer" / "core_controller_actions.py").read_text(encoding="utf-8")

    assert source.count("xr.suggest_interaction_profile_bindings") == 1
    assert '"/user/hand/left/input/thumbstick/touch"' in source
    assert '"/user/hand/left/input/grip/pose"' in source
