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

    assert "xr_quad_layer_enabled" not in profile
    assert profile["glow_mode"] == "surround"
    assert profile["controller_hdr_lighting"] is False
    assert profile["glow_intensity_multiplier"] == 0.0
    assert profile["glow_shell_intensity_multiplier"] == 1.85
    assert profile["frosted_glow_blend"] == 2.40
    assert profile["frosted_glow_thickness"] == 2.40
    assert profile["lighting_preset_index"] == 0
    assert profile["lighting_presets"][0]["glow_mode"] == "surround"


def test_builtin_environment_profiles_do_not_disable_quad_layer():
    import json

    for name in ("Default", "Cinema"):
        profile_path = SRC / "xr_viewer" / "environments" / name / "profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        assert "xr_quad_layer_enabled" not in profile


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

    is_panorama, path, cfg = viewer._panorama_profile_config(
        {
            "environment_type": "panorama",
            "panorama": {
                "image": "background.jpg",
                "stereo_layout": "sbs",
                "wall_light_mask": "mask.png",
                "screen_light_layout": {"uv": [0.4, 0.6], "radius": [0.2, 0.1]},
            },
        },
        str(room),
    )

    assert is_panorama
    assert path == str(image)
    assert cfg["stereo_layout"] == "sbs"
    assert cfg["wall_light_mask"] == "mask.png"
    assert cfg["screen_light_layout"]["uv"] == [0.4, 0.6]


def test_panorama_profile_auto_wall_mask_bakes_cached_png(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    room = tmp_path / "PanoramaRoom"
    room.mkdir()
    image = room / "background.jpg"
    image.write_bytes(b"not-a-real-image")

    is_panorama, path, cfg = viewer._panorama_profile_config(
        {
            "environment_type": "panorama",
            "panorama": {
                "image": "background.jpg",
                "wall_light_mask": "auto",
                "wall_light_mask_resolution": [64, 32],
                "screen_light_layout": {"uv": [0.25, 0.5], "radius": [0.1, 0.2]},
            },
        },
        str(room),
    )

    mask_path = room / cfg["wall_light_mask"]
    assert is_panorama
    assert path == str(image)
    assert mask_path.is_file()
    assert mask_path.parent.name == ".d2s_bake"
    assert cfg["wall_light_mask"].endswith(".png")
    from PIL import Image
    assert Image.open(mask_path).size == (64, 32)


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


def test_panorama_hdr_texture_uses_float_upload(monkeypatch, tmp_path):
    from xr_viewer import environment_renderer

    viewer = _make_default_viewer(monkeypatch)
    hdr = tmp_path / "tiny.hdr"
    hdr.write_bytes(
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n"
        + bytes([128, 64, 32, 129, 0, 0, 0, 0])
    )

    class _Texture:
        filter = None
        repeat_x = False
        repeat_y = False

        def build_mipmaps(self):
            pass

    calls = []

    class _Ctx:
        info = {"GL_MAX_TEXTURE_SIZE": 8192}

        def texture(self, size, components, data, **kwargs):
            calls.append((size, components, len(data), kwargs.get("dtype")))
            return _Texture()

    def _fail_ldr(_arr):
        raise AssertionError("HDR panorama should not be tone-mapped before float upload")

    viewer.ctx = _Ctx()
    viewer._panorama_background_path = str(hdr)
    viewer._panorama_tex = None
    viewer._panorama_tex_path = None
    monkeypatch.setattr(environment_renderer, "_hdr_to_ldr_u8", _fail_ldr)

    assert viewer._get_panorama_texture() is not None
    assert calls == [((2, 1), 3, 2 * 1 * 3 * 2, "f2")]


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


def test_openxr_loop_uses_fast_env_model_initializer():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    run_body = impl_text.split("def run(self, first_rgb=None", 1)[1].split("    # Cleanup", 1)[0]

    assert "def _ensure_env_model_initialized" in impl_text
    assert "_ensure_env_model_initialized(\"Preview-only\")" in run_body
    assert "_ensure_env_model_initialized(\"Lazy\")" in run_body
    assert "_init_env_model()" not in run_body


def test_env_model_initializer_skips_panorama_background(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._env_model_init_done = False
    viewer._environment_enabled = True
    viewer._panorama_background_path = "background.jpg"
    viewer._env_model_path = "environment.glb"
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    viewer._called = False
    viewer._panorama_called = False

    def _init_env_model():
        viewer._called = True

    def _get_panorama_texture():
        viewer._panorama_called = True
        return object()

    viewer._init_env_model = _init_env_model
    viewer._get_panorama_texture = _get_panorama_texture
    viewer._ensure_env_model_initialized("Test")

    assert viewer._env_model_init_done
    assert not viewer._called
    assert viewer._panorama_called
    assert not viewer._env_model_visible
    assert viewer._env_model_prims == []


def test_panorama_background_is_preloaded_outside_render_path():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    model_text = (SRC / "xr_viewer" / "environment_model.py").read_text(encoding="utf-8")
    init_func = impl_text.split("def _ensure_env_model_initialized", 1)[1].split("    # Main blocking loop", 1)[0]
    settings_func = render_text.split("def _panorama_render_settings", 1)[1].split("def _render_panorama_background", 1)[0]
    render_func = render_text.split("def _render_panorama_background", 1)[1].split("def _render_env_model", 1)[0]
    switch_func = model_text.split("def _switch_environment_model", 1)[1]
    pano_frag = glsl_text.split("_PANORAMA_FRAG", 1)[1].split("_GLOW_DOWNSAMPLE_FRAG", 1)[0]

    assert "get_panorama_texture()" in init_func
    assert "_panorama_texture_ready()" in render_func
    assert "_panorama_light_mask_texture_ready()" in render_func
    assert "_get_panorama_texture()" not in render_func
    assert "_get_panorama_light_mask_texture()" not in render_func
    assert "_panorama_light_mask_path_from_settings()" not in render_func
    assert "_ensure_env_model_initialized(\"Switch\")" in switch_func
    assert "_init_env_model()" not in switch_func
    controller_render = (SRC / "xr_viewer" / "core_laser_render.py").read_text(encoding="utf-8").split(
        "def _render_controllers", 1
    )[1]
    assert "_panorama_texture_ready()" in controller_render
    assert "_get_panorama_texture()" not in controller_render
    assert "np.linalg.inv(view_mat)" not in controller_render
    assert "uniform sampler2D u_screen_light_tex" in pano_frag
    assert "uniform sampler2D u_wall_light_mask_tex" in pano_frag
    assert "vec3 screen_light_probe_color()" in pano_frag
    assert "textureLod(u_screen_light_tex" in pano_frag
    assert "textureLod(u_wall_light_mask_tex" in pano_frag
    assert "return color * (1.0 / 9.0)" in pano_frag
    assert "u_screen_light_uv" in pano_frag
    assert "u_screen_light_radius" in pano_frag
    assert "uniform int u_stereo_layout" in pano_frag
    assert "uniform int u_eye_index" in pano_frag
    assert "sample_uv.x = pano_uv.x * 0.5 + (u_eye_index == 1 ? 0.5 : 0.0)" in pano_frag
    assert "screen_light_layout" in settings_func
    assert "stereo_layout_raw" in settings_func
    assert "'sbs'" in settings_func
    assert "u_stereo_layout" in render_func
    assert "u_eye_index" in render_func
    assert "light_layout.get('uv'" in settings_func
    assert "light_layout.get('radius'" in settings_func
    assert "return False" in render_func
    assert "return True" in render_func
    assert "except Exception as exc:" in render_func
    assert "openxr_background_panorama_failed" in render_func
    assert "finally:" in render_func
    render_finally = render_func.split("finally:", 1)[1]
    assert "self.ctx.depth_mask = previous_depth_mask" in render_finally
    assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally
    assert "tex.use(location=8)" in render_func
    assert "_bind_screen_light_source_texture(location=10)" in render_func
    assert "mask_tex.use(location=11)" in render_func
    assert "_view_mat_inv(view_rot)" in render_func
    assert "np.linalg.inv(view_rot)" not in render_func
    assert "_panorama_render_settings()" in render_func
    assert "screen_light_layout" not in render_func
    assert "_panorama_render_settings_key" in settings_func
    assert "_panorama_light_mask_path_from_settings" in render_text
    assert "_panorama_light_mask_path_key" in render_text
    eye_background = impl_text.split("background_start = time.perf_counter()", 1)[1].split(
        "self._breakdown_add_time('openxr_background'", 1
    )[0]
    assert "if self._render_panorama_background(mgl_fbo, view_mat, proj_mat):" in eye_background
    assert "self._breakdown_inc('openxr_background_panorama')" in eye_background
    assert "if eye_index == 0:" in eye_background
    assert eye_background.count("if eye_index == 0:") == 3


def test_quad_screen_path_skips_glb_environment_mesh_hot_path():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_eye = impl_text.split("def _render_eye", 1)[1].split("# 3. Keyboard", 1)[0]
    env_block = render_eye.split("# -3. Environment model", 1)[1].split("if not background_rendered:", 1)[0]

    assert "quad_unavailable_reason = self._quad_layer_unavailable_reason()" in render_eye
    assert "draw_projection_screen = quad_unavailable_reason is not None" in render_eye
    assert "if draw_projection_screen and self._env_model_visible and self._env_model_prims:" in env_block
    assert "self._render_env_model(mgl_fbo, vp_mat, view_mat)" in env_block


def test_env_model_render_failure_restores_gl_state():
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    render_func = render_text.split("def _render_env_model", 1)[1]
    render_body = render_func.split("        if self._env_perf_log:", 1)[0]

    assert "previous_depth_mask = self.ctx.depth_mask" in render_body
    assert "try:" in render_body
    assert "except Exception as exc:" in render_body
    assert "openxr_background_env_model_failed" in render_body
    assert "finally:" in render_body
    render_finally = render_body.split("finally:", 1)[1]
    assert "self.ctx.disable(moderngl.CULL_FACE)" in render_finally
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally
    assert "self.ctx.depth_mask = previous_depth_mask" in render_finally
    assert "glFrontFace(GL_CCW)" in render_finally
    assert "self._env_prog['u_use_texture'].value = 1" in render_finally
    assert "self._env_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)" in render_finally
    assert "self._env_prog['u_base_alpha'].value = 1.0" in render_finally


def test_env_model_initializer_preloads_panorama_light_mask(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._env_model_init_done = False
    viewer._environment_enabled = True
    viewer._panorama_background_path = "background.jpg"
    viewer._env_model_path = None
    viewer._panorama_called = False
    viewer._mask_called = False

    def _get_panorama_texture():
        viewer._panorama_called = True
        return object()

    def _get_panorama_light_mask_texture():
        viewer._mask_called = True
        return object()

    viewer._get_panorama_texture = _get_panorama_texture
    viewer._get_panorama_light_mask_texture = _get_panorama_light_mask_texture
    viewer._ensure_env_model_initialized("Test")

    assert viewer._panorama_called
    assert viewer._mask_called


def test_panorama_wall_light_mask_missing_is_diagnostic(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    viewer._panorama_background_path = str(tmp_path / "background.jpg")
    viewer._panorama_background_settings = {"wall_light_mask": "missing-mask.png"}
    viewer._panorama_light_mask_tex = None
    viewer._panorama_light_mask_path = None
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._get_panorama_light_mask_texture() is None
    assert viewer._get_panorama_light_mask_texture() is None

    assert inc_calls == [("openxr_wall_light_mask_missing", 1)]
    assert viewer._panorama_light_mask_missing_path.endswith("missing-mask.png")


def test_panorama_wall_light_mask_disabled_is_diagnostic_once(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._panorama_background_path = "background.jpg"
    viewer._panorama_background_settings = {}
    viewer._panorama_light_mask_tex = object()
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._get_panorama_light_mask_texture() is None
    assert viewer._get_panorama_light_mask_texture() is None
    assert viewer._panorama_light_mask_texture_ready() is None
    assert viewer._panorama_light_mask_texture_ready() is None

    assert inc_calls == [("openxr_wall_light_mask_disabled", 1)]


def test_panorama_wall_light_uses_gpu_light_probe_grid():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    pano_frag = glsl_text.split("_PANORAMA_FRAG", 1)[1].split("_GLOW_DOWNSAMPLE_FRAG", 1)[0]
    probe_func = pano_frag.split("vec3 screen_light_probe_color()", 1)[1].split("void main()", 1)[0]

    assert probe_func.count("textureLod(u_screen_light_tex") == 9
    assert "vec2(0.25, 0.25)" in probe_func
    assert "vec2(0.50, 0.50)" in probe_func
    assert "vec2(0.75, 0.75)" in probe_func
    assert "screen_light_probe_color()" in pano_frag.split("void main()", 1)[1]
    assert ".cpu(" not in pano_frag
    assert ".numpy(" not in pano_frag
    assert "glReadPixels" not in pano_frag


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
    base_text = (SRC / "xr_viewer" / "base.py").read_text(encoding="utf-8")
    no_room_glow = base_text.split("def _render_glow", 1)[1].split("def _render_shadow", 1)[0]

    assert "def _screen_effect_source_texture(self, *, allow_runtime_eye=True):" in effects_text
    assert effects_text.count("_screen_effect_source_texture(allow_runtime_eye=False)") == 4
    assert "_runtime_effect_safe_source_tex" in source_func
    assert "_promote_runtime_effect_ready_texture" in source_func
    assert "_runtime_eye_textures" not in source_func
    assert "_current_eye_index" not in source_func
    assert "def _screen_effect_source_texture(self):" in base_text
    assert "_runtime_effect_safe_source_tex" in base_text
    assert "_promote_runtime_effect_ready_texture" in base_text
    assert "_screen_effect_source_texture()" in no_room_glow
    assert "if getattr(self, '_runtime_direct_source', False):" in no_room_glow
    assert "_cached_glow_downsample_texture(source_tex, source_size)" in no_room_glow
    assert "else:\n            glow_tex = self._prepare_glow_downsample_texture" in no_room_glow
    assert "getattr(self, 'color_tex', None)" not in no_room_glow


def test_screen_effect_source_texture_is_cached_per_frame(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 11})()
    viewer._frame_count = 8
    viewer._runtime_direct_source = True
    viewer._runtime_effect_safe_source_tex = source_tex
    viewer._runtime_effect_safe_source_size = (1280, 720)
    viewer._runtime_effect_safe_source_frame_id = 4
    viewer._promote_count = 0
    viewer._age_count = 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    def _promote():
        viewer._promote_count += 1

    def _record_age(_source_tex):
        viewer._age_count += 1

    viewer._promote_runtime_effect_ready_texture = _promote
    viewer._record_screen_effect_safe_age = _record_age

    assert viewer._screen_effect_source_texture(allow_runtime_eye=False) == (source_tex, (1280, 720))
    assert viewer._screen_effect_source_texture(allow_runtime_eye=False) == (source_tex, (1280, 720))
    assert viewer._promote_count == 1
    assert viewer._age_count == 1
    assert ("openxr_screen_effect_source_reuse", 1) in inc_calls


def test_screen_effect_source_cache_refreshes_when_safe_source_changes(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    first_tex = type("Tex", (), {"glo": 21})()
    next_tex = type("Tex", (), {"glo": 22})()
    viewer._frame_count = 8
    viewer._runtime_direct_source = True
    viewer._runtime_effect_safe_source_tex = first_tex
    viewer._runtime_effect_safe_source_size = (1280, 720)
    viewer._runtime_effect_safe_source_frame_id = 4
    viewer._promote_count = 0
    viewer._age_count = 0
    viewer._breakdown_inc = lambda *args, **kwargs: None

    def _promote():
        viewer._promote_count += 1

    def _record_age(_source_tex):
        viewer._age_count += 1

    viewer._promote_runtime_effect_ready_texture = _promote
    viewer._record_screen_effect_safe_age = _record_age

    assert viewer._screen_effect_source_texture(allow_runtime_eye=False) == (first_tex, (1280, 720))
    viewer._runtime_effect_safe_source_tex = next_tex
    viewer._runtime_effect_safe_source_frame_id = 5
    assert viewer._screen_effect_source_texture(allow_runtime_eye=False) == (next_tex, (1280, 720))

    assert viewer._promote_count == 2
    assert viewer._age_count == 2


def test_no_room_screen_effect_source_texture_uses_safe_runtime_source(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 14})()
    viewer._frame_count = 6
    viewer._runtime_direct_source = True
    viewer._runtime_effect_safe_source_tex = source_tex
    viewer._runtime_effect_safe_source_size = (640, 360)
    viewer._runtime_effect_safe_source_frame_id = 2
    viewer._promote_count = 0
    viewer._age_count = 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    def _promote():
        viewer._promote_count += 1

    def _record_age(_source_tex):
        viewer._age_count += 1

    viewer._promote_runtime_effect_ready_texture = _promote
    viewer._record_screen_effect_safe_age = _record_age

    assert viewer._screen_effect_source_texture() == (source_tex, (640, 360))
    assert viewer._screen_effect_source_texture() == (source_tex, (640, 360))
    assert viewer._promote_count == 1
    assert viewer._age_count == 1
    assert ("openxr_screen_effect_source_reuse", 1) in inc_calls


def test_no_room_screen_effect_source_texture_caches_color_source(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 16})()
    viewer._frame_count = 7
    viewer._runtime_direct_source = False
    viewer.color_tex = source_tex
    viewer._texture_size = (320, 180)
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._screen_effect_source_texture() == (source_tex, (320, 180))
    assert viewer._screen_effect_source_texture() == (source_tex, (320, 180))

    assert ("openxr_screen_effect_source_reuse", 1) in inc_calls


def test_screen_effect_safe_age_records_once_per_safe_texture_per_frame(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 12})()
    other_tex = type("Tex", (), {"glo": 13})()
    values = []
    viewer._frame_count = 9
    viewer._runtime_effect_safe_source_frame_id = 5
    viewer._breakdown_add_value = lambda name, value: values.append((name, value))

    viewer._record_screen_effect_safe_age(source_tex)
    viewer._record_screen_effect_safe_age(source_tex)
    viewer._record_screen_effect_safe_age(other_tex)

    assert values == [
        ("openxr_effect_ready_age_frames", 4.0),
        ("openxr_effect_ready_age_frames", 4.0),
    ]


def test_glow_downsample_cache_is_shared_across_eyes():
    quality_text = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    key_func = quality_text.split("def _glow_downsample_key_and_size", 1)[1].split(
        "def _cached_glow_downsample_texture", 1
    )[0]
    cached_func = quality_text.split("def _cached_glow_downsample_texture", 1)[1].split(
        "def _prepare_glow_downsample_texture", 1
    )[0]
    prepare_func = quality_text.split("def _prepare_glow_downsample_texture", 1)[1].split(
        "def _is_runtime_eye_texture_ready", 1
    )[0]
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    shell_func = effects_text.split("def _render_glow_shell", 1)[1].split(
        "def _render_screen_background_effects", 1
    )[0]

    assert "_glow_ds_cache_key" in cached_func
    assert "_current_eye_index" not in key_func + cached_func + prepare_func
    assert "_runtime_effect_safe_source_frame_id" in key_func
    assert "source_frame_id if source_frame_id is not None else getattr(self, '_frame_count', 0)" in key_func
    assert "_cached_glow_downsample_texture(source_tex, source_size)" in prepare_func
    assert "if getattr(self, '_runtime_direct_source', False):" in shell_func
    assert "_cached_glow_downsample_texture(source_tex, source_size)" in shell_func
    assert "else:\n            glow_tex = self._prepare_glow_downsample_texture" in shell_func
    assert "except Exception as exc:" in prepare_func
    assert "openxr_glow_downsample_failed" in prepare_func
    assert "finally:" in prepare_func
    render_finally = prepare_func.split("finally:", 1)[1]
    assert "self.ctx.viewport = prev_viewport" in render_finally
    assert "self.ctx.depth_mask = prev_depth_mask" in render_finally
    assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally


def test_screen_quality_pass_restores_gl_state_on_failure():
    quality_text = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    func = quality_text.split("def _prepare_screen_quality_texture", 1)[1].split(
        "def _ensure_glow_downsample_resources", 1
    )[0]

    assert "try:" in func
    assert "except Exception as exc:" in func
    assert "openxr_screen_quality_failed" in func
    assert "return None" in func
    assert "finally:" in func
    render_finally = func.split("finally:", 1)[1]
    assert "self.ctx.viewport = prev_viewport" in render_finally
    assert "self.ctx.depth_mask = prev_depth_mask" in render_finally
    assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally


def test_screen_light_uses_effect_source_texture_not_runtime_eye_texture():
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    source_func = render_text.split("def _screen_light_source_texture", 1)[1].split("def _apply_cinema_light_uniforms", 1)[0]
    runtime_direct_block = source_func.split("if getattr(self, '_runtime_direct_source', False):", 1)[1].split(
        "source_tex = getattr(self, 'color_tex'", 1
    )[0]

    assert "_runtime_effect_safe_source_tex" in source_func
    assert "_runtime_effect_safe_source_size" in source_func
    assert "_promote_runtime_effect_ready_texture" in source_func
    assert "_record_screen_effect_safe_age" in source_func
    assert "_prepare_glow_downsample_texture" in source_func
    assert "_prepare_glow_downsample_texture" not in runtime_direct_block
    assert "_cached_glow_downsample_texture" in runtime_direct_block
    assert "_glow_ds_size" in source_func
    assert "_runtime_eye_textures" not in source_func
    assert "_current_eye_index" not in source_func


def test_screen_light_source_texture_reuses_prewarmed_downsample(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 7})()
    light_tex = object()
    viewer._frame_count = 12
    viewer._runtime_direct_source = True
    viewer._runtime_effect_safe_source_tex = source_tex
    viewer._runtime_effect_safe_source_size = (1920, 1080)
    viewer._runtime_effect_safe_source_frame_id = 3
    viewer._glow_ds_tex = light_tex
    viewer._glow_ds_size = (96, 54)
    viewer._glow_ds_cache_key = (3, 7, 1920, 1080, 96, 54)
    viewer._promote_count = 0
    viewer._age_count = 0
    viewer._prepare_count = 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._record_screen_effect_safe_age = lambda _source_tex: setattr(viewer, "_age_count", viewer._age_count + 1)

    def _promote():
        viewer._promote_count += 1

    def _prepare(_source_tex, _source_size):
        viewer._prepare_count += 1
        return object()

    viewer._promote_runtime_effect_ready_texture = _promote
    viewer._prepare_glow_downsample_texture = _prepare

    assert viewer._screen_light_source_texture() == (light_tex, (96, 54))
    assert viewer._screen_light_source_texture() == (light_tex, (96, 54))
    assert viewer._promote_count == 1
    assert viewer._age_count == 1
    assert viewer._prepare_count == 0
    assert ("openxr_screen_light_downsample_source", 1) in inc_calls
    assert ("openxr_screen_light_source_reuse", 1) in inc_calls


def test_screen_light_source_cache_refreshes_when_safe_source_changes(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    first_tex = type("Tex", (), {"glo": 31})()
    next_tex = type("Tex", (), {"glo": 32})()
    viewer._frame_count = 12
    viewer._runtime_direct_source = True
    viewer._runtime_effect_safe_source_tex = first_tex
    viewer._runtime_effect_safe_source_size = (1920, 1080)
    viewer._runtime_effect_safe_source_frame_id = 3
    viewer._promote_count = 0
    viewer._age_count = 0
    viewer._prepare_count = 0
    viewer._breakdown_inc = lambda *args, **kwargs: None
    viewer._record_screen_effect_safe_age = lambda _source_tex: setattr(viewer, "_age_count", viewer._age_count + 1)

    def _promote():
        viewer._promote_count += 1

    def _prepare(_source_tex, _source_size):
        viewer._prepare_count += 1
        return object()

    viewer._promote_runtime_effect_ready_texture = _promote
    viewer._prepare_glow_downsample_texture = _prepare

    assert viewer._screen_light_source_texture() == (first_tex, (1920, 1080))
    viewer._runtime_effect_safe_source_tex = next_tex
    viewer._runtime_effect_safe_source_frame_id = 4
    assert viewer._screen_light_source_texture() == (next_tex, (1920, 1080))

    assert viewer._promote_count == 2
    assert viewer._age_count == 2
    assert viewer._prepare_count == 0


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

    viewer._screen_light_intensity = 3.5
    viewer._panorama_background_path = "background.jpg"

    assert viewer._runtime_effects_need_source_texture()

    viewer._panorama_background_path = None
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]

    assert viewer._runtime_effects_need_source_texture()

    viewer._openxr_async_effects_enabled = False
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0

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
    assert "ensure_staging(self.ctx, w, h)" in core_text
    assert "ctx.texture((w, h), 4, dtype='f1')" in core_text
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


def test_no_room_glow_pass_restores_gl_state_on_failure():
    base_text = (SRC / "xr_viewer" / "base.py").read_text(encoding="utf-8")
    func = base_text.split("def _render_glow", 1)[1].split("def _render_shadow", 1)[0]

    assert "previous_depth_mask = self.ctx.depth_mask" in func
    assert "try:" in func
    assert "except Exception as exc:" in func
    assert "openxr_screen_glow_failed" in func
    assert "finally:" in func
    render_finally = func.split("finally:", 1)[1]
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally
    assert "self.ctx.depth_mask = previous_depth_mask" in render_finally
def test_screen_effect_passes_restore_gl_state_on_failure():
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    checks = [
        ("def _render_glow", "def _render_frosted_glow", "openxr_screen_glow_failed"),
        ("def _render_frosted_glow", "def _render_frosted_veil", "openxr_frosted_glow_failed"),
        ("def _render_frosted_veil", "def _render_glow_shell", "openxr_frosted_veil_failed"),
        ("def _render_glow_shell", "def _render_screen_background_effects", "openxr_glow_shell_failed"),
    ]

    for start, end, diagnostic in checks:
        func = effects_text.split(start, 1)[1].split(end, 1)[0]
        assert "previous_depth_mask = self.ctx.depth_mask" in func
        assert "try:" in func
        assert "except Exception as exc:" in func
        assert diagnostic in func
        assert "finally:" in func
        render_finally = func.split("finally:", 1)[1]
        assert "self.ctx.disable(moderngl.BLEND)" in render_finally
        assert "self.ctx.depth_mask = previous_depth_mask" in render_finally
        assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally


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
    runtime_eye_text = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    renderer_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    screen_quality_text = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    base_text = (SRC / "xr_viewer" / "base.py").read_text(encoding="utf-8")
    source_func = effects_text.split("def _screen_effect_source_texture", 1)[1].split("def _render_glow", 1)[0]
    light_func = renderer_text.split("def _screen_light_source_texture", 1)[1].split(
        "def _apply_cinema_light_uniforms", 1
    )[0]
    downsample_func = screen_quality_text.split("def _prepare_glow_downsample_texture", 1)[1].split(
        "def ", 1
    )[0]

    assert not (SRC / "xr_viewer" / "core_glow.py").exists()
    assert "def _sample_glow_target_color" not in upload_text
    assert "_maybe_sample_glow_target_color" not in upload_text
    assert "_maybe_sample_glow_target_color" not in runtime_eye_text
    assert "_mark_upload('sample_glow')" not in upload_text
    assert "_advance_glow_color" not in effects_text
    assert "_glow_target_color" not in effects_text
    assert "_glow_target_color" not in impl_text
    assert "_glow_target_color" not in base_text
    assert "_screen_light_target_colors" not in effects_text
    assert "_screen_light_target_colors" not in impl_text
    assert "_screen_light_colors" not in effects_text
    assert "_screen_light_colors" not in impl_text
    assert "_glow_color_counter" not in impl_text
    for realtime_func in (source_func, light_func, downsample_func):
        assert ".cpu(" not in realtime_func
        assert ".numpy(" not in realtime_func
        assert "glReadPixels" not in realtime_func
        assert ".read(" not in realtime_func
        assert "synchronize(" not in realtime_func


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
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    profile_text = (SRC / "xr_viewer" / "environment_profiles.py").read_text(encoding="utf-8")
    ctrl_frag = glsl_text.split("_CTRL_FRAG", 1)[1].split("_ENV_VERT", 1)[0]

    assert "uniform sampler2D u_env_tex" in ctrl_frag
    assert "uniform sampler2D u_screen_light_tex" in ctrl_frag
    assert "uniform int u_env_stereo_layout" in ctrl_frag
    assert "uniform int u_env_eye_index" in ctrl_frag
    assert "vec2 env_sample_uv(vec3 dir)" in ctrl_frag
    assert "uv.x = uv.x * 0.5 + (u_env_eye_index == 1 ? 0.5 : 0.0)" in ctrl_frag
    assert "textureLod(u_env_tex, env_sample_uv(R), 3.0)" in ctrl_frag
    assert "textureLod(u_env_tex, env_sample_uv(N), 5.0)" in ctrl_frag
    assert "textureLod(u_screen_light_tex" in ctrl_frag
    assert "u_light_color" not in ctrl_frag
    assert "u_ambient_color" not in ctrl_frag
    assert "_panorama_texture_ready" in render_text
    assert "_get_panorama_texture" not in render_text
    assert "_bind_screen_light_source_texture(location=10)" in render_text
    assert "_runtime_eye_textures" not in render_text
    assert "_controller_hdr_lighting" in render_text
    assert "if getattr(self, '_controller_hdr_lighting', True):" in render_text
    assert "_panorama_render_settings()" in render_text
    assert "u_env_stereo_layout" in render_text
    assert "u_env_eye_index" in render_text
    assert "u_env_stereo_layout" in impl_text
    assert "u_env_eye_index" in impl_text
    assert "u_use_env_tex" in render_text
    assert "u_screen_light_enabled" in render_text
    assert "controller_hdr_lighting" in profile_text
    assert "controller_hdr_reflection" in profile_text
    assert "== '.hdr'" in profile_text


def test_environment_light_binds_latest_safe_texture_before_uniform_cache_skip():
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    profiles_text = (SRC / "xr_viewer" / "environment_profiles.py").read_text(encoding="utf-8")
    apply_func = render_text.split("def _apply_cinema_light_uniforms", 1)[1].split("def _get_panorama_texture", 1)[0]
    before_skip = apply_func.split("if state_key == last_state_key", 1)[0]

    assert not (SRC / "xr_viewer" / "render.py").exists()
    assert "from .render import" not in render_text
    assert "def _view_mat_inv" in render_text
    assert "def _bind_screen_light_source_texture" in render_text
    assert "self._bind_screen_light_source_texture()" in before_skip
    assert "source_tex.use(location=8)" not in apply_func
    assert "_screen_light_dynamic" not in render_text
    assert "_screen_light_dynamic" not in impl_text
    assert "_screen_light_dynamic" not in profiles_text
    assert "_screen_light_sample_interval" not in impl_text
    assert "_screen_light_sample_interval" not in profiles_text
    assert "screen_light_lerp" not in render_text
    assert "screen_light_lerp" not in impl_text
    assert "screen_light_lerp" not in profiles_text


def test_controller_touch_bindings_share_profile_suggestion_call():
    source = (SRC / "xr_viewer" / "core_controller_actions.py").read_text(encoding="utf-8")

    assert source.count("xr.suggest_interaction_profile_bindings") == 1
    assert '"/user/hand/left/input/thumbstick/touch"' in source
    assert '"/user/hand/left/input/grip/pose"' in source
