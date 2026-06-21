from viewer.settings import resolve_viewer_settings


BASE_SETTINGS = {
    "Monitor Index": 1,
    "Display Mode": "Half-SBS",
    "Stereo Output": None,
    "Processing Resolution": "Auto",
    "Capture Mode": "Monitor",
    "Window Title": "",
    "Target FPS": 60,
    "Language": "EN",
    "Show FPS": False,
    "Depth Strength": 2.0,
    "IPD": 0.064,
    "Convergence": 0.0,
    "Fill 16:9": True,
    "Upscaler": "Off",
    "Upscaler Sharpness": 0.35,
    "Controller Model": "PICO",
    "Environment Model": "None",
    "XR Preview Window": False,
}


def test_resolve_viewer_settings_reads_vsync_key():
    settings = dict(BASE_SETTINGS, VSync=False)

    resolved = resolve_viewer_settings(settings)

    assert resolved.local_vsync is False


def test_resolve_viewer_settings_requires_vsync_key():
    settings = dict(BASE_SETTINGS)

    try:
        resolve_viewer_settings(settings)
    except KeyError as exc:
        assert exc.args == ("VSync",)
    else:
        raise AssertionError("resolve_viewer_settings should require VSync")
