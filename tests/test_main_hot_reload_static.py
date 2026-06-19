
from pathlib import Path


def test_main_has_limited_stereo_hot_reload_path():
    source = Path(__file__).resolve().parents[1] / "src" / "main.py"
    text = source.read_text(encoding="utf-8")

    assert "def _apply_stereo_hot_reload_if_needed" in text
    assert "def _hot_reload_value_snapshot" in text
    assert "_apply_stereo_hot_reload_if_needed()" in text
    assert '"depth_strength"' in text
    assert '"convergence"' in text
    assert '"ipd_mm"' in text
    assert '"stereo_scale"' in text
    assert '"max_shift_ratio"' in text
    assert '"cross_eyed"' in text
    assert '"temporal_strength"' in text
    assert '"scene_reset_threshold"' in text
    assert '"reset_cooldown_frames"' in text
    assert '"foreground_scale"' in text
    assert "def _clamp_foreground_scale_hot_reload" in text
    assert "_clamp_foreground_scale_hot_reload(settings_dict.get(\"Foreground Scale\"" in text
    assert '"depth_antialias_strength"' in text
    assert '"edge_dilation"' in text
    assert '"edge_threshold"' in text
    assert '"anaglyph_method"' in text
    assert '"depth_safety"' not in text
    assert "def _to_bool_hot_reload" in text
    assert "def _to_optional_bool_hot_reload" not in text
    assert "stereo_runtime.config = replace(stereo_runtime.config, **values)" in text
    assert "reset_temporal=False" in text

def test_stereo_warmup_uses_runtime_frame_shape_and_dedup_key():
    source = Path(__file__).resolve().parents[1] / "src" / "main.py"
    text = source.read_text(encoding="utf-8")

    assert "def _stereo_warmup_key(rgb_frame):" in text
    assert "shape = tuple(getattr(rgb_frame, \"shape\", ()))" in text
    assert "_stereo_warmup_keys" in text
    assert "_warmup_stereo_once_for_frame(runtime_rgb)" in text
    key_start = text.index("def _stereo_warmup_key(rgb_frame):")
    key_end = text.index("def _warmup_stereo_once_for_frame", key_start)
    key_block = text[key_start:key_end]
    assert "foreground_scale" not in key_block
    assert "depth_antialias_strength" not in key_block
    assert "warmup_stereo_kernels_for_frame(rgb_frame)" in text

    runtime_source = Path(__file__).resolve().parents[1] / "src" / "stereo_runtime" / "runtime.py"
    runtime_text = runtime_source.read_text(encoding="utf-8")
    assert "foreground_scales =" in runtime_text
    assert "antialias_values =" in runtime_text
    assert "def warmup_stereo_kernels_for_frame(self, rgb_frame: torch.Tensor)" in runtime_text
    assert "_, height, width = rgb_frame.shape" in runtime_text
    assert "3840" not in runtime_text[runtime_text.index("def warmup_stereo_kernels_for_frame"):runtime_text.index("def reset_stats")]

def test_stereo_scene_auto_switch_removed():
    source = Path(__file__).resolve().parents[1] / "src" / "main.py"
    text = source.read_text(encoding="utf-8")

    assert "AutoModeRuntime" not in text
    assert "AutoModeSignals" not in text
    assert "_update_auto_stereo_mode(runtime_rgb)" not in text
    assert "_ensure_auto_signal_sampler()" not in text
    assert "auto={'on' if stereo_auto_enabled else 'off'}" not in text
    assert 'return False, "cinema" if preset == "auto" else preset' in text