from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"


def _read_impl():
    return (SRC_ROOT / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")


def test_openxr_triggers_use_touch_injection_before_mouse_fallback():
    source = _read_impl()
    trigger_body = source.split("    def _handle_triggers(self):", 1)[1].split("    def _press_key", 1)[0]

    assert "_TOUCH_AVAILABLE" in trigger_body
    assert "_touch_injector.set" in trigger_body
    assert "_touch_injector.flush()" in trigger_body
    assert "_TOUCH_PINCH_SPREAD_GAIN" in trigger_body
    assert trigger_body.index("_touch_injector.flush()") < trigger_body.index("left_laser_usable")


def test_openxr_cursor_publishes_per_hand_touch_positions():
    source = _read_impl()
    cursor_body = source.split("    def _handle_cursor(self):", 1)[1].split("    def _send_arrow", 1)[0]

    for token in (
        "self._touch_px_l",
        "self._touch_px_r",
        "self._touch_valid_l",
        "self._touch_valid_r",
        "self._overlay_hit_l",
        "self._overlay_hit_r",
        "KB_CURSOR_PRIORITY_BIAS",
        "self._cursor_click_ts_l",
        "both_touch_down",
    ):
        assert token in cursor_body
