from pathlib import Path


XR_IMPL = Path(__file__).resolve().parents[1] / "src" / "xr_viewer" / "implementation.py"


def test_right_grip_screen_size_and_distance_use_exponential_stick_speed():
    text = XR_IMPL.read_text(encoding="utf-8")

    assert "def _stick_exp_speed" in text
    assert "self._size_speed_base = 0.18" in text
    assert "self._size_speed_max  = 1.2" in text
    assert "self._dist_speed_base = 0.35" in text
    assert "self._dist_speed_exp  = 3.0" in text

    branch_start = text.index("if grip_r and not grip_l and not seat_adjust_active:")
    branch_end = text.index("elif grip_l and not grip_r and not seat_adjust_active:", branch_start)
    branch = text[branch_start:branch_end]

    assert "self._stick_exp_speed(rx, self._size_speed_base, self._size_speed_max, self._size_speed_exp)" in branch
    assert "math.copysign(_speed * dt, rx)" in branch
    assert "self._stick_exp_speed(ry, self._dist_speed_base, self._dist_speed_max, self._dist_speed_exp)" in branch
    assert "RESIZE_SPEED" not in branch
