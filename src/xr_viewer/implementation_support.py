# Desktop2Stereo OpenXR viewer: small shared implementation helpers.

import os

try:
    import xr
except ImportError:
    xr = None


def _openxr_app_api_version():
    """Request OpenXR 1.0 for broad runtime compatibility."""
    if xr is not None and hasattr(xr, "Version"):
        return xr.Version(1, 0, 0)
    return xr.XR_CURRENT_API_VERSION


def _openxr_optional_extensions(*names):
    if xr is None:
        return []
    try:
        available = {
            prop.extension_name.decode('ascii')
            if isinstance(prop.extension_name, bytes)
            else str(prop.extension_name)
            for prop in xr.enumerate_instance_extension_properties()
        }
    except Exception:
        return []
    return [name for name in names if name and name in available]


def _openxr_requested_display_refresh_rate():
    raw = str(os.environ.get("D2S_OPENXR_DISPLAY_REFRESH_RATE", "90") or "").strip().lower()
    if raw in {"", "0", "off", "false", "none"}:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0.0 else None


def _openxr_refresh_rate_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(getattr(value, "value"))


def _request_openxr_display_refresh_rate(session, *, quiet=False):
    requested = _openxr_requested_display_refresh_rate()
    if xr is None or requested is None:
        return
    enumerate_rates = getattr(xr, "enumerate_display_refresh_rates_fb", None)
    get_rate = getattr(xr, "get_display_refresh_rate_fb", None)
    request_rate = getattr(xr, "request_display_refresh_rate_fb", None)
    if not callable(enumerate_rates) or not callable(get_rate) or not callable(request_rate):
        if not quiet:
            print("[OpenXRViewer] Display refresh rate extension unavailable")
        return
    try:
        available = [_openxr_refresh_rate_float(rate) for rate in enumerate_rates(session)]
        current = _openxr_refresh_rate_float(get_rate(session))
    except Exception as exc:
        if not quiet:
            print(f"[OpenXRViewer] Display refresh rate query failed: {exc}")
        return
    if not quiet:
        print(
            f"[OpenXRViewer] Display refresh rates: available={available} "
            f"current={current:.2f} requested={requested:.2f}"
        )
    if available and all(abs(rate - requested) > 0.5 for rate in available):
        if not quiet:
            print(f"[OpenXRViewer] Display refresh rate {requested:.2f} not advertised by runtime")
        return
    try:
        request_rate(session, requested)
        after = _openxr_refresh_rate_float(get_rate(session))
        if not quiet:
            print(f"[OpenXRViewer] Display refresh rate after request: {after:.2f}")
    except Exception as exc:
        if not quiet:
            print(f"[OpenXRViewer] Display refresh rate request failed: {type(exc).__name__}: {exc!r}")


def _float_option(kwargs, key, env_name, default, min_value=None, max_value=None):
    raw = kwargs.get(key, os.environ.get(env_name, default))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = float(default)
    if min_value is not None:
        value = max(float(min_value), value)
    if max_value is not None:
        value = min(float(max_value), value)
    return value
