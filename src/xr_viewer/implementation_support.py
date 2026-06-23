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