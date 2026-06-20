"""OpenXR viewer split package with runtime-direct GPU interop support."""


def __getattr__(name):
    if name == "BaseOpenXRViewer":
        from .base import OpenXRViewer

        value = OpenXRViewer
    elif name == "EnvironmentOpenXRViewer":
        from .environment import OpenXRViewer

        value = OpenXRViewer
    elif name == "OpenXRViewer":
        from .base import OpenXRViewer as value
    elif name in {"OPENXR_AVAILABLE", "OpenXRViewerCore", "load_glb_model"}:
        from . import implementation

        value = getattr(implementation, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "OPENXR_AVAILABLE",
    "OpenXRViewer",
    "OpenXRViewerCore",
    "BaseOpenXRViewer",
    "EnvironmentOpenXRViewer",
    "load_glb_model",
]