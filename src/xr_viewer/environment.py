# xrviewer_env.py
# Desktop2Stereo OpenXR viewer: room/environment profile.
# Shared runtime/rendering code is in xrviewer_core.py; room-specific code lives here.

from .implementation import *
from .environment_effects import EnvironmentEffectsMixin
from .environment_layout import EnvironmentLayoutMixin
from .environment_model import EnvironmentModelMixin
from .environment_profiles import EnvironmentProfileMixin
from .environment_renderer import EnvironmentRendererMixin
from .overlay import OverlayMixin


class OpenXRViewer(EnvironmentRendererMixin, EnvironmentEffectsMixin, EnvironmentModelMixin, EnvironmentLayoutMixin, EnvironmentProfileMixin, OpenXRViewerCore, OverlayMixin):
    """Room/environment viewer.

    This class keeps the environment-specific behavior separate from the normal
    no-room viewer: room discovery, profile.json layout, GLB room loading,
    environment switching, and environment rendering.
    """

    ENVIRONMENT_MODE = True
    DEFAULT_ENVIRONMENT_MODEL = 'Default'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('environment_model', self.DEFAULT_ENVIRONMENT_MODEL)
        super().__init__(*args, **kwargs)

# Standalone smoke test helper shared by the viewer entry modules.
def _smoke_test(viewer_cls):
    if not OPENXR_AVAILABLE:
        print("[TEST] pyopenxr not available - cannot run standalone test")
        sys.exit(1)

    import queue as _q
    W, H = 1280, 720
    white_rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    zero_depth = np.zeros((H, W), dtype=np.float32)

    depth_q = _q.Queue(maxsize=2)
    depth_q.put((white_rgb, zero_depth, time.perf_counter()))

    viewer = viewer_cls(
        frame_size=(W, H),
        fps=60,
        depth_q=depth_q,
        show_fps=True,
    )

    try:
        viewer.run(first_rgb=white_rgb, first_depth=zero_depth)
    except KeyboardInterrupt:
        print("[TEST] Interrupted")
    finally:
        viewer.cleanup()


def _run_standalone_test():
    _smoke_test(OpenXRViewer)


if __name__ == "__main__":
    _run_standalone_test()
