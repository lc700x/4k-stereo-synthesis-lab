import ctypes

try:
    import xr
except ImportError:
    xr = None


class ScreenLayerPresenter:
    def __init__(self, viewer):
        self.viewer = viewer

    def update_or_reuse(self, *, screen_frame_uploaded=False):
        return self.viewer._update_quad_layer_swapchains(force=screen_frame_uploaded)

    def make_quad_layers(self, updated_quad_eyes):
        viewer = self.viewer
        quad_layers = []
        quad_layer_headers = []
        for quad_eye_index in updated_quad_eyes:
            try:
                quad_layer = viewer._make_quad_layer(quad_eye_index)
                if quad_layer is None:
                    raise RuntimeError(f"missing quad layer for eye {quad_eye_index}")
                quad_layers.append(quad_layer)
                quad_layer_headers.append(
                    ctypes.cast(
                        ctypes.pointer(quad_layer),
                        ctypes.POINTER(xr.CompositionLayerBaseHeader),
                    )
                )
            except Exception as exc:
                viewer._xr_quad_layer_active = False
                viewer._xr_quad_layer_failed = True
                viewer._breakdown_inc('openxr_quad_layer_failed')
                print(f"[OpenXRViewer] Quad layer build failed: {type(exc).__name__}: {exc}")
                return [], [], []
        return quad_layers, quad_layer_headers, list(updated_quad_eyes)
