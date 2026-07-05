import ctypes

try:
    import xr
except ImportError:
    xr = None


class ScreenLayerPresenter:
    def __init__(self, viewer):
        self.viewer = viewer
        self._frame_projection_layer = None
        self._frame_quad_layers = []

    def update_or_reuse(self, *, screen_frame_uploaded=False):
        return self.viewer._update_quad_layer_swapchains(force=screen_frame_uploaded)

    def prepare_frame_layers(self, *, screen_frame_uploaded=False):
        self._frame_projection_layer = None
        self._frame_quad_layers = []
        updated_quad_eyes = self.update_or_reuse(screen_frame_uploaded=screen_frame_uploaded)
        quad_layers, quad_layer_headers, updated_quad_eyes = self.make_quad_layers(updated_quad_eyes)
        self._frame_quad_layers = quad_layers
        render_projection_layer = self.viewer._projection_layer_needed()
        if not render_projection_layer:
            self.viewer._breakdown_inc('openxr_projection_layer_skipped')
        return quad_layers, quad_layer_headers, updated_quad_eyes, render_projection_layer

    def append_frame_layers(self, composition_layers, *, projection_views=(), projection_space=None, quad_layer_headers=()):
        if projection_views:
            projection_layer = xr.CompositionLayerProjection(
                space=projection_space,
                views=projection_views,
            )
            self._frame_projection_layer = projection_layer
            composition_layers.append(
                ctypes.cast(
                    ctypes.pointer(projection_layer),
                    ctypes.POINTER(xr.CompositionLayerBaseHeader),
                )
            )
        composition_layers.extend(quad_layer_headers)
        return composition_layers

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
