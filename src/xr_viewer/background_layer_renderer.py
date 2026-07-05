import ctypes

try:
    import xr
except ImportError:
    xr = None


class BackgroundLayerRenderer:
    def __init__(self, viewer):
        self.viewer = viewer
        self._frame_background_layers = []

    def panorama_ready(self):
        ready = getattr(self.viewer, '_panorama_texture_ready', None)
        return bool(callable(ready) and ready() is not None)

    def native_background_available(self):
        if xr is None or not self.panorama_ready():
            return False
        return bool(getattr(self.viewer, '_openxr_equirect_background_supported', False))

    def make_background_layers(self):
        self._frame_background_layers = []
        if not self.panorama_ready():
            return [], False
        if not self.native_background_available():
            self.viewer._breakdown_inc('openxr_background_projection_fallback')
            return [], True
        make_layer = getattr(self.viewer, '_make_equirect_background_layer', None)
        if not callable(make_layer):
            self.viewer._breakdown_inc('openxr_background_projection_fallback')
            return [], True
        try:
            layer = make_layer()
        except Exception as exc:
            print(f"[OpenXRViewer] Background equirect layer failed: {type(exc).__name__}: {exc}")
            self.viewer._breakdown_inc('openxr_background_layer_failed')
            return [], True
        if layer is None:
            self.viewer._breakdown_inc('openxr_background_projection_fallback')
            return [], True
        self._frame_background_layers = [layer]
        self.viewer._breakdown_inc('openxr_background_layer')
        return [ctypes.cast(ctypes.pointer(layer), ctypes.POINTER(xr.CompositionLayerBaseHeader))], False
