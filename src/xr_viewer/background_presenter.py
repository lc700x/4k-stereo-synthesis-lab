import time

from OpenGL.GL import GL_DEPTH_BUFFER_BIT, glClear

from .background_layer_renderer import BackgroundLayerRenderer


class BackgroundPresenter:
    def __init__(self, viewer):
        self.viewer = viewer

    def projection_fallback_needed(self):
        renderer = getattr(self.viewer, '_background_layer_renderer', None)
        if renderer is None:
            renderer = BackgroundLayerRenderer(self.viewer)
            self.viewer._background_layer_renderer = renderer
        return renderer.panorama_ready() and not renderer.native_background_available()

    def render_projection_background(self, mgl_fbo, view_mat, proj_mat, vp_mat, *, eye_index, projection_screen_enabled):
        viewer = self.viewer
        start = time.perf_counter()
        rendered = False
        has_panorama = self.projection_fallback_needed()
        enabled = bool(projection_screen_enabled or has_panorama)
        if enabled and has_panorama:
            if viewer._render_panorama_background(mgl_fbo, view_mat, proj_mat):
                if eye_index == 0:
                    viewer._breakdown_inc('openxr_background_panorama')
                rendered = True
                mgl_fbo.use()
                glClear(GL_DEPTH_BUFFER_BIT)
        if not rendered and eye_index == 0:
            viewer._breakdown_inc('openxr_background_idle')
        viewer._breakdown_add_time('openxr_background', time.perf_counter() - start)
        return rendered
