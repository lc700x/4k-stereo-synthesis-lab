import time

from OpenGL.GL import GL_DEPTH_BUFFER_BIT, glClear


class BackgroundPresenter:
    def __init__(self, viewer):
        self.viewer = viewer

    def render_projection_background(self, mgl_fbo, view_mat, proj_mat, vp_mat, *, eye_index, enabled):
        viewer = self.viewer
        start = time.perf_counter()
        rendered = False
        if enabled and getattr(viewer, '_panorama_background_path', None):
            if viewer._render_panorama_background(mgl_fbo, view_mat, proj_mat):
                if eye_index == 0:
                    viewer._breakdown_inc('openxr_background_panorama')
                rendered = True
                mgl_fbo.use()
                glClear(GL_DEPTH_BUFFER_BIT)
        if (
            enabled
            and not getattr(viewer, '_panorama_background_path', None)
            and getattr(viewer, '_env_model_visible', False)
            and getattr(viewer, '_env_model_prims', None)
        ):
            if eye_index == 0:
                viewer._breakdown_inc('openxr_background_env_model')
            rendered = True
            viewer._render_env_model(mgl_fbo, vp_mat, view_mat)
            mgl_fbo.use()
            glClear(GL_DEPTH_BUFFER_BIT)
        if not rendered and eye_index == 0:
            viewer._breakdown_inc('openxr_background_idle')
        viewer._breakdown_add_time('openxr_background', time.perf_counter() - start)
        return rendered
