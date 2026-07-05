import ctypes
import math

import moderngl
from OpenGL.GL import (
    glBindFramebuffer,
    glCheckFramebufferStatus,
    glDeleteFramebuffers,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGetError,
    GL_COLOR_ATTACHMENT0,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_TEXTURE_2D,
)

try:
    import xr
except ImportError:
    xr = None


class BackgroundLayerRenderer:
    def __init__(self, viewer):
        self.viewer = viewer
        self._frame_background_layers = []

    def _panorama_texture(self):
        ready = getattr(self.viewer, '_panorama_texture_ready', None)
        return ready() if callable(ready) else None

    def panorama_ready(self):
        return self._panorama_texture() is not None

    def native_background_available(self):
        if xr is None or not hasattr(xr, 'CompositionLayerEquirect2KHR') or not self.panorama_ready():
            return False
        return bool(
            getattr(self.viewer, '_openxr_equirect_background_supported', False)
            and getattr(self.viewer, '_background_equirect_swapchain', None) is not None
            and getattr(self.viewer, '_background_equirect_size', None) is not None
        )

    def _get_or_create_equirect_fbo(self, image_index, gl_tex, width, height):
        cache = self.viewer._background_equirect_fbo_cache
        key = int(image_index)
        cached = cache.get(key)
        if cached and cached[2] == width and cached[3] == height:
            return cached[0]
        if cached:
            try:
                cached[0].release()
            except Exception:
                pass
            try:
                glDeleteFramebuffers(1, [cached[1]])
            except Exception:
                pass
        while glGetError() != 0:
            pass
        raw_fbo = int(glGenFramebuffers(1))
        glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Equirect background FBO incomplete: {status:#x}")
        mgl_fbo = self.viewer.ctx.detect_framebuffer(raw_fbo)
        cache[key] = (mgl_fbo, raw_fbo, width, height)
        return mgl_fbo

    def _source_key(self, tex):
        return (
            int(getattr(tex, 'glo', 0) or 0),
            tuple(getattr(tex, 'size', ()) or ()),
            getattr(self.viewer, '_panorama_tex_path', None),
        )

    def _upload_equirect_texture(self, tex):
        viewer = self.viewer
        swapchain = viewer._background_equirect_swapchain
        width, height = viewer._background_equirect_size
        source_key = self._source_key(tex)
        if viewer._background_equirect_uploaded_key == source_key:
            return
        img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
        viewer._wait_swapchain_image(swapchain)
        released = False
        prev_viewport = getattr(viewer.ctx, 'viewport', None)
        prev_depth_mask = getattr(viewer.ctx, 'depth_mask', None)
        try:
            sc_image = viewer._background_equirect_images[img_index]
            mgl_fbo = self._get_or_create_equirect_fbo(img_index, sc_image.image, width, height)
            mgl_fbo.use()
            viewer.ctx.viewport = (0, 0, width, height)
            viewer.ctx.disable(moderngl.DEPTH_TEST)
            viewer.ctx.disable(moderngl.BLEND)
            viewer.ctx.depth_mask = False
            tex.use(location=0)
            viewer._quad_copy_prog['u_flip_y'].value = 0
            viewer._quad_copy_vao.render(moderngl.TRIANGLE_STRIP)
            xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
            released = True
            viewer._background_equirect_uploaded_key = source_key
            viewer._breakdown_inc('openxr_background_layer_upload')
        finally:
            if not released:
                try:
                    xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                except Exception:
                    pass
            if prev_viewport is not None:
                viewer.ctx.viewport = prev_viewport
            if prev_depth_mask is not None:
                viewer.ctx.depth_mask = prev_depth_mask
            viewer.ctx.enable(moderngl.DEPTH_TEST)

    def _panorama_stereo_layout(self):
        settings = getattr(self.viewer, '_panorama_render_settings', None)
        if callable(settings):
            try:
                return int(settings()[3])
            except Exception:
                return 0
        profile = getattr(self.viewer, '_panorama_background_settings', {}) or {}
        raw = str(profile.get('stereo_layout', 'mono')).lower()
        return 1 if raw in ('sbs', 'side_by_side', 'side-by-side', 'stereo_sbs') else 0

    def _make_equirect_layer(self, eye_visibility, x, width, height):
        return xr.CompositionLayerEquirect2KHR(
            space=self.viewer._xr_space,
            eye_visibility=eye_visibility,
            sub_image=xr.SwapchainSubImage(
                swapchain=self.viewer._background_equirect_swapchain,
                image_rect=xr.Rect2Di(
                    offset=xr.Offset2Di(x=int(x), y=0),
                    extent=xr.Extent2Di(width=int(width), height=int(height)),
                ),
                image_array_index=0,
            ),
            pose=xr.Posef(),
            radius=0.0,
            central_horizontal_angle=float(math.tau),
            upper_vertical_angle=float(math.pi * 0.5),
            lower_vertical_angle=float(-math.pi * 0.5),
        )

    def _make_equirect_layers(self):
        tex = self._panorama_texture()
        if tex is None:
            return None
        source_key = self._source_key(tex)
        if self.viewer._background_equirect_uploaded_key != source_key:
            self.viewer._background_equirect_pending_tex = tex
            return None
        width, height = self.viewer._background_equirect_size
        if self._panorama_stereo_layout() == 1:
            eye_w = int(width) // 2
            return [
                self._make_equirect_layer(xr.EyeVisibility.LEFT, 0, eye_w, height),
                self._make_equirect_layer(xr.EyeVisibility.RIGHT, eye_w, eye_w, height),
            ]
        return [self._make_equirect_layer(xr.EyeVisibility.BOTH, 0, width, height)]

    def flush_pending_upload_after_submit(self):
        tex = getattr(self.viewer, '_background_equirect_pending_tex', None)
        if tex is None:
            return False
        self.viewer._background_equirect_pending_tex = None
        self._upload_equirect_texture(tex)
        return True

    def make_background_layers(self):
        self._frame_background_layers = []
        if not self.panorama_ready():
            return [], False
        if not self.native_background_available():
            self.viewer._breakdown_inc('openxr_background_projection_fallback')
            return [], True
        make_layer = getattr(self.viewer, '_make_equirect_background_layer', None)
        if not callable(make_layer):
            make_layer = self._make_equirect_layers
        try:
            layer = make_layer()
        except Exception as exc:
            print(f"[OpenXRViewer] Background equirect layer failed: {type(exc).__name__}: {exc}")
            self.viewer._breakdown_inc('openxr_background_layer_failed')
            return [], True
        if layer is None:
            self.viewer._breakdown_inc('openxr_background_projection_fallback')
            return [], True
        layers = list(layer) if isinstance(layer, (list, tuple)) else [layer]
        if not layers:
            self.viewer._breakdown_inc('openxr_background_projection_fallback')
            return [], True
        self._frame_background_layers = layers
        self.viewer._breakdown_inc('openxr_background_layer')
        return [
            ctypes.cast(ctypes.pointer(item), ctypes.POINTER(xr.CompositionLayerBaseHeader))
            for item in layers
        ], False
