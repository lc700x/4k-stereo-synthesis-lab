import ctypes
import math
import time

import moderngl
from .gl_state import get_depth_mask, set_depth_mask
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

    def native_background_available(self, *, panorama_ready=None):
        if panorama_ready is None:
            panorama_ready = self.panorama_ready()
        if xr is None or not hasattr(xr, 'CompositionLayerEquirect2KHR') or not panorama_ready:
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
            set_depth_mask(False)
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
                set_depth_mask(prev_depth_mask)
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
            if getattr(self.viewer, '_background_equirect_failed_key', None) == source_key:
                self.viewer._breakdown_inc('openxr_background_layer_upload_suppressed')
                return None
            self.viewer._background_equirect_pending_tex = tex
            return None
        width, height = self.viewer._background_equirect_size
        self._record_safe_background_reuse(source_key)
        if self._panorama_stereo_layout() == 1:
            eye_w = int(width) // 2
            return [
                self._make_equirect_layer(xr.EyeVisibility.LEFT, 0, eye_w, height),
                self._make_equirect_layer(xr.EyeVisibility.RIGHT, eye_w, eye_w, height),
            ]
        return [self._make_equirect_layer(xr.EyeVisibility.BOTH, 0, width, height)]

    def _record_safe_background_reuse(self, source_key):
        viewer = self.viewer
        frame_id = int(getattr(viewer, '_frame_count', 0) or 0)
        if getattr(viewer, '_background_equirect_last_reuse_frame', None) == frame_id:
            return
        uploaded_frame = getattr(viewer, '_background_equirect_uploaded_frame', None)
        if uploaded_frame is not None:
            viewer._breakdown_add_value(
                'openxr_background_safe_age_frames',
                max(0.0, float(frame_id - int(uploaded_frame))),
            )
        if getattr(viewer, '_background_equirect_last_layer_key', None) == source_key:
            viewer._breakdown_inc('openxr_background_reuse')
        viewer._background_equirect_last_layer_key = source_key
        viewer._background_equirect_last_reuse_frame = frame_id

    def flush_pending_upload_after_submit(self):
        tex = getattr(self.viewer, '_background_equirect_pending_tex', None)
        if tex is None:
            return False
        self.viewer._background_equirect_pending_tex = None
        if bool(getattr(self.viewer, '_openxr_background_upload_budget_skip_armed', False)):
            self.viewer._openxr_background_upload_budget_skip_armed = False
            self.viewer._breakdown_inc('openxr_background_upload_budget_skip')
            return False
        start = time.perf_counter()
        source_key = self._source_key(tex)
        try:
            self._upload_equirect_texture(tex)
            self.viewer._background_equirect_uploaded_frame = int(getattr(self.viewer, '_frame_count', 0) or 0)
            self.viewer._background_equirect_failed_key = None
        except Exception as exc:
            print(f"[OpenXRViewer] Background equirect upload failed: {type(exc).__name__}: {exc}")
            self.viewer._background_equirect_failed_key = source_key
            self.viewer._breakdown_inc('openxr_background_layer_upload_failed')
            return True
        finally:
            elapsed = time.perf_counter() - start
            self.viewer._breakdown_add_time('openxr_background_upload', elapsed)
            budget_ms = float(getattr(self.viewer, '_openxr_background_upload_budget_ms', 0.0) or 0.0)
            if budget_ms > 0.0:
                self.viewer._openxr_background_upload_budget_skip_armed = (elapsed * 1000.0) > budget_ms
        return True

    def make_background_layers(self):
        self._frame_background_layers = []
        panorama_ready = self.panorama_ready()
        if not panorama_ready:
            return [], False
        if not self.native_background_available(panorama_ready=panorama_ready):
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
