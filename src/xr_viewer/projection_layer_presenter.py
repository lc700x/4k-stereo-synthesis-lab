import ctypes

import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_DRAW_FRAMEBUFFER,
    GL_LINEAR,
    GL_READ_FRAMEBUFFER,
    glBindFramebuffer,
    glBlitFramebuffer,
    glReadBuffer,
)

from . import d3d_interop as _d3d_interop
from .xr_math import _fov_to_proj_mat4, _fov_to_proj_mat4_d3d, _pose_to_view_mat4

try:
    import xr
except ImportError:
    xr = None


class ProjectionLayerPresenter:
    def __init__(self, viewer):
        self.viewer = viewer

    def render_d3d11_native(self, views, default_fov, default_proj_d3d):
        viewer = self.viewer
        eye_layer_views = []
        for eye_index in range(2):
            swapchain = viewer._xr_swapchains[eye_index]
            img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
            viewer._wait_swapchain_image(swapchain)
            released = False
            try:
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                proj_mat = _fov_to_proj_mat4_d3d(view.fov) if view else default_proj_d3d
                mvp = proj_mat @ view_mat @ viewer._build_model_mat4()
                if viewer._runtime_direct_source:
                    viewer._d3d11_native_renderer.render_runtime_eye(
                        sc_image.texture,
                        sc_w,
                        sc_h,
                        eye_index,
                        mvp,
                    )
                else:
                    runtime_rgb_depth_max_disparity_px = float(
                        getattr(viewer, '_runtime_rgb_depth_max_disparity_px', 0.0)
                    )
                    runtime_rgb_depth_render_width = int(
                        getattr(viewer, '_runtime_rgb_depth_render_width', 0) or 0
                    )
                    if runtime_rgb_depth_render_width <= 0:
                        source_size = viewer._texture_size or (0, 0)
                        runtime_rgb_depth_render_width = int(source_size[0] or 0)
                    screen_disparity_uv = 0.0
                    if runtime_rgb_depth_render_width > 0:
                        screen_disparity_uv = max(0.0, runtime_rgb_depth_max_disparity_px) / float(runtime_rgb_depth_render_width)
                    screen_depth_strength = max(
                        0.0,
                        float(getattr(viewer, '_runtime_rgb_depth_depth_strength', viewer.depth_strength) or 0.0),
                    )
                    eye_sign = -1.0 if eye_index == 0 else 1.0
                    viewer._d3d11_native_renderer.render_eye(
                        sc_image.texture,
                        sc_w,
                        sc_h,
                        eye_index,
                        eye_sign * screen_disparity_uv / 2.0,
                        screen_depth_strength,
                        float(viewer.convergence),
                        mvp,
                        roll=viewer.screen_roll,
                    )
                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released = True
                eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
            except Exception as exc:
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                print(f"[OpenXRViewer] D3D11 native render failed: {exc}")
                try:
                    viewer._d3d11_native_renderer.cleanup()
                except Exception:
                    pass
                viewer._d3d11_native_renderer = None
                viewer._texture_size = None
                return []
        return eye_layer_views

    def render_nv_dx_interop(self, views, default_fov, default_proj):
        viewer = self.viewer
        eye_layer_views = []
        for eye_index in range(2):
            swapchain = viewer._xr_swapchains[eye_index]
            img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
            viewer._wait_swapchain_image(swapchain)
            released = False
            try:
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                proj_mat = _fov_to_proj_mat4(view.fov) if view else default_proj

                mgl_fbo, _raw_fbo = viewer._get_or_create_nv_interop_fbo(
                    eye_index, img_index, sc_image.texture, sc_w, sc_h,
                )
                _, _, dx_obj = viewer._nv_dx_objects[(eye_index, img_index)]
                _d3d_interop._wglDXLockObjectsNV(viewer._nv_dx_device, 1, ctypes.byref(dx_obj))
                try:
                    viewer._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)
                finally:
                    _d3d_interop._wglDXUnlockObjectsNV(viewer._nv_dx_device, 1, ctypes.byref(dx_obj))

                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released = True
                eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
            except Exception as exc:
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                viewer._disable_nv_interop_after_failure(exc)
                return []
        return eye_layer_views

    def render_d3d11_pbo(self, views, default_fov, default_proj):
        viewer = self.viewer
        pending = []
        eye_layer_views = []

        for eye_index in range(2):
            swapchain = viewer._xr_swapchains[eye_index]
            img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
            viewer._wait_swapchain_image(swapchain)
            released = False
            try:
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                proj_mat = _fov_to_proj_mat4(view.fov) if view else default_proj

                mgl_fbo, raw_fbo_id = viewer._get_or_create_offscreen_fbo(eye_index, img_index, sc_w, sc_h)
                viewer._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)

                pbo_id = viewer._get_or_create_d3d11_pbo(eye_index, img_index, sc_w, sc_h)
                viewer._submit_pbo_readback(raw_fbo_id, pbo_id, sc_w, sc_h)
                pending.append((pbo_id, sc_image.texture, sc_w, sc_h, swapchain, view))
                released = True
            except Exception as exc:
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                viewer._breakdown_inc('openxr_projection_render_failed')
                print(f"[OpenXRViewer] D3D11 PBO projection render failed: {type(exc).__name__}: {exc}")
                for _pbo_id, _tex, _w, _h, pending_swapchain, _view in pending:
                    try:
                        xr.release_swapchain_image(pending_swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                return []

        for pbo_id, d3d11_tex, sc_w, sc_h, swapchain, view in pending:
            released = False
            try:
                viewer._upload_pbo_to_d3d11(pbo_id, d3d11_tex, sc_w, sc_h)
                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released = True
            except Exception as exc:
                viewer._breakdown_inc('openxr_projection_render_failed')
                print(f"[OpenXRViewer] D3D11 PBO projection upload failed: {type(exc).__name__}: {exc}")
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                return []
            eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
        return eye_layer_views

    def render_opengl(self, views, default_fov, default_proj, *, updated_quad_eyes=()):
        viewer = self.viewer
        eye_layer_views = []
        for eye_index in range(2):
            swapchain = viewer._xr_swapchains[eye_index]
            img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
            viewer._wait_swapchain_image(swapchain)
            released = False
            try:
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                proj_mat = _fov_to_proj_mat4(view.fov) if view else default_proj

                raw_fbo, mgl_fbo = viewer._get_or_create_fbo(
                    eye_index, img_index, sc_image.image, sc_w, sc_h
                )
                viewer._render_eye(eye_index, mgl_fbo, view_mat, proj_mat)

                if viewer._preview_active and eye_index == 0 and not updated_quad_eyes:
                    self._mirror_preview(raw_fbo, sc_w, sc_h)

                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released = True
            except Exception as exc:
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                viewer._breakdown_inc('openxr_projection_render_failed')
                print(f"[OpenXRViewer] OpenGL projection render failed: {type(exc).__name__}: {exc}")
                return []

            eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
        return eye_layer_views

    def _projection_view(self, swapchain, sc_w, sc_h, view, default_fov):
        return xr.CompositionLayerProjectionView(
            pose=view.pose if view else xr.Posef(),
            fov=view.fov if view else default_fov,
            sub_image=xr.SwapchainSubImage(
                swapchain=swapchain,
                image_rect=xr.Rect2Di(
                    offset=xr.Offset2Di(x=0, y=0),
                    extent=xr.Extent2Di(width=sc_w, height=sc_h),
                ),
            ),
        )

    def _mirror_preview(self, raw_fbo, sc_w, sc_h):
        viewer = self.viewer
        pw, ph = glfw.get_window_size(viewer.window)
        if pw <= 0 or ph <= 0:
            return
        glBindFramebuffer(GL_READ_FRAMEBUFFER, raw_fbo)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, sc_w, sc_h, 0, 0, pw, ph, GL_COLOR_BUFFER_BIT, GL_LINEAR)
        glfw.swap_buffers(viewer.window)
