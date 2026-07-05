import math

import moderngl
import numpy as np

try:
    import xr
except ImportError:
    xr = None

from OpenGL.GL import (
    glBindFramebuffer,
    glCheckFramebufferStatus,
    glDeleteFramebuffers,
    glFramebufferTexture2D,
    glFramebufferTextureLayer,
    glGenFramebuffers,
    glGetError,
    GL_COLOR_ATTACHMENT0,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_NO_ERROR,
    GL_TEXTURE_2D,
)


class CoreQuadLayerMixin:
    def _get_or_create_quad_fbo(self, eye_index, image_index, gl_tex, width, height):
        array_size = int(self._quad_swapchain_array_size.get(eye_index, 1))
        layer_index = int(eye_index) if array_size > 1 else 0
        key = (int(eye_index), int(image_index), layer_index, array_size)
        cached = self._quad_fbo_cache.get(key)
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
        while glGetError() != GL_NO_ERROR:
            pass
        raw_fbo = int(glGenFramebuffers(1))
        err = glGetError()
        if err != GL_NO_ERROR:
            raise RuntimeError(f"Quad layer glGenFramebuffers failed: {err:#x}")
        glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
        err = glGetError()
        if err != GL_NO_ERROR:
            raise RuntimeError(f"Quad layer glBindFramebuffer failed: {err:#x}")
        if array_size > 1:
            glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, gl_tex, 0, layer_index)
        else:
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
        err = glGetError()
        if err != GL_NO_ERROR:
            raise RuntimeError(
                f"Quad layer framebuffer attach failed: {err:#x} "
                f"array_size={array_size} layer={layer_index} tex={int(gl_tex)}"
            )
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if status != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError(f"Quad layer FBO incomplete: {status:#x}")
        mgl_fbo = self.ctx.detect_framebuffer(raw_fbo)
        self._quad_fbo_cache[key] = (mgl_fbo, raw_fbo, width, height)
        return mgl_fbo

    def _quad_layer_source_texture(self, eye_index=0):
        if not self._runtime_direct_source:
            return None, None, True
        if not getattr(self, '_runtime_eye_has_frame', False) and not getattr(self, '_use_d3d11', False):
            return None, None, True
        if getattr(self, '_use_d3d11', False):
            renderer = getattr(self, '_d3d11_native_renderer', None)
            if renderer is not None and renderer.has_frame and renderer.runtime_eye_size is not None:
                return renderer, renderer.runtime_eye_size, False
        textures = getattr(self, '_runtime_eye_textures', [None, None])
        source_tex = textures[eye_index] if eye_index < len(textures) else None
        if source_tex is None:
            for candidate in textures:
                if candidate is not None:
                    source_tex = candidate
                    break
        return source_tex, self._runtime_eye_texture_size, True

    def _quad_layer_unavailable_reason(self):
        if not getattr(self, '_xr_quad_layer_enabled', False):
            return "disabled"
        if getattr(self, '_xr_quad_layer_failed', False):
            return "failed"
        if not getattr(self, '_xr_quad_layer_active', False):
            return "inactive"
        if getattr(self, '_screen_curved', False):
            return "curved_screen"
        if not getattr(self, '_runtime_direct_source', False):
            return "not_runtime_direct"
        if 0 not in self._quad_swapchains or 1 not in self._quad_swapchains:
            return "missing_swapchain"
        source0, size0, _flip0 = self._quad_layer_source_texture(0)
        source1, size1, _flip1 = self._quad_layer_source_texture(1)
        if source0 is None or source1 is None or size0 is None or size1 is None:
            return "missing_source_texture"
        return None

    def _quad_layer_screen_presentable(self):
        reason = self._quad_layer_unavailable_reason()
        return reason is None or (reason == 'missing_source_texture' and self._quad_layer_has_presented_frame())

    def _update_quad_layer_swapchain(self, eye_index):
        if not (self._xr_quad_layer_active and eye_index in self._quad_swapchains):
            return False
        self._quad_swapchain_presented_eyes = getattr(self, '_quad_swapchain_presented_eyes', set())
        source_tex, source_size, flip_y = self._quad_layer_source_texture(eye_index)
        if source_tex is None or source_size is None:
            return False
        quad_w, quad_h = self._quad_swapchain_sizes[eye_index]
        swapchain = self._quad_swapchains[eye_index]
        img_index = xr.acquire_swapchain_image(swapchain, self._xr_sc_acquire_info)
        self._wait_swapchain_image(swapchain)
        released = False
        prev_viewport = None
        prev_depth_mask = None
        try:
            sc_image = self._quad_swapchain_images[eye_index][img_index]
            if getattr(self, '_use_d3d11', False):
                source_tex.render_runtime_eye(sc_image.texture, quad_w, quad_h, eye_index, np.eye(4, dtype=np.float32))
            else:
                mgl_fbo = self._get_or_create_quad_fbo(eye_index, img_index, sc_image.image, quad_w, quad_h)
                prev_viewport = self.ctx.viewport
                prev_depth_mask = self.ctx.depth_mask
                mgl_fbo.use()
                self.ctx.viewport = (0, 0, quad_w, quad_h)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.disable(moderngl.BLEND)
                self.ctx.depth_mask = False
                source_tex.use(location=0)
                self._quad_copy_prog['u_flip_y'].value = 1 if flip_y else 0
                self._quad_copy_vao.render(moderngl.TRIANGLE_STRIP)
            xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
            released = True
            self._quad_swapchain_presented_eyes.add(int(eye_index))
            return True
        except Exception as exc:
            self._xr_quad_layer_active = False
            self._xr_quad_layer_failed = True
            print(f"[OpenXRViewer] Quad layer update failed: {type(exc).__name__}: {exc}")
            return False
        finally:
            if not released:
                try:
                    xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
                except Exception:
                    pass
            if prev_viewport is not None:
                self.ctx.viewport = prev_viewport
            if prev_depth_mask is not None:
                self.ctx.depth_mask = prev_depth_mask
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _quad_layer_has_presented_frame(self):
        presented = getattr(self, '_quad_swapchain_presented_eyes', set())
        return 0 in presented and 1 in presented

    def _update_quad_layer_swapchains(self, *, force=False):
        if not force and self._quad_layer_has_presented_frame():
            if self._quad_layer_screen_presentable():
                self._breakdown_inc('openxr_quad_reused_screen_frame')
                return [0, 1]
            return []
        if not self._quad_layer_screen_presentable():
            return []
        self._quad_swapchain_presented_eyes = getattr(self, '_quad_swapchain_presented_eyes', set())
        shared_swapchain = (
            self._quad_swapchains.get(0) is not None
            and self._quad_swapchains.get(0) is self._quad_swapchains.get(1)
            and int(self._quad_swapchain_array_size.get(0, 1)) > 1
        )
        if not shared_swapchain:
            updated = []
            for eye_index in range(2):
                if self._update_quad_layer_swapchain(eye_index):
                    updated.append(eye_index)
            if len(updated) != 2:
                if self._quad_layer_has_presented_frame():
                    breakdown_inc = getattr(self, '_breakdown_inc', None)
                    if callable(breakdown_inc):
                        breakdown_inc('openxr_quad_update_partial_reuse')
                    return [0, 1]
                self._xr_quad_layer_active = False
                self._xr_quad_layer_failed = True
                breakdown_inc = getattr(self, '_breakdown_inc', None)
                if callable(breakdown_inc):
                    breakdown_inc('openxr_quad_layer_failed')
                return []
            return updated

        source0, size0, flip0 = self._quad_layer_source_texture(0)
        source1, size1, flip1 = self._quad_layer_source_texture(1)
        if source0 is None or source1 is None or size0 is None or size1 is None:
            if self._quad_layer_has_presented_frame():
                breakdown_inc = getattr(self, '_breakdown_inc', None)
                if callable(breakdown_inc):
                    breakdown_inc('openxr_quad_missing_source_reuse')
                return [0, 1]
            return []
        quad_w, quad_h = self._quad_swapchain_sizes[0]
        swapchain = self._quad_swapchains[0]
        img_index = xr.acquire_swapchain_image(swapchain, self._xr_sc_acquire_info)
        self._wait_swapchain_image(swapchain)
        released = False
        prev_viewport = None
        prev_depth_mask = None
        try:
            sc_image = self._quad_swapchain_images[0][img_index]
            prev_viewport = self.ctx.viewport
            prev_depth_mask = self.ctx.depth_mask
            for eye_index, source_tex, flip_y in ((0, source0, flip0), (1, source1, flip1)):
                mgl_fbo = self._get_or_create_quad_fbo(eye_index, img_index, sc_image.image, quad_w, quad_h)
                mgl_fbo.use()
                self.ctx.viewport = (0, 0, quad_w, quad_h)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.disable(moderngl.BLEND)
                self.ctx.depth_mask = False
                source_tex.use(location=0)
                self._quad_copy_prog['u_flip_y'].value = 1 if flip_y else 0
                self._quad_copy_vao.render(moderngl.TRIANGLE_STRIP)
            xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
            released = True
            self._quad_swapchain_presented_eyes.update((0, 1))
            return [0, 1]
        except Exception as exc:
            self._xr_quad_layer_active = False
            self._xr_quad_layer_failed = True
            print(f"[OpenXRViewer] Quad stereo layer update failed: {type(exc).__name__}: {exc}")
            return []
        finally:
            if not released:
                try:
                    xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
                except Exception:
                    pass
            if prev_viewport is not None:
                self.ctx.viewport = prev_viewport
            if prev_depth_mask is not None:
                self.ctx.depth_mask = prev_depth_mask
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _screen_pose_quat_xyzw(self):
        cy = math.cos(self.screen_yaw * 0.5)
        sy = math.sin(self.screen_yaw * 0.5)
        cp = math.cos(self.screen_pitch * 0.5)
        sp = math.sin(self.screen_pitch * 0.5)
        cr = math.cos(self.screen_roll * 0.5)
        sr = math.sin(self.screen_roll * 0.5)
        # Quaternion order matches R_yaw @ R_pitch @ R_roll used by _build_model_mat4.
        qy = np.array([0.0, sy, 0.0, cy], dtype=np.float64)
        qp = np.array([sp, 0.0, 0.0, cp], dtype=np.float64)
        qr = np.array([0.0, 0.0, sr, cr], dtype=np.float64)

        def mul(a, b):
            ax, ay, az, aw = a
            bx, by, bz, bw = b
            return np.array([
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ], dtype=np.float64)

        q = mul(mul(qy, qp), qr)
        q /= np.linalg.norm(q) + 1e-12
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])

    def _quad_layer_pose_state(self):
        self._ensure_screen_dimensions()
        offset = float(getattr(self, '_xr_quad_layer_debug_offset', 0.0))
        key = (
            int(getattr(self, '_frame_count', 0) or 0),
            float(self.screen_yaw),
            float(self.screen_pitch),
            float(self.screen_roll),
            float(self.screen_pan_x),
            float(self.screen_pan_y),
            float(self.screen_distance),
            float(self.screen_width),
            float(self.screen_height),
            offset,
        )
        if getattr(self, '_quad_layer_pose_state_key', None) == key:
            return self._quad_layer_pose_state_value
        q = self._screen_pose_quat_xyzw()
        cp = math.cos(self.screen_pitch)
        sp = math.sin(self.screen_pitch)
        sy = math.sin(self.screen_yaw)
        cy = math.cos(self.screen_yaw)
        normal = np.array([cp * sy, -sp, cp * cy], dtype=np.float64)
        pos = np.array([
            float(self.screen_pan_x),
            float(self.screen_pan_y),
            float(-self.screen_distance),
        ], dtype=np.float64)
        if offset != 0.0:
            pos = pos - normal * offset
            if not self._xr_quad_layer_debug_logged:
                print(f"[OpenXRViewer] Quad layer debug offset active: {offset:.3f}m toward viewer")
                self._xr_quad_layer_debug_logged = True
        value = (q, pos, (float(self.screen_width), float(self.screen_height)))
        self._quad_layer_pose_state_key = key
        self._quad_layer_pose_state_value = value
        return value

    def _make_quad_layer(self, eye_index):
        if not (self._xr_quad_layer_active and eye_index in self._quad_swapchains):
            return None
        (qx, qy, qz, qw), pos, (screen_width, screen_height) = self._quad_layer_pose_state()
        return xr.CompositionLayerQuad(
            space=self._xr_space,
            eye_visibility=xr.EyeVisibility.LEFT if eye_index == 0 else xr.EyeVisibility.RIGHT,
            sub_image=xr.SwapchainSubImage(
                swapchain=self._quad_swapchains[eye_index],
                image_rect=xr.Rect2Di(
                    offset=xr.Offset2Di(x=0, y=0),
                    extent=xr.Extent2Di(
                        width=int(self._quad_swapchain_sizes[eye_index][0]),
                        height=int(self._quad_swapchain_sizes[eye_index][1]),
                    ),
                ),
                image_array_index=int(eye_index) if int(self._quad_swapchain_array_size.get(eye_index, 1)) > 1 else 0,
            ),
            pose=xr.Posef(
                orientation=xr.Quaternionf(x=qx, y=qy, z=qz, w=qw),
                position=xr.Vector3f(
                    x=float(pos[0]),
                    y=float(pos[1]),
                    z=float(pos[2]),
                ),
            ),
            size=xr.Extent2Df(
                width=screen_width,
                height=screen_height,
            ),
        )
