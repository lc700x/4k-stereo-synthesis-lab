import math
import os
from pathlib import Path

import moderngl
from .gl_state import get_depth_mask, set_depth_mask
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


_GL_RGBA8 = 0x8058


def _fit_even_size(width, height, max_w, max_h):
    width = max(16, int(width))
    height = max(16, int(height))
    max_w = max(16, int(max_w))
    max_h = max(16, int(max_h))
    scale = min(max_w / width, max_h / height, 1.0)
    return max(16, int(width * scale)) & ~1, max(16, int(height * scale)) & ~1


def _latest_vdxr_swapchain_detail():
    log_path = os.environ.get("D2S_VDXR_OPENXR_LOG")
    if not log_path:
        program_data = os.environ.get("ProgramData")
        if not program_data:
            return ""
        log_path = str(Path(program_data) / "Virtual Desktop" / "OpenXR.log")
    try:
        lines = Path(log_path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if "xrCreateSwapchain:" not in line:
            continue
        detail = []
        if "ovrResult failure" in line:
            detail.append("ovrResult=" + line.rsplit("ovrResult failure", 1)[1].strip())
        for next_line in lines[idx + 1:idx + 4]:
            stripped = next_line.strip()
            if stripped.startswith("Origin:"):
                detail.append("origin=" + stripped.split(":", 1)[1].strip())
            elif stripped.startswith("Source:"):
                detail.append("source=" + stripped.split(":", 1)[1].strip())
        return " ".join(detail)
    return ""


class CoreQuadLayerMixin:
    def _set_quad_layer_failed(self, reason):
        self._xr_quad_layer_active = False
        self._xr_quad_layer_failed = True
        self._xr_quad_layer_failure_reason = str(reason or "failed")

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

    def _release_quad_layer_swapchains(self):
        raw_fbos = []
        for mgl_fbo, raw_fbo, _width, _height in self._quad_fbo_cache.values():
            try:
                mgl_fbo.release()
            except Exception:
                pass
            raw_fbos.append(raw_fbo)
        if raw_fbos:
            try:
                glDeleteFramebuffers(len(raw_fbos), raw_fbos)
            except Exception:
                pass
        self._quad_fbo_cache.clear()

        seen = set()
        for swapchain in self._quad_swapchains.values():
            key = int(swapchain) if isinstance(swapchain, int) else id(swapchain)
            if key in seen:
                continue
            seen.add(key)
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._quad_swapchains.clear()
        self._quad_swapchain_images.clear()
        self._quad_swapchain_sizes.clear()
        self._quad_swapchain_array_size.clear()
        self._quad_swapchain_presented_eyes = set()

    def _ensure_quad_layer_swapchains_for_source(self, source_size):
        if source_size is None:
            return False
        image_type = getattr(self, '_quad_swapchain_image_type', None)
        fmt = getattr(self, '_quad_swapchain_format', None)
        formats = getattr(self, '_quad_swapchain_formats', None)
        if formats is None:
            formats = (fmt,) if fmt is not None else ()
        else:
            formats = tuple(formats)
        max_size = getattr(self, '_quad_swapchain_max_size', None)
        if image_type is None or not formats or max_size is None:
            return False
        src_w, src_h = source_size
        max_w, max_h = max_size
        quad_w, quad_h = _fit_even_size(src_w, src_h, max_w, max_h)
        scale = min(max_w / max(1, int(src_w)), max_h / max(1, int(src_h)), 1.0)
        size_note = (
            f"source={int(src_w)}x{int(src_h)} max={int(max_w)}x{int(max_h)} "
            f"aligned={quad_w}x{quad_h} scale={scale:.3f}"
        )
        if (
            self._quad_swapchain_sizes.get(0) == (quad_w, quad_h)
            and self._quad_swapchain_sizes.get(1) == (quad_w, quad_h)
            and 0 in self._quad_swapchains
            and 1 in self._quad_swapchains
        ):
            self._xr_quad_layer_active = True
            self._xr_quad_layer_failed = False
            self._xr_quad_layer_failure_reason = None
            return True

        backend = 'D3D11' if getattr(self, '_use_d3d11', False) else 'OpenGL'
        last_exc = None
        for fmt in formats:
            for attempt in range(1, 3):
                self._release_quad_layer_swapchains()
                failed_eye = None
                try:
                    for eye_index in range(2):
                        failed_eye = eye_index
                        sc_info = xr.SwapchainCreateInfo(
                            usage_flags=(
                                xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                                xr.SwapchainUsageFlags.SAMPLED_BIT
                            ),
                            format=fmt,
                            sample_count=1,
                            width=quad_w,
                            height=quad_h,
                            face_count=1,
                            array_size=1,
                            mip_count=1,
                        )
                        swapchain = xr.create_swapchain(self._xr_session, sc_info)
                        self._quad_swapchains[eye_index] = swapchain
                        self._quad_swapchain_images[eye_index] = xr.enumerate_swapchain_images(
                            swapchain, image_type
                        )
                        self._quad_swapchain_sizes[eye_index] = (quad_w, quad_h)
                        self._quad_swapchain_array_size[eye_index] = 1
                    self._quad_swapchain_format = fmt
                    self._xr_quad_layer_active = True
                    self._xr_quad_layer_failed = False
                    self._xr_quad_layer_failure_reason = None
                    retry_note = f" recovered_after_retry={attempt - 1}" if attempt > 1 else ""
                    print(
                        f"[OpenXRViewer] Quad layer {backend} swapchains: "
                        f"{quad_w}x{quad_h}/eye format={fmt} active=True {size_note}{retry_note}"
                    )
                    return True
                except Exception as exc:
                    last_exc = exc
                    runtime_detail = _latest_vdxr_swapchain_detail()
                    detail_note = f" {runtime_detail}" if runtime_detail else ""
                    print(
                        f"[OpenXRViewer] Quad layer {backend} swapchain create retry: "
                        f"format={fmt} attempt={attempt}/2 eye={failed_eye} {size_note} "
                        f"runtime_result={type(exc).__name__.removesuffix('Error')}{detail_note}"
                    )
            print(f"[OpenXRViewer] Quad layer {backend} swapchain format rejected after retries: format={fmt} {size_note}")

        self._release_quad_layer_swapchains()
        self._set_quad_layer_failed(f"swapchain_create_failed_{type(last_exc).__name__}")
        print(f"[OpenXRViewer] Quad layer unavailable for formats {formats}: {type(last_exc).__name__}: {last_exc}")
        return False

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
        if getattr(self, '_xr_quad_layer_failed', False):
            return getattr(self, '_xr_quad_layer_failure_reason', None) or "failed"
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
                prev_depth_mask = get_depth_mask()
                mgl_fbo.use()
                self.ctx.viewport = (0, 0, quad_w, quad_h)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.disable(moderngl.BLEND)
                set_depth_mask(False)
                source_tex.use(location=0)
                self._quad_copy_prog['u_flip_y'].value = 1 if flip_y else 0
                self._quad_copy_prog['u_linearize_srgb'].value = (
                    1 if getattr(self, '_quad_swapchain_format', None) == _GL_RGBA8 else 0
                )
                self._quad_copy_vao.render(moderngl.TRIANGLE_STRIP)
            xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
            released = True
            self._quad_swapchain_presented_eyes.add(int(eye_index))
            return True
        except Exception as exc:
            self._set_quad_layer_failed(f"update_failed_{type(exc).__name__}")
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
                set_depth_mask(prev_depth_mask)
            self.ctx.enable(moderngl.DEPTH_TEST)

    def _quad_layer_has_presented_frame(self):
        presented = getattr(self, '_quad_swapchain_presented_eyes', set())
        return 0 in presented and 1 in presented

    def _update_quad_layer_swapchains(self, *, force=False):
        source0, size0, _flip0 = self._quad_layer_source_texture(0)
        source1, size1, _flip1 = self._quad_layer_source_texture(1)
        if force:
            source_size = size0 or size1
            if not self._ensure_quad_layer_swapchains_for_source(source_size):
                return []
        has_presented = self._quad_layer_has_presented_frame()
        if not force and has_presented:
            if self._quad_layer_screen_presentable():
                self._breakdown_inc('openxr_quad_reused_screen_frame')
                return [0, 1]
            return []
        if not force and not has_presented:
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
                try:
                    eye_updated = self._update_quad_layer_swapchain(eye_index)
                except Exception as exc:
                    self._set_quad_layer_failed(f"update_failed_{type(exc).__name__}")
                    print(f"[OpenXRViewer] Quad layer update failed: {type(exc).__name__}: {exc}")
                    eye_updated = False
                if eye_updated:
                    updated.append(eye_index)
            if len(updated) != 2:
                if self._quad_layer_has_presented_frame():
                    breakdown_inc = getattr(self, '_breakdown_inc', None)
                    if callable(breakdown_inc):
                        breakdown_inc('openxr_quad_update_partial_reuse')
                    return [0, 1]
                if not getattr(self, '_xr_quad_layer_failed', False):
                    self._set_quad_layer_failed('partial_update_without_presented_frame')
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
            prev_depth_mask = get_depth_mask()
            for eye_index, source_tex, flip_y in ((0, source0, flip0), (1, source1, flip1)):
                mgl_fbo = self._get_or_create_quad_fbo(eye_index, img_index, sc_image.image, quad_w, quad_h)
                mgl_fbo.use()
                self.ctx.viewport = (0, 0, quad_w, quad_h)
                self.ctx.disable(moderngl.DEPTH_TEST)
                self.ctx.disable(moderngl.BLEND)
                set_depth_mask(False)
                source_tex.use(location=0)
                self._quad_copy_prog['u_flip_y'].value = 1 if flip_y else 0
                self._quad_copy_prog['u_linearize_srgb'].value = (
                    1 if getattr(self, '_quad_swapchain_format', None) == _GL_RGBA8 else 0
                )
                self._quad_copy_vao.render(moderngl.TRIANGLE_STRIP)
            xr.release_swapchain_image(swapchain, self._xr_sc_release_info)
            released = True
            self._quad_swapchain_presented_eyes.update((0, 1))
            return [0, 1]
        except Exception as exc:
            self._set_quad_layer_failed(f"stereo_update_failed_{type(exc).__name__}")
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
                set_depth_mask(prev_depth_mask)
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
        screen_basis = getattr(self, '_screen_basis', None)
        if callable(screen_basis):
            screen_height, screen_pos, _r_ax, _u_ax, normal = screen_basis()
            pos = np.array(screen_pos, dtype=np.float64)
            normal = np.array(normal, dtype=np.float64)
        else:
            screen_height = self.screen_height
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
        value = (q, pos, (float(self.screen_width), float(screen_height)))
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
