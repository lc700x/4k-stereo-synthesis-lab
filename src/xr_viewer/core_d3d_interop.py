# Desktop2Stereo OpenXR viewer: D3D11 GPU interop setup and cleanup.

import ctypes
import sys

import numpy as np
from OpenGL.GL import (
    glBindFramebuffer,
    glBindTexture,
    glCheckFramebufferStatus,
    glDeleteFramebuffers,
    glDeleteTextures,
    glFinish,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    GL_COLOR_ATTACHMENT0,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_RGBA8,
    GL_TEXTURE_2D,
)

from . import d3d_interop as _d3d_interop
from .d3d_interop import (
    _GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
    _create_d3d11_shared_texture,
)


class CoreD3DInteropMixin:
    """D3D11 GPU interop setup, fallback, and cleanup."""

    @staticmethod
    def _is_nvidia_gpu():
        """Detect NVIDIA GPU via OpenGL renderer string."""
        try:
            from OpenGL.GL import glGetString, GL_RENDERER
            r = glGetString(GL_RENDERER)
            if r:
                return b'NVIDIA' in r.upper() if isinstance(r, bytes) else 'NVIDIA' in r.upper()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return 'NVIDIA' in torch.cuda.get_device_name(0)
        except Exception:
            pass
        return False

    def _setup_gpu_interop_d3d11(self):
        """Attempt GPU interop to eliminate the PBO readback path.

        Order: NV_DX_interop2 for NVIDIA GPUs, EXT_memory_object for all others.
        Falls back to the PBO path (already configured) if neither is available.

        Interop is skipped for BGRA swapchains (common on WMR) because GL
        renders RGBA natively and the R/B mismatch would swap colours.
        The PBO path handles BGRA via GL_BGRA readback format.
        """
        if not sys.platform == "win32":
            return

        if self._swapchain_is_bgra:
            print("[OpenXRViewer] BGRA swapchain - GPU interop disabled (using PBO with GL_BGRA readback)")
            return

        is_nv = self._is_nvidia_gpu()

        if is_nv and _d3d_interop._load_nv_dx_interop():
            try:
                self._init_interop_nv()
                self._interop_mode = 'nv_dx'
                print("[OpenXRViewer] GPU interop active: NV_DX_interop2 (zero-copy)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] NV_DX_interop2 setup failed: {e}")

        if _d3d_interop._load_ext_memory_object():
            try:
                self._init_interop_ext_mem()
                self._interop_mode = 'ext_mem'
                print("[OpenXRViewer] GPU interop active: EXT_memory_object (GPU-side blit)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] EXT_memory_object setup failed: {e}")

        self._interop_mode = None
        print("[OpenXRViewer] GPU interop unavailable - using PBO fallback")

    def _disable_nv_interop_after_failure(self, reason):
        print(f"[OpenXRViewer] NV_DX_interop2 disabled after swapchain registration failure: {reason}")
        self._cleanup_interop()
        if _d3d_interop._load_ext_memory_object():
            try:
                self._init_interop_ext_mem()
                self._interop_mode = 'ext_mem'
                print("[OpenXRViewer] GPU interop active: EXT_memory_object (GPU-side blit)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] EXT_memory_object setup failed after NV fallback: {e}")
        self._interop_mode = None
        print("[OpenXRViewer] D3D11 GPU interop unavailable - using PBO fallback")

    def _init_interop_nv(self):
        """Set up WGL_NV_DX_interop2: register the D3D11 device with GL.

        Individual swapchain textures are registered per-frame the first time
        each image index is seen (see _get_or_create_nv_interop_fbo).
        """
        self._nv_dx_device = _d3d_interop._wglDXOpenDeviceNV(self._d3d11_device)
        if not self._nv_dx_device:
            raise RuntimeError("wglDXOpenDeviceNV returned NULL")

    def _get_or_create_nv_interop_fbo(self, eye_index, img_index, d3d11_tex, w, h):
        """Register a swapchain D3D11 texture with GL via NV_DX_interop2.

        Each unique (eye, img_index) pair is registered once and cached.
        Returns (mgl_fbo, raw_fbo_id) for direct rendering into the D3D11 texture.
        """
        key = (eye_index, img_index)
        if key in self._nv_dx_objects:
            gl_tex, raw_fbo, _dx_obj = self._nv_dx_objects[key]
            return self.ctx.detect_framebuffer(raw_fbo), raw_fbo

        gl_tex = glGenTextures(1)
        dx_obj = None
        raw_fbo = None
        # Register the D3D11 texture as a GL texture
        register_errors = []
        for access, access_name in (
            (0x0002, "WRITE_DISCARD"),  # WGL_ACCESS_WRITE_DISCARD_NV
            (0x0001, "READ_WRITE"),     # WGL_ACCESS_READ_WRITE_NV
        ):
            try:
                dx_obj = _d3d_interop._wglDXRegisterObjectNV(
                    self._nv_dx_device,
                    d3d11_tex,
                    gl_tex,
                    GL_TEXTURE_2D,
                    access,
                )
                if dx_obj:
                    if access_name != "WRITE_DISCARD":
                        print(f"[OpenXRViewer] NV_DX_interop registered with {access_name}")
                    break
                register_errors.append(f"{access_name}: returned NULL")
            except Exception as e:
                register_errors.append(f"{access_name}: {type(e).__name__}: {e}")
        if not dx_obj:
            try:
                glDeleteTextures(1, [gl_tex])
            except Exception:
                pass
            detail = "; ".join(register_errors) if register_errors else "no detail"
            raise RuntimeError(
                f"wglDXRegisterObjectNV failed for eye {eye_index} img {img_index} "
                f"(format={self._d3d11_swapchain_fmt}, {detail})"
            )

        try:
            # Set up FBO attached to the registered texture
            raw_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
            locked = _d3d_interop._wglDXLockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
            if not locked:
                raise RuntimeError("wglDXLockObjectsNV returned false")
            try:
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
                status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
                if status != GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(
                        f"NV_DX_interop FBO incomplete for eye {eye_index} "
                        f"img {img_index}: {status:#x}"
                    )
            finally:
                _d3d_interop._wglDXUnlockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
        except Exception:
            try:
                if raw_fbo:
                    glDeleteFramebuffers(1, [raw_fbo])
            except Exception:
                pass
            try:
                _d3d_interop._wglDXUnregisterObjectNV(self._nv_dx_device, dx_obj)
            except Exception:
                pass
            try:
                glDeleteTextures(1, [gl_tex])
            except Exception:
                pass
            raise

        self._nv_dx_objects[key] = (gl_tex, raw_fbo, dx_obj)
        return self.ctx.detect_framebuffer(raw_fbo), raw_fbo

    def _init_interop_ext_mem(self):
        """Set up EXT_memory_object_win32: create shared D3D11 textures and
        import them into GL once.  Render to the GL side, then CopyResource
        to the swapchain image each frame (GPU-side blit, no CPU round-trip).
        """
        for eye_index in range(2):
            sc_w, sc_h = self._swapchain_sizes[eye_index]
            fmt = self._d3d11_swapchain_fmt
            d3d11_tex, nt_handle = _create_d3d11_shared_texture(
                self._d3d11_device, sc_w, sc_h, fmt,
            )

            # Import into GL
            mem_obj = ctypes.c_uint(0)
            _d3d_interop._glCreateMemoryObjectsEXT(1, ctypes.byref(mem_obj))
            _d3d_interop._glImportMemoryWin32HandleEXT(
                mem_obj, sc_w * sc_h * 4,
                _GL_HANDLE_TYPE_OPAQUE_WIN32_EXT,
                nt_handle,
            )

            # Create GL texture backed by the imported memory
            gl_tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, gl_tex)
            _d3d_interop._glTextureStorageMem2DEXT(gl_tex, 1, GL_RGBA8, sc_w, sc_h, mem_obj, 0)
            glBindTexture(GL_TEXTURE_2D, 0)

            # FBO
            raw_fbo = int(glGenFramebuffers(1))
            glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
            st = glCheckFramebufferStatus(GL_FRAMEBUFFER)
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            if st != GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"EXT_mem FBO incomplete for eye {eye_index}: {st:#x}")

            mgl_fbo = self.ctx.detect_framebuffer(raw_fbo)
            self._ext_shared_tex[eye_index] = (d3d11_tex, mem_obj, gl_tex, mgl_fbo, raw_fbo)

    def _blit_ext_to_swapchain(self, eye_index, d3d11_swapchain_tex):
        """GPU-side CopyResource from our shared texture to the swapchain image."""
        d3d11_shared_tex = self._ext_shared_tex[eye_index][0]
        # ID3D11DeviceContext::CopyResource at vtable index 47
        ctx = self._d3d11_context
        vtbl = ctypes.cast(ctx, ctypes.POINTER(ctypes.c_void_p)).contents.value
        copy_fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,  # this
            ctypes.c_void_p,  # pDstResource
            ctypes.c_void_p,  # pSrcResource
        )(ctypes.cast(vtbl + 47 * ctypes.sizeof(ctypes.c_void_p),
                    ctypes.POINTER(ctypes.c_void_p)).contents.value)
        # Sync: ensure GL is done before D3D11 reads the shared texture
        glFinish()
        copy_fn(ctx, d3d11_swapchain_tex, d3d11_shared_tex)

    def _cleanup_interop(self):
        """Release all GPU interop resources."""
        if self._interop_mode == 'nv_dx' and self._nv_dx_device:
            for (gl_tex, raw_fbo, dx_obj) in self._nv_dx_objects.values():
                try:
                    _d3d_interop._wglDXUnregisterObjectNV(self._nv_dx_device, dx_obj)
                except Exception:
                    pass
                try:
                    glDeleteFramebuffers(1, [raw_fbo])
                except Exception:
                    pass
                try:
                    glDeleteTextures(1, [gl_tex])
                except Exception:
                    pass
            self._nv_dx_objects.clear()
            try:
                _d3d_interop._wglDXCloseDeviceNV(self._nv_dx_device)
            except Exception:
                pass
            self._nv_dx_device = None

        if self._interop_mode == 'ext_mem':
            for d3d11_tex, mem_obj, gl_tex, mgl_fbo, raw_fbo in self._ext_shared_tex.values():
                try:
                    glDeleteFramebuffers(1, [raw_fbo])
                except Exception:
                    pass
                try:
                    glDeleteTextures(1, [gl_tex])
                except Exception:
                    pass
                try:
                    _d3d_interop._glDeleteMemoryObjectsEXT(1, ctypes.byref(ctypes.c_uint(mem_obj)))
                except Exception:
                    pass
                # Release D3D11 texture
                try:
                    tex_vtbl = ctypes.cast(d3d11_tex, ctypes.POINTER(ctypes.c_void_p)).contents.value
                    tex_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                        ctypes.cast(tex_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p),
                                    ctypes.POINTER(ctypes.c_void_p)).contents.value
                    )
                    tex_rel(d3d11_tex)
                except Exception:
                    pass
            self._ext_shared_tex.clear()

        self._interop_mode = None