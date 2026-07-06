# Desktop2Stereo OpenXR viewer: OpenGL-backed OpenXR session creation.

import ctypes
import sys

import numpy as np

try:
    import xr
except ImportError:
    xr = None

from .implementation_support import _openxr_app_api_version

_GL_SRGB8_ALPHA8 = 0x8C43
_GL_RGBA8 = 0x8058
_OPENGL_PREFERRED_FORMATS = (_GL_SRGB8_ALPHA8, _GL_RGBA8)


class CoreOpenXROpenGLMixin:
    """OpenGL-backed OpenXR session and swapchain creation."""

    def _init_openxr_opengl(self, quiet=False):
        """Create or resume an OpenGL-backed OpenXR runtime/session."""
        if self._xr_backend not in (None, 'opengl'):
            raise RuntimeError(f"OpenXR backend mismatch: {self._xr_backend}")
        if self._xr_instance is None:
            app_info = xr.ApplicationInfo(
                application_name="Desktop2Stereo",
                application_version=1,
                engine_name="D2S",
                engine_version=1,
                api_version=_openxr_app_api_version(),
            )
            create_info = xr.InstanceCreateInfo(
                application_info=app_info,
                enabled_extension_names=[xr.KHR_OPENGL_ENABLE_EXTENSION_NAME],
            )
            self._xr_instance = xr.create_instance(create_info)
            self._xr_backend = 'opengl'
            self._use_d3d11 = False
            if not quiet:
                print("[OpenXRViewer] XrInstance created (OpenGL)")

        if self._xr_system_id is None:
            self._xr_system_id = xr.get_system(
                self._xr_instance,
                xr.SystemGetInfo(form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY),
            )

        if self._xr_session is not None:
            return

        # 3. Verify GL requirements (mandatory before session creation)
        _pfn = ctypes.cast(
            xr.get_instance_proc_addr(self._xr_instance, "xrGetOpenGLGraphicsRequirementsKHR"),
            xr.PFN_xrGetOpenGLGraphicsRequirementsKHR,
        )
        _reqs = xr.GraphicsRequirementsOpenGLKHR()
        xr.check_result(xr.Result(_pfn(self._xr_instance, self._xr_system_id, ctypes.byref(_reqs))))

        # 4. Graphics binding -platform-specific
        if sys.platform == "win32":
            import glfw
            from OpenGL.WGL import wglGetCurrentContext, wglGetCurrentDC
            if getattr(self, 'window', None) is not None:
                glfw.make_context_current(self.window)
            h_dc = wglGetCurrentDC()
            h_glrc = wglGetCurrentContext()
            if not h_dc or not h_glrc:
                raise RuntimeError(f"OpenGL context is not current: h_dc={h_dc} h_glrc={h_glrc}")
            binding = xr.GraphicsBindingOpenGLWin32KHR(
                h_dc=h_dc,
                h_glrc=h_glrc,
            )
        else:
            from OpenGL.GLX import glXGetCurrentContext, glXGetCurrentDisplay, glXGetCurrentDrawable
            binding = xr.GraphicsBindingOpenGLXlibKHR(
                x_display=glXGetCurrentDisplay(),
                glx_drawable=glXGetCurrentDrawable(),
                glx_context=glXGetCurrentContext(),
            )

        # 5. Session
        session_info = xr.SessionCreateInfo(
            system_id=self._xr_system_id,
            next=binding,
        )
        self._xr_session = xr.create_session(self._xr_instance, session_info)
        if not quiet:
            print("[OpenXRViewer] XrSession created (OpenGL)")

        # 6. Reference space -prefer STAGE (floor origin), fall back to LOCAL
        available_spaces = xr.enumerate_reference_spaces(self._xr_session)
        ref_type = (
            xr.ReferenceSpaceType.STAGE
            if xr.ReferenceSpaceType.STAGE in available_spaces
            else xr.ReferenceSpaceType.LOCAL
        )
        self._xr_ref_space_type = ref_type
        self._xr_space = xr.create_reference_space(
            self._xr_session,
            xr.ReferenceSpaceCreateInfo(
                reference_space_type=ref_type,
                pose_in_reference_space=xr.Posef(),
            ),
        )
        self._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)

        # 7. Swapchains -one per eye
        view_configs = xr.enumerate_view_configuration_views(
            self._xr_instance,
            self._xr_system_id,
            xr.ViewConfigurationType.PRIMARY_STEREO,
        )
        runtime_fmts = list(xr.enumerate_swapchain_formats(self._xr_session))
        print(f"[OpenXRViewer] OpenGL runtime swapchain formats: {runtime_fmts}")
        chosen_fmt = None
        for preferred in _OPENGL_PREFERRED_FORMATS:
            if preferred in runtime_fmts:
                chosen_fmt = preferred
                break
        if chosen_fmt is None:
            raise RuntimeError(f"No supported OpenGL swapchain format. Runtime offers: {runtime_fmts}")
        if chosen_fmt != _GL_SRGB8_ALPHA8:
            print(f"[OpenXRViewer] OpenGL swapchain format selected from runtime fallback: {chosen_fmt}")
        else:
            print(f"[OpenXRViewer] OpenGL swapchain format selected from runtime: {chosen_fmt}")
        for eye_index, vcv in enumerate(view_configs):
            rec_w = vcv.recommended_image_rect_width
            rec_h = vcv.recommended_image_rect_height
            scale = float(getattr(self, '_xr_render_scale', 1.0))
            max_w = int(getattr(vcv, 'max_image_rect_width', rec_w) or rec_w)
            max_h = int(getattr(vcv, 'max_image_rect_height', rec_h) or rec_h)
            sc_w = min(max_w, max(16, int(rec_w * scale))) & ~1
            sc_h = min(max_h, max(16, int(rec_h * scale))) & ~1
            print(f"[OpenXRViewer] Eye {eye_index} swapchain: {sc_w}x{sc_h} (scale={scale:.2f})")

            selected_format = getattr(self, '_xr_opengl_swapchain_format', None)
            candidate_formats = (selected_format,) if selected_format is not None else (chosen_fmt,)
            last_exc = None
            for fmt in candidate_formats:
                sc_info = xr.SwapchainCreateInfo(
                    usage_flags=(
                        xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                        xr.SwapchainUsageFlags.SAMPLED_BIT
                    ),
                    format=fmt,
                    sample_count=1,
                    width=sc_w,
                    height=sc_h,
                    face_count=1,
                    array_size=1,
                    mip_count=1,
                )
                swapchain = None
                try:
                    swapchain = xr.create_swapchain(self._xr_session, sc_info)
                    images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
                    break
                except Exception as exc:
                    last_exc = exc
                    if swapchain is not None:
                        try:
                            xr.destroy_swapchain(swapchain)
                        except Exception:
                            pass
            else:
                raise RuntimeError(
                    f"OpenGL swapchain create failed for formats {candidate_formats}"
                ) from last_exc
            self._xr_swapchains[eye_index] = swapchain
            self._swapchain_images[eye_index] = images
            self._swapchain_sizes[eye_index] = (sc_w, sc_h)
            self._xr_opengl_swapchain_format = sc_info.format
            if eye_index == 0:
                print(f"[OpenXRViewer] OpenGL swapchain format selected: {sc_info.format}")

        if view_configs:
            src_w, src_h = self.frame_size
            max_w = max(int(getattr(v, 'max_image_rect_width', src_w) or src_w) for v in view_configs)
            max_h = max(int(getattr(v, 'max_image_rect_height', src_h) or src_h) for v in view_configs)
            self._quad_swapchain_format = getattr(self, '_xr_opengl_swapchain_format', _GL_SRGB8_ALPHA8)
            self._quad_swapchain_image_type = xr.SwapchainImageOpenGLKHR
            self._quad_swapchain_max_size = (max_w, max_h)
            print(f"[OpenXRViewer] Quad layer OpenGL lazy swapchains armed max={max_w}x{max_h}")

        # 8. Controller actions (optional -silently disabled if action set creation fails)
        try:
            if self._action_set is None:
                self._init_controller_actions()
            else:
                self._attach_controller_actions_to_session()
        except Exception as e:
            print(f"[OpenXRViewer] Controller actions unavailable: {e}")
