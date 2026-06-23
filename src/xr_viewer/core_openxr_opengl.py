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
            from OpenGL.WGL import wglGetCurrentContext, wglGetCurrentDC
            binding = xr.GraphicsBindingOpenGLWin32KHR(
                h_dc=wglGetCurrentDC(),
                h_glrc=wglGetCurrentContext(),
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
        for eye_index, vcv in enumerate(view_configs):
            rec_w = vcv.recommended_image_rect_width
            rec_h = vcv.recommended_image_rect_height
            scale = float(getattr(self, '_xr_render_scale', 1.0))
            max_w = int(getattr(vcv, 'max_image_rect_width', rec_w) or rec_w)
            max_h = int(getattr(vcv, 'max_image_rect_height', rec_h) or rec_h)
            sc_w = min(max_w, max(16, int(rec_w * scale))) & ~1
            sc_h = min(max_h, max(16, int(rec_h * scale))) & ~1
            print(f"[OpenXRViewer] Eye {eye_index} swapchain: {sc_w}x{sc_h} (scale={scale:.2f})")

            sc_info = xr.SwapchainCreateInfo(
                usage_flags=(
                    xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                    xr.SwapchainUsageFlags.SAMPLED_BIT
                ),
                format=_GL_SRGB8_ALPHA8,
                sample_count=1,
                width=sc_w,
                height=sc_h,
                face_count=1,
                array_size=1,
                mip_count=1,
            )
            swapchain = xr.create_swapchain(self._xr_session, sc_info)
            images = xr.enumerate_swapchain_images(swapchain, xr.SwapchainImageOpenGLKHR)
            self._xr_swapchains[eye_index] = swapchain
            self._swapchain_images[eye_index] = images
            self._swapchain_sizes[eye_index] = (sc_w, sc_h)

        if self._xr_quad_layer_enabled and view_configs:
            try:
                src_w, src_h = self.frame_size
                max_w = max(int(getattr(v, 'max_image_rect_width', src_w) or src_w) for v in view_configs)
                max_h = max(int(getattr(v, 'max_image_rect_height', src_h) or src_h) for v in view_configs)
                quad_w = min(max_w, max(16, int(src_w))) & ~1
                quad_h = min(max_h, max(16, int(src_h))) & ~1
                for eye_index in range(2):
                    sc_info = xr.SwapchainCreateInfo(
                        usage_flags=(
                            xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT |
                            xr.SwapchainUsageFlags.SAMPLED_BIT
                        ),
                        format=_GL_SRGB8_ALPHA8,
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
                        swapchain, xr.SwapchainImageOpenGLKHR
                    )
                    self._quad_swapchain_sizes[eye_index] = (quad_w, quad_h)
                    self._quad_swapchain_array_size[eye_index] = 1
                print(f"[OpenXRViewer] Quad per-eye swapchains: {quad_w}x{quad_h}x2 active=True")
                self._xr_quad_layer_active = True
                print(f"[OpenXRViewer] Quad layer swapchains: {quad_w}x{quad_h}/eye active={self._xr_quad_layer_active}")
            except Exception as exc:
                self._xr_quad_layer_active = False
                self._xr_quad_layer_failed = True
                print(f"[OpenXRViewer] Quad layer unavailable: {type(exc).__name__}: {exc}")

        # 8. Controller actions (optional -silently disabled if action set creation fails)
        try:
            if self._action_set is None:
                self._init_controller_actions()
            else:
                self._attach_controller_actions_to_session()
        except Exception as e:
            print(f"[OpenXRViewer] Controller actions unavailable: {e}")
