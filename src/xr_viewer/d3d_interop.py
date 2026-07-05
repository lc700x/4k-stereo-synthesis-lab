# Desktop2Stereo OpenXR viewer: D3D11 and GL/D3D interop helpers.

import ctypes
import sys

from OpenGL.GL import glGetError, GL_NO_ERROR

# D3D11 ctypes helpers (Windows only)
# DXGI / D3D11 format constants used for swapchain negotiation
_DXGI_FORMAT_R8G8B8A8_UNORM_SRGB = 29
_DXGI_FORMAT_R8G8B8A8_UNORM      = 28
_DXGI_FORMAT_B8G8R8A8_UNORM_SRGB = 91
_DXGI_FORMAT_B8G8R8A8_UNORM      = 87

_D3D11_PREFERRED_FORMATS = [
    _DXGI_FORMAT_R8G8B8A8_UNORM,
    _DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
    _DXGI_FORMAT_B8G8R8A8_UNORM,
    _DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
]

if sys.platform == "win32":
    import ctypes.wintypes as _wintypes

    _d3d11 = None
    _dxgi  = None

    def _load_d3d11():
        global _d3d11, _dxgi
        if _d3d11 is None:
            _d3d11 = ctypes.windll.LoadLibrary("d3d11.dll")
            _dxgi  = ctypes.windll.LoadLibrary("dxgi.dll")

    # D3D_DRIVER_TYPE / D3D_FEATURE_LEVEL constants
    _D3D_DRIVER_TYPE_HARDWARE  = 1
    _D3D_DRIVER_TYPE_WARP      = 5
    _D3D11_SDK_VERSION         = 7
    _D3D_FEATURE_LEVEL_11_0    = 0xb000
    _D3D_FEATURE_LEVEL_10_1    = 0xa100
    _D3D_FEATURE_LEVEL_10_0    = 0xa000

    def _create_d3d11_device(adapter_luid=None):
        """Create an ID3D11Device + ID3D11DeviceContext via ctypes.
        Returns (device_ptr, context_ptr, feature_level) as c_void_p values.
        adapter_luid: optional _LUID from GraphicsRequirementsD3D11KHR to pick the correct adapter.
        """
        _load_d3d11()

        feature_levels = (ctypes.c_int * 3)(
            _D3D_FEATURE_LEVEL_11_0,
            _D3D_FEATURE_LEVEL_10_1,
            _D3D_FEATURE_LEVEL_10_0,
        )
        device      = ctypes.c_void_p(0)
        context     = ctypes.c_void_p(0)
        feat_out    = ctypes.c_int(0)

        # If adapter_luid provided, try to find the matching IDXGIAdapter
        adapter = ctypes.c_void_p(0)
        if adapter_luid is not None:
            try:
                adapter = _find_dxgi_adapter(adapter_luid)
            except Exception as e:
                print(f"[OpenXRViewer] LUID adapter lookup failed ({e}), using default")
                adapter = ctypes.c_void_p(0)

        hr = _d3d11.D3D11CreateDevice(
            adapter,                              # pAdapter (NULL = default)
            _D3D_DRIVER_TYPE_HARDWARE if not adapter else 0,  # DriverType (0=unknown when adapter set)
            None,                                 # Software
            0,                                    # Flags
            feature_levels,
            3,                                    # FeatureLevels count
            _D3D11_SDK_VERSION,
            ctypes.byref(device),
            ctypes.byref(feat_out),
            ctypes.byref(context),
        )
        if hr != 0:
            raise RuntimeError(f"D3D11CreateDevice failed: hr=0x{hr & 0xFFFFFFFF:08x}")
        return device, context, feat_out.value

    # IID_IDXGIFactory1  {770aae78-f26f-4dba-a829-253c83d1b387}
    _IID_IDXGIFactory1 = (ctypes.c_byte * 16)(
        0x78, 0xae, 0x0a, 0x77, 0x6f, 0xf2, 0xba, 0x4d,
        0xa8, 0x29, 0x25, 0x3c, 0x83, 0xd1, 0xb3, 0x87,
    )

    def _find_dxgi_adapter(luid):
        """Return an IDXGIAdapter* (c_void_p) matching the given _LUID, or raise."""
        _load_d3d11()
        factory = ctypes.c_void_p(0)
        hr = _dxgi.CreateDXGIFactory1(ctypes.byref((ctypes.c_byte * 16)(*_IID_IDXGIFactory1)),
                                    ctypes.byref(factory))
        if hr != 0:
            raise RuntimeError(f"CreateDXGIFactory1 failed: 0x{hr & 0xFFFFFFFF:08x}")

        # IDXGIFactory1 vtable: [0]=QI [1]=AddRef [2]=Release ... [7]=EnumAdapters1
        # We use EnumAdapters (index 6) which is on IDXGIFactory (parent interface)
        # Layout: IUnknown(0-2) + IDXGIObject(3-6) + IDXGIFactory(7=EnumAdapters, 8=MakeWindowAssoc, 9=GetWindowAssoc, 10=CreateSwapChain, 11=CreateSoftwareAdapter) + IDXGIFactory1(12=EnumAdapters1, 13=IsCurrent)
        vtbl = ctypes.cast(factory, ctypes.POINTER(ctypes.c_void_p)).contents.value
        enum_adapters = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))(
            ctypes.cast(vtbl + 7 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
        )
        release_fn = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
            ctypes.cast(vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
        )

        idx = 0
        result_adapter = ctypes.c_void_p(0)
        while True:
            adapter = ctypes.c_void_p(0)
            hr = enum_adapters(factory, idx, ctypes.byref(adapter))
            if hr != 0:
                break
            # IDXGIAdapter vtable: QI(0) AddRef(1) Release(2) SetPrivateData(3) SetPrivateDataInterface(4) GetPrivateData(5) GetParent(6) EnumOutputs(7) GetDesc(8)
            # DXGI_ADAPTER_DESC is: Description[128 wchar], VendorId, DeviceId, SubSysId, Revision, DedicatedVideoMemory, DedicatedSystemMemory, SharedSystemMemory, AdapterLuid
            adapter_vtbl = ctypes.cast(adapter, ctypes.POINTER(ctypes.c_void_p)).contents.value
            # GetDesc is vtbl[8]
            get_desc = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
                ctypes.cast(adapter_vtbl + 8 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            # DXGI_ADAPTER_DESC: 128*2 bytes description + 4*4 IDs + 3*8 memory + 8 luid = 128*2+16+24+8 = 304 bytes
            desc_buf = (ctypes.c_byte * 304)()
            get_desc(adapter, desc_buf)
            # LUID is at offset 128*2 + 4*4 + 3*8 = 256+16+24 = 296 bytes
            luid_low  = ctypes.c_ulong.from_buffer_copy(desc_buf, 296).value
            luid_high = ctypes.c_long.from_buffer_copy(desc_buf, 300).value
            adapter_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                ctypes.cast(adapter_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            if luid_low == luid.low_part and luid_high == luid.high_part:
                result_adapter = adapter
                break
            adapter_rel(adapter)
            idx += 1

        release_fn(factory)
        if not result_adapter:
            raise RuntimeError("Matching DXGI adapter not found for LUID")
        return result_adapter

    def _d3d11_update_subresource(context, dst, src_ptr, row_pitch):
        """Write CPU data into a D3D11 texture via UpdateSubresource (vtbl index 48).
        Works with any format including SRGB -no staging texture needed.
        src_ptr: integer address of the source data (already row-reversed).
        """
        _UPDATE_SR_VTBL_IDX = 48
        vtbl = ctypes.cast(context, ctypes.POINTER(ctypes.c_void_p)).contents.value
        fn_ptr = ctypes.cast(
            vtbl + _UPDATE_SR_VTBL_IDX * ctypes.sizeof(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ).contents.value
        UpdateFn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_void_p,  # this
            ctypes.c_void_p,  # pDstResource
            ctypes.c_uint,    # DstSubresource
            ctypes.c_void_p,  # pDstBox (NULL = whole texture)
            ctypes.c_void_p,  # pSrcData
            ctypes.c_uint,    # SrcRowPitch
            ctypes.c_uint,    # SrcDepthPitch
        )(fn_ptr)
        UpdateFn(
            context.value,
            ctypes.cast(dst, ctypes.c_void_p).value,
            0,        # subresource 0
            None,     # full texture
            src_ptr,
            row_pitch,
            0,
        )



    # NV_DX_interop2 helpers (NVIDIA only, zero-copy GL->D3D11)
    _nv_dx_interop_available = False
    _wglDXOpenDeviceNV        = None
    _wglDXCloseDeviceNV       = None
    _wglDXRegisterObjectNV    = None
    _wglDXLockObjectsNV       = None
    _wglDXUnlockObjectsNV     = None
    _wglDXUnregisterObjectNV  = None

    def _load_nv_dx_interop():
        """Try to load WGL_NV_DX_interop2 extension functions."""
        global _nv_dx_interop_available, _wglDXOpenDeviceNV, _wglDXCloseDeviceNV
        global _wglDXRegisterObjectNV, _wglDXLockObjectsNV, _wglDXUnlockObjectsNV
        global _wglDXUnregisterObjectNV
        if _nv_dx_interop_available:
            return True
        try:
            from OpenGL.WGL.NV.DX_interop2 import (
                wglDXOpenDeviceNV, wglDXCloseDeviceNV,
                wglDXRegisterObjectNV, wglDXLockObjectsNV,
                wglDXUnlockObjectsNV, wglDXUnregisterObjectNV,
            )
            _wglDXOpenDeviceNV       = wglDXOpenDeviceNV
            _wglDXCloseDeviceNV      = wglDXCloseDeviceNV
            _wglDXRegisterObjectNV   = wglDXRegisterObjectNV
            _wglDXLockObjectsNV      = wglDXLockObjectsNV
            _wglDXUnlockObjectsNV    = wglDXUnlockObjectsNV
            _wglDXUnregisterObjectNV = wglDXUnregisterObjectNV
            _nv_dx_interop_available = True
            return True
        except ImportError:
            try:
                # Fallback: load via wglGetProcAddress
                from OpenGL.WGL.NV.DX_interop import (
                    wglDXOpenDeviceNV, wglDXCloseDeviceNV,
                    wglDXRegisterObjectNV, wglDXLockObjectsNV,
                    wglDXUnlockObjectsNV, wglDXUnregisterObjectNV,
                )
                _wglDXOpenDeviceNV       = wglDXOpenDeviceNV
                _wglDXCloseDeviceNV      = wglDXCloseDeviceNV
                _wglDXRegisterObjectNV   = wglDXRegisterObjectNV
                _wglDXLockObjectsNV      = wglDXLockObjectsNV
                _wglDXUnlockObjectsNV    = wglDXUnlockObjectsNV
                _wglDXUnregisterObjectNV = wglDXUnregisterObjectNV
                _nv_dx_interop_available = True
                return True
            except (ImportError, AttributeError):
                return False
        except (ImportError, AttributeError):
            return False

else:
    def _create_d3d11_device(adapter_luid=None):
        raise RuntimeError("D3D11 only available on Windows")

__all__ = [name for name in globals() if name.startswith('_DXGI_') or name.startswith('_D3D')]
__all__ += [
    '_D3D11_PREFERRED_FORMATS',
    '_create_d3d11_device',
    '_find_dxgi_adapter',
    '_d3d11_update_subresource',
    '_load_nv_dx_interop',
    '_nv_dx_interop_available',
    '_wglDXOpenDeviceNV',
    '_wglDXCloseDeviceNV',
    '_wglDXRegisterObjectNV',
    '_wglDXLockObjectsNV',
    '_wglDXUnlockObjectsNV',
    '_wglDXUnregisterObjectNV',
]
