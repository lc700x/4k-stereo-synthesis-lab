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

    # EXT_memory_object_win32 helpers (cross-vendor, GL 4.5+) ----------
    _ext_mem_available       = False
    _glImportMemoryWin32HandleEXT  = None
    _glTextureStorageMem2DEXT      = None
    _glCreateMemoryObjectsEXT      = None
    _glDeleteMemoryObjectsEXT      = None

    # Handle types for glImportMemoryWin32HandleEXT
    _GL_HANDLE_TYPE_OPAQUE_WIN32_EXT        = 0x9587
    _GL_HANDLE_TYPE_D3D11_TEXTURE_KTX_Z     = None  # not used

    def _load_ext_memory_object():
        """Try to load GL_EXT_memory_object_win32 + GL_EXT_memory_object."""
        global _ext_mem_available, _glImportMemoryWin32HandleEXT
        global _glTextureStorageMem2DEXT, _glCreateMemoryObjectsEXT
        global _glDeleteMemoryObjectsEXT
        if _ext_mem_available:
            return True
        try:
            from OpenGL.GL.EXT.memory_object_win32 import glImportMemoryWin32HandleEXT
            from OpenGL.GL.EXT.memory_object import (
                glCreateMemoryObjectsEXT, glDeleteMemoryObjectsEXT,
            )
            from OpenGL.GL.EXT.memory_object_fd import glTextureStorageMem2DEXT
            _glImportMemoryWin32HandleEXT = glImportMemoryWin32HandleEXT
            _glCreateMemoryObjectsEXT     = glCreateMemoryObjectsEXT
            _glDeleteMemoryObjectsEXT     = glDeleteMemoryObjectsEXT
            _glTextureStorageMem2DEXT     = glTextureStorageMem2DEXT
            _ext_mem_available = True
            return True
        except (ImportError, AttributeError):
            # Fallback: load via raw ctypes from wglGetProcAddress
            try:
                from OpenGL.GL import wglGetProcAddress
                _names = {
                    'glImportMemoryWin32HandleEXT': ctypes.c_void_p,
                    'glTextureStorageMem2DEXT':     ctypes.c_void_p,
                    'glCreateMemoryObjectsEXT':     ctypes.c_void_p,
                    'glDeleteMemoryObjectsEXT':     ctypes.c_void_p,
                }
                _ptrs = {}
                for name in _names:
                    addr = wglGetProcAddress(name.encode() if hasattr(name, 'encode') else name)
                    if not addr:
                        raise RuntimeError(f"{name} not found")
                    _ptrs[name] = addr
                # Build ctypes function wrappers
                _glImportMemoryWin32HandleEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint, ctypes.c_void_p,
                )(ctypes.cast(_ptrs['glImportMemoryWin32HandleEXT'], ctypes.c_void_p).value)
                _glTextureStorageMem2DEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint64,
                )(ctypes.cast(_ptrs['glTextureStorageMem2DEXT'], ctypes.c_void_p).value)
                _glCreateMemoryObjectsEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
                )(ctypes.cast(_ptrs['glCreateMemoryObjectsEXT'], ctypes.c_void_p).value)
                _glDeleteMemoryObjectsEXT = ctypes.CFUNCTYPE(
                    None, ctypes.c_int, ctypes.POINTER(ctypes.c_uint),
                )(ctypes.cast(_ptrs['glDeleteMemoryObjectsEXT'], ctypes.c_void_p).value)
                _ext_mem_available = True
                return True
            except Exception:
                return False

    def _create_d3d11_shared_texture(device, w, h, fmt=_DXGI_FORMAT_R8G8B8A8_UNORM):
        """Create a D3D11 texture with D3D11_RESOURCE_MISC_SHARED_NTHANDLE.

        Returns (texture_ptr, shared_handle) as c_void_p values.
        The shared_handle is an NT kernel handle suitable for
        glImportMemoryWin32HandleEXT.
        """
        desc = (
            ctypes.c_uint(w),           # Width
            ctypes.c_uint(h),           # Height
            ctypes.c_uint(1),           # MipLevels
            ctypes.c_uint(1),           # ArraySize
            ctypes.c_uint(fmt),         # Format
            # DXGI_SAMPLE_DESC
            ctypes.c_uint(1),           # Count
            ctypes.c_uint(0),           # Quality
            ctypes.c_uint(0),           # Usage (D3D11_USAGE_DEFAULT)
            ctypes.c_uint(0x40),        # BindFlags (D3D11_BIND_SHADER_RESOURCE = 0x80 | D3D11_BIND_RENDER_TARGET = 0x20)
            ctypes.c_uint(0),           # CPUAccessFlags
            ctypes.c_uint(0x2),         # MiscFlags (D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800, but we also need SHARED = 0x2)
        )
        # Actually D3D11_RESOURCE_MISC_SHARED_NTHANDLE is the right flag for NT handles.
        # Let me redo the struct properly.
        # D3D11_TEXTURE2D_DESC layout (order matters):
        # Width:UINT, Height:UINT, MipLevels:UINT, ArraySize:UINT,
        # Format:DXGI_FORMAT, SampleDesc:DXGI_SAMPLE_DESC,
        # Usage:D3D11_USAGE, BindFlags:UINT, CPUAccessFlags:UINT, MiscFlags:UINT

        _D3D11_BIND_SHADER_RESOURCE = 0x8
        _D3D11_BIND_RENDER_TARGET   = 0x20
        _D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800

        _TEX2D_DESC_FMT = (
            'I I I I I I I I I I I'
        )  # 11 UINTs
        # We need to pack SampleDesc.Count and SampleDesc.Quality as two UINTs

        tex_desc = (
            ctypes.c_uint * 11
        )(
            w, h, 1, 1, fmt,
            1, 0,              # SampleDesc
            0,                  # D3D11_USAGE_DEFAULT
            _D3D11_BIND_SHADER_RESOURCE | _D3D11_BIND_RENDER_TARGET,
            0,                  # CPUAccessFlags
            _D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
        )

        tex_ptr = ctypes.c_void_p(0)
        vtbl = ctypes.cast(device, ctypes.POINTER(ctypes.c_void_p)).contents.value
        # ID3D11Device::CreateTexture2D at vtable index 5
        create_tex2d = ctypes.CFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,                          # this
            ctypes.POINTER(ctypes.c_uint * 11),       # pDesc
            ctypes.c_void_p,                          # pInitialData
            ctypes.POINTER(ctypes.c_void_p),          # ppTexture2D
        )(ctypes.cast(vtbl + 5 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value)

        hr = create_tex2d(device, ctypes.byref(tex_desc), None, ctypes.byref(tex_ptr))
        if hr != 0:
            raise RuntimeError(f"CreateTexture2D(shared) failed: hr=0x{hr & 0xFFFFFFFF:08x}")

        # Get shared handle via IDXGIResource1::CreateSharedHandle
        # First get IDXGIResource from ID3D11Texture2D via QueryInterface
        # IID_IDXGIResource1 = {7632e1f5-ee65-4ca2-87fd-4c20ee8d71a9}
        _IID_IDXGIResource1 = (ctypes.c_byte * 16)(
            0xf5, 0xe1, 0x32, 0x76, 0x65, 0xee, 0xa2, 0x4c,
            0x87, 0xfd, 0x4c, 0x20, 0xee, 0x8d, 0x71, 0xa9,
        )
        dxgi_res = ctypes.c_void_p(0)
        tex_vtbl = ctypes.cast(tex_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
        qi_fn = ctypes.CFUNCTYPE(
            ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(ctypes.c_byte * 16), ctypes.POINTER(ctypes.c_void_p),
        )(ctypes.cast(tex_vtbl, ctypes.POINTER(ctypes.c_void_p)).contents.value)
        hr = qi_fn(tex_ptr, ctypes.byref(_IID_IDXGIResource1), ctypes.byref(dxgi_res))
        if hr != 0 or not dxgi_res:
            tex_vtbl2 = ctypes.cast(tex_ptr, ctypes.POINTER(ctypes.c_void_p)).contents.value
            release_fn2 = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                ctypes.cast(tex_vtbl2 + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            release_fn2(tex_ptr)
            raise RuntimeError(f"QueryInterface(IDXGIResource1) failed: hr=0x{hr & 0xFFFFFFFF:08x}")

        # IDXGIResource1::CreateSharedHandle
        # Params: dwAccess, lpAttributes, dwAccessFlags, lpName, pHandle
        _DXGI_SHARED_RESOURCE_READ = 0x80000000
        _DXGI_SHARED_RESOURCE_WRITE = 1
        dxgi_vtbl = ctypes.cast(dxgi_res, ctypes.POINTER(ctypes.c_void_p)).contents.value
        create_sh = ctypes.CFUNCTYPE(
            ctypes.c_long,
            ctypes.c_void_p,      # this
            ctypes.c_uint,        # dwAccess
            ctypes.c_void_p,      # lpAttributes (NULL)
            ctypes.c_uint,        # dwAccessFlags
            ctypes.c_void_p,     # lpName (NULL)
            ctypes.POINTER(ctypes.c_void_p),  # pHandle (out)
        )(ctypes.cast(dxgi_vtbl + 12 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value)

        shared_handle = ctypes.c_void_p(0)
        hr = create_sh(
            dxgi_res,
            _DXGI_SHARED_RESOURCE_READ | _DXGI_SHARED_RESOURCE_WRITE,
            None, 0, None,
            ctypes.byref(shared_handle),
        )
        # Release DXGI resource (we only needed it for CreateSharedHandle)
        dxgi_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
            ctypes.cast(dxgi_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
        )
        dxgi_rel(dxgi_res)

        if hr != 0 or not shared_handle:
            tex_rel = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                ctypes.cast(tex_vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
            )
            tex_rel(tex_ptr)
            raise RuntimeError(f"CreateSharedHandle failed: hr=0x{hr & 0xFFFFFFFF:08x}")

        return tex_ptr, shared_handle

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
    '_load_ext_memory_object',
    '_create_d3d11_shared_texture',
    '_nv_dx_interop_available',
    '_wglDXOpenDeviceNV',
    '_wglDXCloseDeviceNV',
    '_wglDXRegisterObjectNV',
    '_wglDXLockObjectsNV',
    '_wglDXUnlockObjectsNV',
    '_wglDXUnregisterObjectNV',
    '_ext_mem_available',
    '_glImportMemoryWin32HandleEXT',
    '_glTextureStorageMem2DEXT',
    '_glCreateMemoryObjectsEXT',
    '_glDeleteMemoryObjectsEXT',
    '_GL_HANDLE_TYPE_OPAQUE_WIN32_EXT',
    '_GL_HANDLE_TYPE_D3D11_TEXTURE_KTX_Z',
]