import sys
import threading, time
import os, platform
import importlib.util
from pathlib import Path

# Debug Mode
DEBUG = False
# App Version
VERSION = "2.5.0Beta"
# Get OS name
OS_NAME = platform.system()
from streaming.audio import STEREO_MIX_NAMES
from streaming.config import DEFAULT_PORT, resolve_streaming_config

from stereo_runtime.model_capabilities import (
    COMPILE_FIX_KEYWORDS,
    DISABLE_COREML_KEYWORDS,
    DISABLE_CUDNN_KEYWORDS,
    DISABLE_OPENVINO_KEYWORDS,
    DISABLE_TRITON_KEYWORDS,
    DISABLE_TRT_KEYWORDS,
    FORCE_FP32_KEYWORDS,
    TRT_FIX_KEYWORDS,
    model_name_mapping,
)
from stereo_runtime.depth_settings import resolve_depth_settings

# Global shutdown event
shutdown_event = threading.Event()

from viewer.assets import crop_icon, get_font_type


from .display import (
    _get_device_name_from_mss_monitor,
    compute_output_resolution,
    get_fps,
    get_monitor_size,
)
from .network import configure_huggingface_endpoint, get_local_ip
from .run_mode import resolve_run_mode
from .settings import load_settings, read_yaml

# load customized settings
settings = load_settings("settings.yaml")

# Ignore wanning for MPS
if OS_NAME == "Darwin":
    import os, warnings
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    warnings.filterwarnings(
        "ignore",
        message=".*aten::upsample_bicubic2d.out.*MPS backend.*",
        category=UserWarning)
from viewer.window_control import (
    hide_window_from_capture,
    send_ctrl_cmd_f,
    set_window_to_bottom,
    show_window_in_capture,
)
from viewer.upscaler import normalize_upscaler, normalize_upscaler_sharpness
        
# Set Hugging Face environment variable
configure_huggingface_endpoint()

# Model Mapping Dict. Keep the Desktop2Stereo settings shape, but make
# stereo_runtime.model_registry the single source of truth for model names.
MODEL_MAPPING = model_name_mapping()

# Streamer Settings
_STREAMING_CONFIG = resolve_streaming_config(settings)
STREAM_QUALITY = _STREAMING_CONFIG.stream_quality
STREAM_PORT = _STREAMING_CONFIG.stream_port
LOCAL_IP = _STREAMING_CONFIG.local_ip

# Get settings
_RUN_MODE_CONFIG = resolve_run_mode(
    settings["Run Mode"],
    os_name=OS_NAME,
    fix_viewer_aspect=settings["Fix Viewer Aspect"],
    lossless_scaling_support=settings["Lossless Scaling Support"],
)
RUN_MODE = _RUN_MODE_CONFIG.run_mode
STREAM_MODE = _RUN_MODE_CONFIG.stream_mode
USE_3D_MONITOR = _RUN_MODE_CONFIG.use_3d_monitor
LOSSLESS_SCALING_SUPPORT = _RUN_MODE_CONFIG.lossless_scaling_support
_DEPTH_SETTINGS = resolve_depth_settings(settings)
MODEL = _DEPTH_SETTINGS.model
MODEL_ID = _DEPTH_SETTINGS.model_id
ALL_MODELS = _DEPTH_SETTINGS.all_models
CACHE_PATH = _DEPTH_SETTINGS.cache_path
DEPTH_RESOLUTION = _DEPTH_SETTINGS.depth_resolution
DEVICE_ID = _DEPTH_SETTINGS.device_id
FP16 = _DEPTH_SETTINGS.fp16
MONITOR_INDEX,  DISPLAY_MODE = settings["Monitor Index"], settings["Display Mode"]
STEREO_DISPLAY_INDEX = settings.get("Stereo Output")
STEREO_DISPLAY_SELECTION = False if not STEREO_DISPLAY_INDEX else True
OUTPUT_RESOLUTION = compute_output_resolution(
    settings.get("Processing Resolution", "Auto"),
    DISPLAY_MODE,
    MONITOR_INDEX,
    STEREO_DISPLAY_INDEX,
)
SHOW_FPS, DEPTH_STRENGTH = settings["Show FPS"], settings["Depth Strength"]
IPD = settings["IPD"]
CONVERGENCE = settings["Convergence"]
CAPTURE_MODE = settings["Capture Mode"]
WINDOW_TITLE = settings["Window Title"] if CAPTURE_MODE == "Window" else None
TARGET_FPS = int(settings.get("Target FPS", 0) or 0)
FPS = TARGET_FPS if 1 <= TARGET_FPS <= 240 else get_fps(WINDOW_TITLE, MONITOR_INDEX)

FOREGROUND_SCALE = _DEPTH_SETTINGS.foreground_scale
AA_STRENGTH = _DEPTH_SETTINGS.aa_strength
USE_TORCH_COMPILE = _DEPTH_SETTINGS.use_torch_compile
USE_TENSORRT = _DEPTH_SETTINGS.use_tensorrt
RECOMPILE_TRT = _DEPTH_SETTINGS.recompile_trt
USE_COREML = _DEPTH_SETTINGS.use_coreml
RECOMPILE_COREML = _DEPTH_SETTINGS.recompile_coreml
USE_OPENVINO = _DEPTH_SETTINGS.use_openvino
RECOMPILE_OPENVINO = _DEPTH_SETTINGS.recompile_openvino

def _load_capture_select():
    # Load the selector without importing capture.__init__, which still depends
    # on utils during the package migration.
    path = Path(__file__).resolve().parents[1] / "capture" / "capture_select.py"
    spec = importlib.util.spec_from_file_location("_d2s_capture_select", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_resolve_capture_tool = _load_capture_select().resolve_capture_tool

CAPTURE_TOOL = _resolve_capture_tool(settings["Capture Tool"])
FILL_16_9 = settings["Fill 16:9"]
LOCAL_VSYNC = settings.get("Local VSync", True)
UPSCALER = normalize_upscaler(settings.get("Upscaler", "Off"))
UPSCALER_SHARPNESS = normalize_upscaler_sharpness(settings.get("Upscaler Sharpness", 0.35))
FIX_VIEWER_ASPECT = _RUN_MODE_CONFIG.fix_viewer_aspect
STEREOMIX_DEVICE = _STREAMING_CONFIG.stereo_mix_device
STREAM_KEY = _STREAMING_CONFIG.stream_key
AUDIO_DELAY = _STREAMING_CONFIG.audio_delay
CRF = _STREAMING_CONFIG.crf
LANG = settings["Language"]

from viewer.controller_help import get_controller_help_rows

ROWS, ENV_ROWS = get_controller_help_rows(LANG)

# Specify the Stereo Display for output
CONTROLLER_MODEL = settings["Controller Model"]
ENVIRONMENT_MODEL = settings.get("Environment Model", "Default")
XR_PREVIEW_WINDOW = settings.get("XR Preview Window", True)

# Initialize Device
from .device import get_device
    
DEVICE, DEVICE_INFO = get_device(DEVICE_ID)
