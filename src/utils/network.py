import os
import socket
import threading

import requests


def get_local_ip():
    """Return the local IP address by creating a UDP socket to a public IP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            # The remote address does not need to be reachable for getsockname().
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def is_cn_ip():
    """Dual-probe detection: only returns False when *both* endpoints are reachable.
    Exception → True (safe side, triggers HF mirror fallback)."""
    google_ok = False
    hf_ok = False

    try:
        requests.get("https://www.google.com", timeout=5)
        google_ok = True
    except Exception:
        google_ok = False

    try:
        requests.get("https://huggingface.co", timeout=5)
        hf_ok = True
    except Exception:
        hf_ok = False

    return not (google_ok and hf_ok)


_HF_ENDPOINT_DEFAULT = "https://hf-mirror.com"
_HF_ENDPOINT_PROBE_LOCK = threading.Lock()
_HF_ENDPOINT_PROBE_STARTED = False


def _set_huggingface_endpoint_from_probe():
    endpoint = "https://hf-mirror.com" if is_cn_ip() else "https://huggingface.co"
    os.environ["HF_ENDPOINT"] = endpoint


def _start_huggingface_endpoint_probe_once():
    global _HF_ENDPOINT_PROBE_STARTED
    with _HF_ENDPOINT_PROBE_LOCK:
        if _HF_ENDPOINT_PROBE_STARTED:
            return
        _HF_ENDPOINT_PROBE_STARTED = True
    thread = threading.Thread(
        target=_set_huggingface_endpoint_from_probe,
        name="HFEndpointProbe",
        daemon=True,
    )
    thread.start()


def configure_huggingface_endpoint(async_probe=True):
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
    if not async_probe:
        _set_huggingface_endpoint_from_probe()
        return os.environ["HF_ENDPOINT"]
    endpoint = os.environ.setdefault("HF_ENDPOINT", _HF_ENDPOINT_DEFAULT)
    _start_huggingface_endpoint_probe_once()
    return endpoint
