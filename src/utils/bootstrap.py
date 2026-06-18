from __future__ import annotations

from .network import configure_huggingface_endpoint
from .platform_env import configure_platform_environment
from .settings import load_settings


def bootstrap_settings(path: str, *, os_name: str) -> dict:
    settings = load_settings(path)
    configure_platform_environment(os_name)
    configure_huggingface_endpoint()
    return settings
