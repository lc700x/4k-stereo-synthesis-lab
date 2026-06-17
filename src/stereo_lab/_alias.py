from __future__ import annotations

import importlib
import sys


def alias_module(name: str) -> None:
    target = importlib.import_module(f"stereo_runtime.{name}")
    sys.modules[f"stereo_lab.{name}"] = target
