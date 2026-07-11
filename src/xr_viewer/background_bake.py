"""Startup-time background bake helpers for OpenXR panorama rooms."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
from PIL import Image


class BackgroundBakeService:
    def __init__(self, cache_dir: str | os.PathLike[str] | None = None):
        self.cache_dir = os.fspath(cache_dir) if cache_dir else None

    def bake_wall_light_mask(self, *, room_dir: str, panorama_path: str | None, settings: dict) -> str:
        layout = settings.get("screen_light_layout")
        if not isinstance(layout, dict):
            layout = {}
        uv = self._pair(layout.get("uv", layout.get("center", settings.get("screen_light_uv"))), (0.5, 0.58))
        radius = self._pair(layout.get("radius", layout.get("size", settings.get("screen_light_radius"))), (0.18, 0.11))
        size = self._size(settings.get("wall_light_mask_resolution", settings.get("mask_resolution")), (1024, 512))
        payload = {
            "panorama": os.path.basename(panorama_path or ""),
            "uv": uv,
            "radius": radius,
            "size": size,
        }
        digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        out_dir = self.cache_dir or os.path.join(room_dir, ".d2s_bake")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.abspath(os.path.join(out_dir, f"wall_light_mask_{digest}.png"))
        if not os.path.isfile(out_path):
            Image.fromarray(self._ellipse_mask(size, uv, radius)).save(out_path)
        try:
            return os.path.relpath(out_path, room_dir)
        except ValueError:
            return out_path

    @staticmethod
    def _pair(value, default):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return default
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _size(value, default):
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return default
        try:
            return (max(16, int(value[0])), max(8, int(value[1])))
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _ellipse_mask(size, uv, radius):
        w, h = size
        xs = (np.arange(w, dtype=np.float32) + 0.5) / float(w)
        ys = (np.arange(h, dtype=np.float32) + 0.5) / float(h)
        x, y = np.meshgrid(xs, ys)
        rx = max(float(radius[0]), 1e-4)
        ry = max(float(radius[1]), 1e-4)
        dist = ((x - float(uv[0])) / rx) ** 2 + ((y - float(uv[1])) / ry) ** 2
        mask = np.clip(1.0 - dist, 0.0, 1.0) ** 2
        return np.rint(mask * 255.0).astype(np.uint8)
