from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from .output import ensure_bchw, ensure_b1hw, to_uint8_image


def load_rgb(path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return data.unsqueeze(0).to(device)


def load_depth(path: str | Path, device: str | torch.device = "cpu") -> torch.Tensor:
    image = Image.open(path).convert("L")
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.view(image.height, image.width).float() / 255.0
    return data.unsqueeze(0).unsqueeze(0).to(device)


def save_rgb(tensor: torch.Tensor, path: str | Path) -> None:
    tensor = ensure_bchw(tensor, name="tensor")
    if tensor.shape[0] != 1:
        raise ValueError("save_rgb currently expects batch size 1")
    image = to_uint8_image(tensor[0]).permute(1, 2, 0).cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)


def save_depth(tensor: torch.Tensor, path: str | Path) -> None:
    tensor = ensure_b1hw(tensor)
    if tensor.shape[0] != 1:
        raise ValueError("save_depth currently expects batch size 1")
    image = to_uint8_image(tensor[0]).squeeze(0).cpu().numpy()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path)
