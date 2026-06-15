from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from .output import ensure_bchw


def absdiff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = ensure_bchw(a, name="a").float()
    b = ensure_bchw(b, name="b").float()
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"channel mismatch: {a.shape[1]} vs {b.shape[1]}")
    return (a - b).abs().clamp(0, 1)


def basic_image_metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = absdiff(a, b)
    mse = float((diff * diff).mean().item())
    mae = float(diff.mean().item())
    psnr = 99.0 if mse <= 1e-12 else float(10.0 * torch.log10(torch.tensor(1.0 / mse)).item())
    return {"mae": mae, "mse": mse, "psnr": psnr}


def make_contact_sheet(images: list[torch.Tensor], columns: int = 2, pad: int = 8) -> torch.Tensor:
    if not images:
        raise ValueError("images must not be empty")
    tensors = [ensure_bchw(x, name="image").float() for x in images]
    if any(x.shape[0] != 1 for x in tensors):
        raise ValueError("contact sheet currently expects batch size 1")

    max_h = max(x.shape[-2] for x in tensors)
    max_w = max(x.shape[-1] for x in tensors)
    rows = (len(tensors) + columns - 1) // columns
    canvas = torch.zeros(1, tensors[0].shape[1], rows * max_h + (rows - 1) * pad, columns * max_w + (columns - 1) * pad)
    for idx, image in enumerate(tensors):
        row = idx // columns
        col = idx % columns
        y = row * (max_h + pad)
        x = col * (max_w + pad)
        h, w = image.shape[-2:]
        canvas[:, :, y : y + h, x : x + w] = image.cpu()
    return canvas


def write_json(data: dict, path: str | Path) -> None:
    import json

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
