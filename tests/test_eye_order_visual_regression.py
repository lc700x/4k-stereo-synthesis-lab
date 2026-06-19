import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.io import save_rgb
from stereo_runtime.output import make_sbs
from stereo_runtime.synthesis import StereoConfig, synthesize_stereo

OUT_DIR = ROOT / "outputs" / "visual_regression" / "eye_order"


_GLYPHS = {
    "L": ["10000", "10000", "10000", "10000", "11111"],
    "E": ["11111", "10000", "11110", "10000", "11111"],
    "F": ["11111", "10000", "11110", "10000", "10000"],
    "T": ["11111", "00100", "00100", "00100", "00100"],
    "R": ["11110", "10001", "11110", "10100", "10010"],
    "I": ["11111", "00100", "00100", "00100", "11111"],
    "G": ["01111", "10000", "10011", "10001", "01111"],
    "H": ["10001", "10001", "11111", "10001", "10001"],
}


def _draw_text(img: torch.Tensor, text: str, x0: int, y0: int, color: tuple[float, float, float], scale: int = 8) -> None:
    color_t = torch.tensor(color, dtype=img.dtype).view(3, 1, 1)
    x = x0
    for char in text:
        glyph = _GLYPHS[char]
        for gy, row in enumerate(glyph):
            for gx, bit in enumerate(row):
                if bit == "1":
                    ys = slice(y0 + gy * scale, y0 + (gy + 1) * scale)
                    xs = slice(x + gx * scale, x + (gx + 1) * scale)
                    img[:, ys, xs] = color_t
        x += (len(glyph[0]) + 1) * scale


def _labelled_eye(label: str, color: tuple[float, float, float], width: int = 192, height: int = 96) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, width).view(1, 1, width).expand(1, height, width)
    y = torch.linspace(0.0, 1.0, height).view(1, height, 1).expand(1, height, width)
    base = torch.cat([0.10 + x * 0.20, 0.10 + y * 0.20, torch.full_like(x, 0.12)], dim=0)
    tint = torch.tensor(color, dtype=base.dtype).view(3, 1, 1)
    img = (base + tint * 0.35).clamp(0, 1)
    _draw_text(img, label, 20, 28, (1.0, 1.0, 1.0), scale=8)
    return img.unsqueeze(0)


def test_labelled_sbs_visual_regression_keeps_left_right_order():
    left = _labelled_eye("LEFT", (0.0, 0.35, 1.0))
    right = _labelled_eye("RIGHT", (1.0, 0.18, 0.0))

    sbs = make_sbs(left, right, "full_sbs", fused=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_rgb(sbs, OUT_DIR / "labelled_left_right_full_sbs.png")

    width = left.shape[-1]
    assert torch.equal(sbs[..., :, :width], left)
    assert torch.equal(sbs[..., :, width:], right)

    crossed = make_sbs(right, left, "full_sbs", fused=False)
    save_rgb(crossed, OUT_DIR / "labelled_cross_eyed_reference_full_sbs.png")
    assert torch.equal(crossed[..., :, :width], right)
    assert torch.equal(crossed[..., :, width:], left)


def test_synthesize_full_sbs_cross_eyed_only_swaps_labelled_eyes():
    left = _labelled_eye("LEFT", (0.0, 0.35, 1.0), width=64, height=32)
    depth = torch.zeros(1, 1, 32, 64)
    config = StereoConfig(
        backend="fast",
        output_format="full_sbs",
        temporal=False,
        fused=False,
        depth_strength=0.0,
        cross_eyed=False,
    )
    normal = synthesize_stereo(left, depth, config)
    crossed = synthesize_stereo(left, depth, StereoConfig(**{**config.__dict__, "cross_eyed": True}))

    width = left.shape[-1]
    save_rgb(normal.sbs, OUT_DIR / "synthesized_normal_full_sbs.png")
    save_rgb(crossed.sbs, OUT_DIR / "synthesized_cross_eyed_full_sbs.png")

    assert torch.equal(normal.sbs[..., :, :width], normal.left_eye)
    assert torch.equal(normal.sbs[..., :, width:], normal.right_eye)
    assert torch.equal(crossed.sbs[..., :, :width], normal.right_eye)
    assert torch.equal(crossed.sbs[..., :, width:], normal.left_eye)