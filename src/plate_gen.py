"""Generate clean synthetic plate crops - guaranteed-working pipeline inputs."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config

_FONT = Path(r"C:\Windows\Fonts\arialbd.ttf")
_DEFAULT_PLATES = ["7H5K829", "ABC1234", "BX09KLM", "TS18E9", "9920FU"]


def make_plate(text, bg=(245, 245, 245), fg=(20, 20, 20),
               width=config.PLATE_WIDTH, height=config.PLATE_HEIGHT):
    img = Image.new("RGB", (width, height), color=bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(str(_FONT), 40)
    spacing = 6
    widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in text]
    total = sum(widths) + spacing * (len(text) - 1)
    x = (width - total) // 2
    for ch, w in zip(text, widths):
        l, t, r, b = font.getbbox(ch)
        y = (height - (b - t)) // 2 - t
        draw.text((x - l, y), ch, fill=fg, font=font)
        x += w + spacing
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (1, 1), (width - 2, height - 2), (40, 40, 40), 2)
    return bgr


def write_default_plates(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for text in _DEFAULT_PLATES:
        path = out_dir / f"plate_{text}.png"
        cv2.imwrite(str(path), make_plate(text))
        paths.append(path)
    return paths
