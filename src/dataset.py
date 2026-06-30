"""Synthetic training data: render 0-9/A-Z from system fonts with augmentation.

No external dataset is required - labels come for free from the rendered glyph.
Each sample is processed through the *same* binarize/crop path used at
inference time so the feature distributions match.
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config, features

_FONT_DIR = Path(r"C:\Windows\Fonts")
_FONT_CANDIDATES = [
    "arial.ttf", "arialbd.ttf", "ARIALN.TTF", "ARIALNB.TTF",
    "calibri.ttf", "calibrib.ttf",
    "Candara.ttf", "Candarab.ttf",
    "consola.ttf", "consolab.ttf",
    "corbel.ttf", "corbelb.ttf",
    "cour.ttf", "courbd.ttf",
    "ebrima.ttf", "ebrimabd.ttf",
    "framd.ttf", "FRAMDCN.TTF",
    "georgia.ttf", "georgiab.ttf",
    "lucon.ttf",
    "segoeui.ttf", "segoeuib.ttf",
    "tahoma.ttf", "tahomabd.ttf",
    "times.ttf", "timesbd.ttf",
    "trebuc.ttf", "trebucbd.ttf",
    "verdana.ttf", "verdanab.ttf",
]
VARIANTS_PER_FONT = 8
_CANVAS = 128


def available_fonts():
    return [str(_FONT_DIR / f) for f in _FONT_CANDIDATES if (_FONT_DIR / f).exists()]


# Plate-like palettes: dark text on a light background so the (grayscale)
# binarisation stays consistent, while giving the HUE features realistic
# variance that matches real plates (white, yellow, light-blue, light-green...).
_BG_COLORS = [(255, 255, 255), (250, 240, 150), (210, 230, 250),
              (220, 250, 220), (245, 245, 245), (240, 225, 180)]
_FG_COLORS = [(0, 0, 0), (20, 20, 20), (10, 20, 80), (80, 10, 10), (40, 40, 40)]


def _render_glyph(char, font_path, font_size, bg, fg):
    """Render one dark glyph centred on a light, coloured canvas (returns BGR)."""
    img = Image.new("RGB", (_CANVAS, _CANVAS), color=bg)
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    l, t, r, b = font.getbbox(char)
    x = (_CANVAS - (r - l)) // 2 - l
    y = (_CANVAS - (b - t)) // 2 - t
    draw.text((x, y), char, fill=fg, font=font)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _augment(bgr, rng):
    """Rotation, shear, low-resolution simulation, blur, noise.

    The low-res + blur steps are what let the classifier survive *real* plates,
    which arrive small and soft after the detector crop is upscaled to 300x60.
    """
    angle = rng.uniform(-12, 12)
    shear = rng.uniform(-0.18, 0.18)
    m = cv2.getRotationMatrix2D((_CANVAS / 2, _CANVAS / 2), angle, 1.0)
    m[0, 1] += shear  # add horizontal shear
    bgr = cv2.warpAffine(bgr, m, (_CANVAS, _CANVAS),
                         borderValue=(255, 255, 255), flags=cv2.INTER_LINEAR)

    # Simulate a low-resolution capture: shrink then enlarge back.
    if rng.random() < 0.6:
        scale = rng.uniform(0.18, 0.5)
        small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        bgr = cv2.resize(small, (_CANVAS, _CANVAS), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
    if rng.random() < 0.5:
        noise = rng.normal(0, 14, bgr.shape).astype(np.int16)
        bgr = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return bgr


def _to_char_crop(bgr):
    """Binarize (char=white) and crop to the glyph bounding box."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    x, y, w, h = cv2.boundingRect(np.vstack(cnts))
    return binary[y:y + h, x:x + w], bgr[y:y + h, x:x + w]


def generate(samples_per_font=VARIANTS_PER_FONT, seed=42):
    """Build the (X, y) feature matrix / label vector for every class."""
    rng = np.random.default_rng(seed)
    fonts = available_fonts()
    if not fonts:
        raise RuntimeError("No usable TrueType fonts found under C:\\Windows\\Fonts")

    rows, labels = [], []
    for label_idx, char in enumerate(config.CHARSET):
        for font_path in fonts:
            for _ in range(samples_per_font):
                size = int(rng.uniform(80, 104))
                bg = _BG_COLORS[rng.integers(len(_BG_COLORS))]
                fg = _FG_COLORS[rng.integers(len(_FG_COLORS))]
                glyph = _render_glyph(char, font_path, size, bg, fg)
                glyph = _augment(glyph, rng)
                # Stroke-width jitter on the rendered glyph.
                if rng.random() < 0.4:
                    k = np.ones((2, 2), np.uint8)
                    op = cv2.erode if rng.random() < 0.5 else cv2.dilate
                    glyph = op(glyph, k, iterations=1)
                char_bin, char_color = _to_char_crop(glyph)
                if char_bin is None:
                    continue
                rows.append(features.extract_features(char_bin, char_color))
                labels.append(label_idx)

    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64), fonts


def load_or_generate(samples_per_font=VARIANTS_PER_FONT, seed=42, use_cache=True):
    config.ensure_dirs()
    cache = config.CACHE_DIR / f"chars_{samples_per_font}_{seed}.npz"
    if use_cache and cache.exists():
        data = np.load(cache)
        return data["X"], data["y"]
    X, y, _ = generate(samples_per_font=samples_per_font, seed=seed)
    np.savez_compressed(cache, X=X, y=y)
    return X, y
