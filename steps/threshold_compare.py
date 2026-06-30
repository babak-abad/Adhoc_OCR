"""Why adaptive thresholding? Compare global (Otsu) vs adaptive on a real plate.

    python steps/threshold_compare.py --image data/samples/real_crops/<file>.png

A single global threshold uses one cut-off for the whole plate, so any lighting
gradient or shadow pushes part of the plate to the wrong side. Adaptive
thresholding picks a local cut-off per neighbourhood and survives it.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, preprocess  # noqa: E402


def add_shadow(bgr, strength=0.75):
    """Simulate uneven lighting: a brightness ramp across the plate (a shadow)."""
    h, w = bgr.shape[:2]
    ramp = np.linspace(1.0, 1.0 - strength, w, dtype=np.float32)
    ramp = np.tile(ramp, (h, 1))[..., None]
    return np.clip(bgr.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def compare(image, shadow=False):
    resized = preprocess.resize_plate(image)
    if shadow:
        resized = add_shadow(resized)
    gray = preprocess.to_gray(resized)
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    _, otsu = cv2.threshold(gray_blur, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, fixed = cv2.threshold(gray_blur, 127, 255, cv2.THRESH_BINARY_INV)
    adaptive = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, config.ADAPTIVE_BLOCK_SIZE, config.ADAPTIVE_C)
    return resized, fixed, otsu, adaptive


def build_figure(image, path=None, shadow=False):
    resized, fixed, otsu, adaptive = compare(image, shadow=shadow)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    panels = [
        ("Plate crop", cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)),
        ("Global threshold (fixed 127)", fixed),
        ("Global threshold (Otsu)", otsu),
        ("Adaptive threshold", adaptive),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 4.2))
    for ax, (title, img) in zip(np.ravel(axes), panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Global vs adaptive thresholding on a real plate", fontsize=13)
    fig.tight_layout()
    path = path or (config.STEPS_DIR / "thresh_compare.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Global vs adaptive threshold demo")
    parser.add_argument("--image", required=True)
    parser.add_argument("--shadow", action="store_true",
                        help="simulate uneven lighting before thresholding")
    args = parser.parse_args()
    config.ensure_dirs()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    out = build_figure(image, shadow=args.shadow)
    print(f"Saved comparison -> {out}")


if __name__ == "__main__":
    main()
