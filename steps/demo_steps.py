"""Step-by-step visualization of every stage, saved as images for the article.

    python steps/demo_steps.py --image data/samples/synthetic/plate_7H5K829.png

Produces in outputs/steps/:
    01_input, 02_resized, 03_gray, 04_threshold, 05_contours,
    06_segmented, 07_recognized   (+ 00_pipeline montage)
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, preprocess, recognize, segment  # noqa: E402


def _save(name, img):
    path = config.STEPS_DIR / f"{name}.png"
    cv2.imwrite(str(path), img)
    return path


def _draw_contours(pre):
    """Green = kept as a character, red = rejected (with the failing rule)."""
    canvas = cv2.cvtColor(pre.binary, cv2.COLOR_GRAY2BGR)
    contours = segment.find_contours(pre.binary)
    all_boxes, _ = segment.filter_characters(contours)
    for b in all_boxes:
        color = (0, 200, 0) if b.accepted else (0, 0, 255)
        cv2.rectangle(canvas, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 1)
    return canvas, all_boxes


def _segmented_strip(pre, accepted, cell=64, pad=6):
    """A horizontal montage of the normalized character crops."""
    if not accepted:
        return np.full((cell, cell, 3), 255, np.uint8)
    tiles = []
    for b in accepted:
        crop = pre.binary[b.y:b.y + b.h, b.x:b.x + b.w]
        sq = cv2.resize(crop, (cell - 2 * pad, cell - 2 * pad), interpolation=cv2.INTER_AREA)
        tile = np.full((cell, cell), 255, np.uint8)
        tile[pad:cell - pad, pad:cell - pad] = 255 - sq
        tiles.append(cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR))
    return np.hstack(tiles)


def run(image_path, model_path=config.MODEL_PATH):
    config.ensure_dirs()
    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    print(f"=== Adhoc OCR - step by step on {image_path} ===\n")

    print("[Stage 1] Resize plate to a canonical 300x60 frame.")
    pre = preprocess.preprocess_plate(image)
    _save("01_input", image)
    _save("02_resized", pre.resized)
    _save("03_gray", pre.gray)

    print("[Stage 2] Adaptive threshold -> white characters on black.")
    _save("04_threshold", pre.binary)

    print("[Stage 3-4] Find contours, then keep only character-like boxes.")
    contour_vis, all_boxes = _draw_contours(pre)
    _save("05_contours", contour_vis)
    accepted = sorted((b for b in all_boxes if b.accepted), key=lambda b: b.x)
    print(f"           {len(all_boxes)} contours -> {len(accepted)} kept as characters.")
    for b in all_boxes:
        if not b.accepted:
            print(f"           rejected box ({b.x},{b.y},{b.w}x{b.h}) by '{b.reason}' rule")

    strip = _segmented_strip(pre, accepted)
    _save("06_segmented", strip)

    print("[Stage 5-6] Extract features and classify each character.")
    result = None
    if Path(model_path).exists():
        pipeline, charset = recognize.load_model(model_path)
        result = recognize.recognize_plate(image, pipeline, charset)
        annotated = recognize.annotate(result)
        _save("07_recognized", annotated)
        print(f"\n   >>> Recognized plate text: {result.text}\n")
    else:
        print(f"   (no model at {model_path}; run `python -m src.train` for stage 6)\n")

    _build_montage(image, pre, contour_vis, strip, result)
    print(f"Stage images written to {config.STEPS_DIR}")
    return result


def _build_montage(image, pre, contour_vis, strip, result):
    """Single labelled figure of the whole pipeline for the article."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    def rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img

    panels = [
        ("1. Input crop", rgb(image)),
        ("2. Resized 300x60", rgb(pre.resized)),
        ("3. Adaptive threshold", pre.binary),
        ("4. Contours (green=kept)", rgb(contour_vis)),
        ("5. Segmented chars", rgb(strip)),
        ("6. Recognized", rgb(recognize.annotate(result)) if result else np.zeros((10, 10))),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(11, 8))
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    sup = result.text if result else "(train a model to see predictions)"
    fig.suptitle(f"Adhoc License-Plate OCR pipeline  ->  {sup}", fontsize=13)
    fig.tight_layout()
    fig.savefig(config.STEPS_DIR / "00_pipeline.png", dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Step-by-step OCR visualization")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    args = parser.parse_args()
    run(args.image, model_path=args.model)


if __name__ == "__main__":
    main()
