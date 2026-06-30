"""Run the whole OCR project on a single plate image - NO learning here.

    python run_pipeline.py --image data/samples/synthetic/plate_7H5K829.png

Requires a trained model (run `python -m src.train` first).
"""
import argparse
from pathlib import Path

import cv2

from src import config, recognize


def main():
    parser = argparse.ArgumentParser(description="Recognize one license plate crop")
    parser.add_argument("--image", required=True, help="path to a cropped plate image")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    parser.add_argument("--out", default=None, help="where to save the annotated result")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")
    if not Path(args.model).exists():
        raise SystemExit(f"Model not found: {args.model}. Run `python -m src.train` first.")

    pipeline, charset = recognize.load_model(args.model)
    result = recognize.recognize_plate(image, pipeline, charset)

    print(f"Recognized text : {result.text}")
    print(f"Characters found: {len(result.chars)}")
    for c in result.chars:
        print(f"  '{c.label}'  conf={c.proba:.2f}  box=({c.box.x},{c.box.y},{c.box.w},{c.box.h})")

    config.ensure_dirs()
    out = args.out or str(config.OUTPUTS_DIR / f"recognized_{Path(args.image).stem}.png")
    cv2.imwrite(out, recognize.annotate(result))
    print(f"Annotated result -> {out}")


if __name__ == "__main__":
    main()
