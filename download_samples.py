"""Fetch a few real sample images + generate clean synthetic plate crops.

Real images come from the OpenALPR Benchmark dataset (public, no auth):
    https://github.com/openalpr/benchmarks  (endtoend/eu, endtoend/us)

These are *full car* photos - in a real system a detector locates the plate
first; here they illustrate the article. The synthetic crops are ready-to-run
inputs for the recognition pipeline.

    python download_samples.py
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from src import config
from src.plate_gen import write_default_plates

load_dotenv()

BASE_URL = os.getenv(
    "SAMPLE_BASE_URL",
    "https://raw.githubusercontent.com/openalpr/benchmarks/master/endtoend",
)
# Filenames known to exist in the OpenALPR benchmark end-to-end folders.
REAL_FILES = [
    "eu/eu1.jpg", "eu/eu2.jpg", "eu/eu3.jpg",
    "us/car1.jpg", "us/car2.jpg", "us/car3.jpg",
]

# Scenes whose plate region we crop (via the .txt annotation) and feed to the
# pipeline, to test the adhoc model on real-world plates.
CROP_SOURCES = (
    [f"us/car{i}" for i in range(1, 16)] +
    [f"eu/eu{i}" for i in range(1, 16)]
)
CROP_PAD = 5


def _fetch(url):
    try:
        r = requests.get(url, timeout=15)
        return r.content if r.status_code == 200 and r.content else None
    except requests.RequestException:
        return None


def download_real_crops(out_dir):
    """Download scene + annotation, crop the plate, and label it with the GT text."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {}
    for rel in CROP_SOURCES:
        img_bytes = _fetch(f"{BASE_URL}/{rel}.jpg")
        txt = _fetch(f"{BASE_URL}/{rel}.txt")
        if not img_bytes or not txt:
            continue
        line = txt.decode("utf-8", "ignore").splitlines()[0].split("\t")
        if len(line) < 6:
            continue
        x, y, w, h = (int(v) for v in line[1:5])
        gt = line[5].strip().upper()

        scene = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        H, W = scene.shape[:2]
        x0, y0 = max(0, x - CROP_PAD), max(0, y - CROP_PAD)
        x1, y1 = min(W, x + w + CROP_PAD), min(H, y + h + CROP_PAD)
        crop = scene[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        name = rel.replace("/", "_")
        path = out_dir / f"{name}_{gt}.png"
        cv2.imwrite(str(path), crop)
        labels[path.name] = gt
        print(f"  cropped {rel}  plate='{gt}'  ({x1 - x0}x{y1 - y0})")

    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    return labels


def download_real(out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for rel in REAL_FILES:
        url = f"{BASE_URL}/{rel}"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and resp.content:
                dst = out_dir / rel.replace("/", "_")
                dst.write_bytes(resp.content)
                saved.append(dst)
                print(f"  downloaded {rel}  ({len(resp.content)//1024} KB)")
            else:
                print(f"  skip {rel}  (HTTP {resp.status_code})")
        except requests.RequestException as exc:
            print(f"  skip {rel}  ({exc})")
    return saved


def main():
    config.ensure_dirs()
    print("Generating synthetic plate crops...")
    synth = write_default_plates(config.SAMPLES_DIR / "synthetic")
    for p in synth:
        print(f"  wrote {p.name}")

    print(f"\nDownloading real scenes from OpenALPR benchmark ({BASE_URL})...")
    real = download_real(config.SAMPLES_DIR / "real")

    print("\nCropping real plates (using the dataset annotations)...")
    crops = download_real_crops(config.SAMPLES_DIR / "real_crops")

    print(f"\nDone. {len(synth)} synthetic + {len(real)} real scenes + "
          f"{len(crops)} real plate crops in {config.SAMPLES_DIR}")
    if not real:
        print("No real images downloaded (offline?) - synthetic crops are enough to run the demo.")


if __name__ == "__main__":
    main()
