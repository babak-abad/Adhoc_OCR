"""Run the adhoc pipeline on REAL cropped plates and report success / failure.

    python steps/real_world.py

Reads data/samples/real_crops/*.png (+ labels.json ground truth), runs
recognition on each, and writes a success/failure gallery to
outputs/real_world_gallery.png.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, recognize  # noqa: E402


def _char_accuracy(pred, gt):
    if not gt:
        return 0.0
    hits = sum(p == g for p, g in zip(pred, gt))
    return hits / max(len(gt), len(pred))


def evaluate():
    crop_dir = config.SAMPLES_DIR / "real_crops"
    labels = json.loads((crop_dir / "labels.json").read_text())
    pipeline, charset = recognize.load_model()

    rows = []
    for name, gt in sorted(labels.items()):
        image = cv2.imread(str(crop_dir / name))
        if image is None:
            continue
        result = recognize.recognize_plate(image, pipeline, charset)
        ok = result.text == gt
        rows.append({
            "image": result.pre.resized, "gt": gt, "pred": result.text,
            "ok": ok, "char_acc": _char_accuracy(result.text, gt),
        })
    return rows


def report(rows):
    ok = [r for r in rows if r["ok"]]
    print(f"\nReal-world plates: {len(rows)}   exact matches: {len(ok)}/{len(rows)}")
    mean_char = np.mean([r["char_acc"] for r in rows]) if rows else 0.0
    print(f"Mean character accuracy: {mean_char:.0%}\n")
    for r in rows:
        flag = "OK  " if r["ok"] else "FAIL"
        print(f"  [{flag}] gt={r['gt']:<9} pred={r['pred']:<11} char_acc={r['char_acc']:.0%}")


def build_gallery(rows, path=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    # Best first - the figure then reads as a success-to-failure spectrum.
    rows = sorted(rows, key=lambda r: -r["char_acc"])
    n = len(rows)
    cols = 3
    rows_n = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.6, rows_n * 1.7))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, r in zip(np.ravel(axes), rows):
        ax.imshow(cv2.cvtColor(r["image"], cv2.COLOR_BGR2RGB))
        if r["ok"]:
            color, tag = "#1a7f37", "SUCCESS"          # green
        elif r["char_acc"] >= 0.5:
            color, tag = "#bf8700", "PARTIAL"          # amber
        else:
            color, tag = "#cf222e", "FAIL"             # red
        ax.set_title(f"{tag}  ({r['char_acc']:.0%})\nGT: {r['gt']}  ->  {r['pred'] or '(none)'}",
                     color=color, fontsize=9)
    fig.suptitle("Adhoc OCR on real-world plates (OpenALPR Benchmark crops)",
                 fontsize=13)
    fig.tight_layout()
    path = path or (config.OUTPUTS_DIR / "real_world_gallery.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    config.ensure_dirs()
    rows = evaluate()
    report(rows)
    out = build_gallery(rows)
    if out:
        print(f"\nGallery -> {out}")


if __name__ == "__main__":
    main()
