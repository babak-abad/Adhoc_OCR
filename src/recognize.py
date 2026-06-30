"""Full recognition pipeline (no learning): image -> plate string.

Wires stages 1-5 together and applies the trained classifier.
"""
from dataclasses import dataclass, field

import cv2
import joblib
import numpy as np

from . import config, features, preprocess, segment


@dataclass
class CharResult:
    box: segment.CharBox
    label: str
    proba: float


@dataclass
class PlateResult:
    text: str
    chars: list = field(default_factory=list)        # list[CharResult]
    pre: preprocess.Preprocessed = None
    all_boxes: list = field(default_factory=list)     # every candidate box


def load_model(path=config.MODEL_PATH):
    payload = joblib.load(path)
    return payload["pipeline"], payload["charset"]


def _predict_char(pipeline, charset, char_bin, char_color):
    vec = features.extract_features(char_bin, char_color).reshape(1, -1)
    idx = int(pipeline.predict(vec)[0])
    if hasattr(pipeline, "predict_proba"):
        proba = float(pipeline.predict_proba(vec)[0][idx])
    else:
        proba = 1.0
    return charset[idx], proba


def recognize_plate(image, pipeline, charset):
    pre = preprocess.preprocess_plate(image)
    contours = segment.find_contours(pre.binary)
    all_boxes, accepted = segment.filter_characters(contours)

    chars = []
    for box in accepted:
        char_bin = pre.binary[box.y:box.y + box.h, box.x:box.x + box.w]
        char_color = pre.resized[box.y:box.y + box.h, box.x:box.x + box.w]
        label, proba = _predict_char(pipeline, charset, char_bin, char_color)
        chars.append(CharResult(box=box, label=label, proba=proba))

    text = "".join(c.label for c in chars)
    return PlateResult(text=text, chars=chars, pre=pre, all_boxes=all_boxes)


def annotate(result, scale=3):
    """Draw accepted boxes + predicted labels on an upscaled plate image."""
    canvas = cv2.resize(result.pre.resized, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_NEAREST)
    for c in result.chars:
        b = c.box
        p1 = (b.x * scale, b.y * scale)
        p2 = ((b.x + b.w) * scale, (b.y + b.h) * scale)
        cv2.rectangle(canvas, p1, p2, (0, 200, 0), 2)
        cv2.putText(canvas, c.label, (b.x * scale, b.y * scale - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return canvas
