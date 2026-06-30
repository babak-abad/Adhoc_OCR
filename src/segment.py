"""Stage 3-4: find contours and keep only the ones that look like characters."""
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class CharBox:
    x: int
    y: int
    w: int
    h: int
    area: float          # contour pixel area
    aspect: float        # w / h
    fill: float          # area / (w * h)
    accepted: bool
    reason: str          # why it was rejected ('' if accepted)


def find_contours(binary):
    """Stage 3 - every white blob is a contour candidate.

    RETR_LIST (not RETR_EXTERNAL) so that characters sitting *inside* a plate
    frame are still returned; the frame itself is dropped later by its size.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def _classify_box(contour):
    x, y, w, h = cv2.boundingRect(contour)
    area = cv2.contourArea(contour)
    aspect = w / h if h else 0.0
    fill = area / (w * h) if w * h else 0.0

    reason = ""
    if w < config.MIN_CHAR_WIDTH or w > config.MAX_CHAR_WIDTH:
        reason = "width"
    elif h < config.MIN_CHAR_HEIGHT or h > config.MAX_CHAR_HEIGHT:
        reason = "height"
    elif aspect < config.MIN_ASPECT or aspect > config.MAX_ASPECT:
        reason = "aspect"
    elif area < config.MIN_AREA:
        reason = "area"
    elif fill < config.MIN_FILL:
        reason = "fill"

    return CharBox(
        x=x, y=y, w=w, h=h, area=area, aspect=aspect, fill=fill,
        accepted=(reason == ""), reason=reason,
    )


def filter_characters(contours):
    """Stage 4 - reject non-character blobs by their geometric properties.

    Returns every candidate (with its accept/reject verdict) and the accepted
    boxes already sorted left-to-right, which is the plate reading order.
    """
    all_boxes = [_classify_box(c) for c in contours]
    kept = _suppress_nested([b for b in all_boxes if b.accepted])
    kept = _filter_by_consistency(kept)
    accepted = _trim_isolated(sorted(kept, key=lambda b: b.x))
    return all_boxes, accepted


def _trim_isolated(boxes):
    """Drop isolated leading/trailing boxes (EU country band, bolts, dirt).

    Characters on a plate are roughly evenly spaced; a box separated from the
    rest by a much larger gap than the median spacing is almost never a glyph.
    """
    if len(boxes) < 4:
        return boxes
    gaps = [boxes[i + 1].x - (boxes[i].x + boxes[i].w) for i in range(len(boxes) - 1)]
    med_gap = float(np.median(gaps))
    if med_gap <= 0:
        return boxes
    while len(boxes) >= 4 and gaps and gaps[0] > 2.5 * med_gap:
        boxes[0].accepted, boxes[0].reason = False, "isolated"
        boxes, gaps = boxes[1:], gaps[1:]
    while len(boxes) >= 4 and gaps and gaps[-1] > 2.5 * med_gap:
        boxes[-1].accepted, boxes[-1].reason = False, "isolated"
        boxes, gaps = boxes[:-1], gaps[:-1]
    return boxes


def _filter_by_consistency(boxes):
    """Real plates have characters of consistent height aligned on one row.

    Drop outliers (EU country bands, state names, screws) whose height or
    vertical centre disagrees with the median. Only applied when there are
    enough boxes to trust the median.
    """
    if len(boxes) < 4:
        return boxes
    heights = np.array([b.h for b in boxes])
    centers = np.array([b.y + b.h / 2 for b in boxes])
    med_h, med_c = float(np.median(heights)), float(np.median(centers))
    kept = []
    for b in boxes:
        if (0.65 * med_h <= b.h <= 1.35 * med_h and
                abs((b.y + b.h / 2) - med_c) <= 0.40 * med_h):
            kept.append(b)
        else:
            b.accepted, b.reason = False, "outlier"
    return kept


def _contains(outer, inner):
    return (outer.x <= inner.x and outer.y <= inner.y and
            outer.x + outer.w >= inner.x + inner.w and
            outer.y + outer.h >= inner.y + inner.h and
            outer is not inner)


def _suppress_nested(boxes):
    """Drop boxes nested inside a larger box (character holes, inner frame)."""
    by_area = sorted(boxes, key=lambda b: b.w * b.h, reverse=True)
    kept = []
    for b in by_area:
        if not any(_contains(k, b) for k in kept):
            kept.append(b)
        else:
            b.accepted, b.reason = False, "nested"
    return kept
