"""Stage 1-2: resize the cropped plate and binarize it with adaptive thresholding."""
from dataclasses import dataclass

import cv2
import numpy as np

from . import config


@dataclass
class Preprocessed:
    resized: np.ndarray   # BGR, PLATE_WIDTH x PLATE_HEIGHT
    gray: np.ndarray      # grayscale
    binary: np.ndarray    # white characters on black background (uint8 0/255)


def resize_plate(image, width=config.PLATE_WIDTH, height=config.PLATE_HEIGHT):
    """Stage 1 - bring every plate to one canonical size."""
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def to_gray(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def adaptive_threshold(gray, block_size=config.ADAPTIVE_BLOCK_SIZE, c=config.ADAPTIVE_C):
    """Stage 2 - local thresholding robust to uneven lighting.

    Characters are darker than the plate, so THRESH_BINARY_INV produces white
    glyphs on a black background, which is what cv2.findContours expects.
    """
    if config.PRE_BLUR_KSIZE > 1:
        gray = cv2.GaussianBlur(gray, (config.PRE_BLUR_KSIZE, config.PRE_BLUR_KSIZE), 0)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, c
    )
    # Light opening removes salt noise without eroding the strokes.
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def preprocess_plate(image):
    """Run stage 1 + stage 2 and return every intermediate image."""
    resized = resize_plate(image)
    gray = to_gray(resized)
    binary = adaptive_threshold(gray)
    return Preprocessed(resized=resized, gray=gray, binary=binary)
