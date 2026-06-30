"""Stage 5: feature extraction.

We start with the two families the project asked for:
  * geometrical features  - image moments (Hu + normalized central), shape
                            descriptors, zoning densities and projection
                            profiles (all translation/scale invariant).
  * hue / colour features - HSV statistics over the character pixels.

Tchebichev / Zernike moments are intentionally left out for now; they can be
appended here if accuracy proves insufficient.
"""
import cv2
import numpy as np

from . import config


def normalize_char(char_bin, size=config.CHAR_SIZE):
    """Pad a character crop to a square and resize to a canonical size."""
    h, w = char_bin.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side), dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = char_bin
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def _hu_log(moments):
    hu = cv2.HuMoments(moments).flatten()
    # Log transform compresses the huge dynamic range into a stable scale.
    return np.array([-np.sign(h) * np.log10(abs(h) + 1e-30) for h in hu])


def _zoning_density(norm_char, grid=config.ZONE_GRID):
    cells = []
    step = norm_char.shape[0] // grid
    for gy in range(grid):
        for gx in range(grid):
            cell = norm_char[gy * step:(gy + 1) * step, gx * step:(gx + 1) * step]
            cells.append(float(np.count_nonzero(cell)) / (cell.size or 1))
    return np.array(cells)


def _projection_profiles(norm_char, bins=config.PROJECTION_BINS):
    fg = (norm_char > 0).astype(np.float32)
    h_proj = fg.sum(axis=1)   # one value per row
    v_proj = fg.sum(axis=0)   # one value per column

    def _resample(profile):
        idx = np.linspace(0, len(profile) - 1, bins)
        sampled = np.interp(idx, np.arange(len(profile)), profile)
        total = sampled.sum()
        return sampled / total if total else sampled

    return np.concatenate([_resample(h_proj), _resample(v_proj)])


def geometric_features(char_bin):
    """Geometric feature block computed from the binary character crop."""
    moments = cv2.moments(char_bin, binaryImage=True)

    h, w = char_bin.shape[:2]
    area = float(np.count_nonzero(char_bin))
    aspect = w / h if h else 0.0
    extent = area / (w * h) if w * h else 0.0

    cnts, _ = cv2.findContours(char_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(biggest))
        solidity = cv2.contourArea(biggest) / hull_area if hull_area else 0.0
    else:
        solidity = 0.0

    nu = np.array([
        moments["nu20"], moments["nu11"], moments["nu02"],
        moments["nu30"], moments["nu21"], moments["nu12"], moments["nu03"],
    ])

    norm_char = normalize_char(char_bin)
    return np.concatenate([
        _hu_log(moments),                 # 7
        nu,                               # 7
        [aspect, extent, solidity],       # 3
        _zoning_density(norm_char),       # ZONE_GRID**2
        _projection_profiles(norm_char),  # 2 * PROJECTION_BINS
    ])


def hue_features(char_color, char_bin):
    """HSV statistics over the character pixels of the colour crop."""
    char_color = cv2.resize(char_color, (char_bin.shape[1], char_bin.shape[0]))
    hsv = cv2.cvtColor(char_color, cv2.COLOR_BGR2HSV)
    mask = char_bin > 0
    if not mask.any():
        return np.zeros(6 + 6)

    h, s, v = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]
    stats = np.array([
        h.mean() / 180.0, h.std() / 180.0,
        s.mean() / 255.0, s.std() / 255.0,
        v.mean() / 255.0, v.std() / 255.0,
    ])
    hist = cv2.calcHist([hsv], [0], mask.astype(np.uint8), [6], [0, 180]).flatten()
    hist = hist / (hist.sum() or 1)
    return np.concatenate([stats, hist])


def extract_features(char_bin, char_color):
    """Full feature vector = geometric block + hue block."""
    return np.concatenate([
        geometric_features(char_bin),
        hue_features(char_color, char_bin),
    ]).astype(np.float32)


def feature_length():
    dummy_bin = np.zeros((config.CHAR_SIZE, config.CHAR_SIZE), np.uint8)
    dummy_bin[8:24, 8:24] = 255
    dummy_color = np.zeros((config.CHAR_SIZE, config.CHAR_SIZE, 3), np.uint8)
    return len(extract_features(dummy_bin, dummy_color))
