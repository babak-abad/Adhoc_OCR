"""Central configuration for the adhoc license-plate OCR pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# --- Directories ---
DATA_DIR = ROOT / "data"
SAMPLES_DIR = DATA_DIR / "samples"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
STEPS_DIR = OUTPUTS_DIR / "steps"
MODEL_PATH = MODELS_DIR / "ocr_model.joblib"

# --- Plate normalization (stage 1) ---
PLATE_WIDTH = 300
PLATE_HEIGHT = 60

# --- Adaptive threshold (stage 2) ---
ADAPTIVE_BLOCK_SIZE = 21   # must be odd
ADAPTIVE_C = 10
PRE_BLUR_KSIZE = 3

# --- Character normalization (for feature extraction) ---
CHAR_SIZE = 32             # normalized square side
ZONE_GRID = 6              # zoning grid -> ZONE_GRID**2 density features
PROJECTION_BINS = 16       # bins per projection profile

# --- Geometric filtering of contours (stage 4), in 300x60 plate space ---
MIN_CHAR_WIDTH = 3
MAX_CHAR_WIDTH = 55
MIN_CHAR_HEIGHT = 22
MAX_CHAR_HEIGHT = 60
MIN_ASPECT = 0.10          # width / height
MAX_ASPECT = 1.10
MIN_AREA = 50              # contour pixel area
MIN_FILL = 0.10            # contour area / bounding-box area

# --- Classifier: 'svm' | 'knn' | 'mlp' ---
CLASSIFIER = "svm"

# --- Target alphabet (36 classes) ---
CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def ensure_dirs() -> None:
    for d in (DATA_DIR, SAMPLES_DIR, CACHE_DIR, MODELS_DIR, OUTPUTS_DIR, STEPS_DIR):
        d.mkdir(parents=True, exist_ok=True)
