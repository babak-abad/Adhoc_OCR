# Adhoc OCR - License-Plate Character Recognition

A small, fully transparent OCR for the characters of an **already-cropped**
license plate. Every stage is visible and explainable. It is an educational
adhoc model, not a production ALPR system.

**Repository:** <https://github.com/babak-abad/Adhoc_OCR>

```bash
git clone https://github.com/babak-abad/Adhoc_OCR.git
cd Adhoc_OCR
```

## Pipeline

1. **Resize** the plate to a canonical `300 x 60` (`src/preprocess.py`)
2. **Adaptive threshold** -> white characters on black (`src/preprocess.py`)
3. **Find all contours** (`src/segment.py`)
4. **Geometric filtering** of non-characters - width/height/aspect/area/fill, plus nested-box suppression (`src/segment.py`)
5. **Feature extraction** - geometric moments (Hu, normalized central, zoning, projections) + hue/HSV features (`src/features.py`)
6. **Classify** each character with an SVM (KNN / MLP selectable) and read left-to-right (`src/train.py`, `src/recognize.py`)

## Data sources

> **No images or trained model ship with this repository.** The `data/` and
> `models/` folders are git-ignored, so after cloning you must obtain the data
> yourself. There are two kinds of images, and you get them with two commands:
>
> ```bash
> # A) generate the synthetic training data + clean test plates (no internet needed)
> python download_samples.py          # writes data/samples/synthetic/
> python -m src.train --classifier svm  # builds & caches the font-rendered training set, saves models/ocr_model.joblib
>
> # B) (optional) download real car scenes + crop their plates (needs internet)
> python download_samples.py          # also fetches data/samples/real/ + real_crops/
> ```
>
> `python download_samples.py` does both the synthetic generation and the real
> download in one run; if you are offline it simply skips the downloads and the
> synthetic plates are enough to run the whole pipeline.

The project uses two *separate* sets of images, obtained two different ways:

- **Training images for the classifier are *created*, not downloaded.** Every
  character `0-9` / `A-Z` is rendered from ~30 installed system fonts and
  augmented (rotation, shear, low-res blur, noise, plate-like colours) in
  [`src/dataset.py`](src/dataset.py). The label is the glyph we rendered, so no
  labelled character dataset is needed.
- **Sample / test plate scenes are *downloaded*** by
  [`download_samples.py`](download_samples.py) from the public **OpenALPR
  Benchmark** dataset (end-to-end EU + US car photos with annotations, no
  registration): <https://github.com/openalpr/benchmarks>. The script reads each
  scene's `.txt` annotation, crops the plate region, and labels the crop with the
  ground-truth text. Base URL is configurable via `SAMPLE_BASE_URL` in `.env`.
  A larger annotated alternative is the Kaggle
  [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection)
  dataset.

In addition, `download_samples.py` generates a handful of clean synthetic plate
crops (`src/plate_gen.py`) as guaranteed-working pipeline inputs.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

## Usage

```bash
# download sample scenes + generate synthetic test plates
python download_samples.py

# LEARNING stage - train + save the classifier
python -m src.train --classifier svm          # or: knn | mlp

# RUN the whole project on one plate (no learning)
python run_pipeline.py --image data/samples/synthetic/plate_7H5K829.png

# STEP-BY-STEP stage images (for the article) -> outputs/steps/
python steps/demo_steps.py --image data/samples/synthetic/plate_7H5K829.png
```

## Layout

```
src/            pipeline modules (config, preprocess, segment, features, dataset, train, recognize, plate_gen)
steps/          step-by-step visualization
run_pipeline.py whole-project runner (no learning)
download_samples.py  sample fetcher + synthetic plate generator
article/        Joomla-ready HTML article + images
outputs/        generated results and stage images
models/         trained model (ocr_model.joblib)
```

## Limitations

Works on clean, high-contrast, well-cropped Latin plates only. It does **not**
handle blur, shadow, dust, joined/connected characters (especially script
languages such as Persian/Arabic), washed-out glyphs, heavy skew, or non-Latin
alphabets. See `https://ai-programmer.com/index.php/projects/ad-hoc-ocr` for details.

Configuration (sizes, thresholds, filter rules, classifier) lives in `src/config.py`.
