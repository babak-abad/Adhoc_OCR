"""Stage 6 (learning): train the character classifier and persist it.

Run directly:  python -m src.train  [--classifier svm|knn|mlp]
"""
import argparse
import time

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import config, dataset


def build_classifier(name=config.CLASSIFIER):
    """Scaler + classifier pipeline, selectable by name."""
    if name == "svm":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)
    elif name == "knn":
        clf = KNeighborsClassifier(n_neighbors=3, weights="distance")
    elif name == "mlp":
        clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                            early_stopping=True, random_state=0)
    else:
        raise ValueError(f"Unknown classifier '{name}' (use svm|knn|mlp)")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def train(classifier=config.CLASSIFIER, samples_per_font=dataset.VARIANTS_PER_FONT,
          use_cache=True):
    config.ensure_dirs()
    print(f"[1/4] Generating synthetic characters (this can take a minute)...")
    X, y = dataset.load_or_generate(samples_per_font=samples_per_font, use_cache=use_cache)
    print(f"      samples={len(X)}  features={X.shape[1]}  classes={len(set(y))}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y
    )

    print(f"[2/4] Training '{classifier}' classifier...")
    t0 = time.time()
    pipeline = build_classifier(classifier)
    pipeline.fit(X_tr, y_tr)
    print(f"      fit done in {time.time() - t0:.1f}s")

    print("[3/4] Evaluating on held-out split...")
    pred = pipeline.predict(X_te)
    acc = accuracy_score(y_te, pred)
    target_names = list(config.CHARSET)
    print(f"      validation accuracy = {acc:.4f}")
    print(classification_report(y_te, pred, target_names=target_names, zero_division=0))

    payload = {"pipeline": pipeline, "charset": config.CHARSET, "classifier": classifier}
    joblib.dump(payload, config.MODEL_PATH)
    print(f"[4/4] Saved model -> {config.MODEL_PATH}")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Train the adhoc OCR classifier")
    parser.add_argument("--classifier", default=config.CLASSIFIER, choices=["svm", "knn", "mlp"])
    parser.add_argument("--samples-per-font", type=int, default=dataset.VARIANTS_PER_FONT)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    train(classifier=args.classifier, samples_per_font=args.samples_per_font,
          use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
