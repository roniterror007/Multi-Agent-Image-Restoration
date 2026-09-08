# reference/reference_retrieval.py
import json
import os
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = BASE_DIR / "data" / "good_pics"
IMAGES_DIR = BASE_DIR / "data" / "images"
WORKSPACE_DIR = BASE_DIR / "data" / "color_reference_workspace"


def compute_sobel_descriptor(image_path: Path) -> np.ndarray:
    """Computes a structural descriptor using Sobel edge gradients."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((64,))

    img = cv2.resize(img, (128, 128))
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)

    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Compute 64-bin orientation histogram weighted by magnitude
    hist, _ = np.histogram(angle.ravel(), bins=64, range=(0, 360), weights=mag.ravel())
    hist = hist / (hist.sum() + 1e-7)  # Normalize
    return hist


def build_index():
    """Indexes all reference images."""
    print("Building reference index...")
    index = {}
    for img_path in REFERENCES_DIR.glob("*.*"):
        if img_path.suffix.lower() in [".jpg", ".png"]:
            index[img_path.name] = compute_sobel_descriptor(img_path)
    return index


def find_monochrome_candidates(index: dict, limit=100):
    """Scans for grayscale images and matches them to the index."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    candidates = {}
    count = 0

    # Pre-compute index matrix for fast cosine similarity
    ref_names = list(index.keys())
    ref_matrix = np.vstack(list(index.values()))

    for img_path in IMAGES_DIR.glob("*.*"):
        if count >= limit:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Simple monochrome check: if saturation channel is very low
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_sat = hsv[:, :, 1].mean()

        if mean_sat < 14.0:  # Matches your documentation threshold
            desc = compute_sobel_descriptor(img_path)

            # Cosine similarity matching
            similarities = np.dot(ref_matrix, desc)
            top_3_indices = np.argsort(similarities)[-3:][::-1]

            matches = [
                {"reference": ref_names[i], "score": float(similarities[i])}
                for i in top_3_indices
            ]
            candidates[img_path.name] = matches
            count += 1

    out_file = WORKSPACE_DIR / "color_reference_candidates.json"
    with open(out_file, "w") as f:
        json.dump(candidates, f, indent=4)
    print(f"Staged {count} monochrome candidates to {out_file}")


if __name__ == "__main__":
    idx = build_index()
    find_monochrome_candidates(idx)
