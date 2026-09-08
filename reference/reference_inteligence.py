# reference/reference_inteligence.py
import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = BASE_DIR / "data" / "good_pics"
OUT_REPORT = BASE_DIR / "data" / "reference_report.json"


def analyze_reference_library():
    """Extracts statistical baselines from the golden standard images."""
    print(f"Scanning reference library: {REFERENCES_DIR}")

    if not REFERENCES_DIR.exists():
        print("Reference directory not found.")
        return

    stats = {
        "total_images": 0,
        "mean_brightness": [],
        "mean_saturation": [],
        "dominant_colors_lab": [],
    }

    for img_path in REFERENCES_DIR.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Convert spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        # Only analyze the object, ignoring the white background (>245)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray < 245

        if not np.any(mask):
            continue

        stats["total_images"] += 1
        stats["mean_brightness"].append(float(hsv[:, :, 2][mask].mean()))
        stats["mean_saturation"].append(float(hsv[:, :, 1][mask].mean()))

        mean_a = float(lab[:, :, 1][mask].mean())
        mean_b = float(lab[:, :, 2][mask].mean())
        stats["dominant_colors_lab"].append([mean_a, mean_b])

    # Aggregate
    report = {
        "library_size": stats["total_images"],
        "global_mean_brightness": float(np.mean(stats["mean_brightness"])),
        "global_mean_saturation": float(np.mean(stats["mean_saturation"])),
        "metallic_threshold_sat": float(
            np.percentile(stats["mean_saturation"], 15)
        ),  # Bottom 15% are likely metals
    }

    with open(OUT_REPORT, "w") as f:
        json.dump(report, f, indent=4)

    print(f"Intelligence report generated: {OUT_REPORT}")
    print(
        f"Calculated Metallic Saturation Threshold: {report['metallic_threshold_sat']:.1f}"
    )


if __name__ == "__main__":
    analyze_reference_library()
