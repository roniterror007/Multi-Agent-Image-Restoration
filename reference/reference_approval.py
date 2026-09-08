# reference/reference_approval.py
import json
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BASE_DIR / "data" / "color_reference_workspace"
REFERENCES_DIR = BASE_DIR / "data" / "good_pics"
IMAGES_DIR = BASE_DIR / "data" / "images"

AUTO_APPROVE_THRESHOLD = 0.95


def extract_color_prior(reference_path: Path) -> np.ndarray:
    """Extracts the dominant a* and b* channels from the reference."""
    img = cv2.imread(str(reference_path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Ignore background (assuming white > 240)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 240

    if not np.any(mask):
        return np.array([128.0, 128.0])

    mean_a = lab[:, :, 1][mask].mean()
    mean_b = lab[:, :, 2][mask].mean()
    return np.array([mean_a, mean_b])


def apply_color_prior(grayscale_path: Path, prior_ab: np.ndarray, output_path: Path):
    """Applies the reference a* and b* values to the grayscale structure."""
    img = cv2.imread(str(grayscale_path))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 240  # Only colorize the object, not the background

    # Apply the exact color shifts needed to reach the prior
    current_a = lab[:, :, 1][mask].mean()
    current_b = lab[:, :, 2][mask].mean()

    lab[:, :, 1][mask] += prior_ab[0] - current_a
    lab[:, :, 2][mask] += prior_ab[1] - current_b

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(output_path), bgr)


def process_approvals():
    candidates_file = WORKSPACE_DIR / "color_reference_candidates.json"
    approvals_log = WORKSPACE_DIR / "approvals.jsonl"
    out_dir = WORKSPACE_DIR / "colorized_output"
    out_dir.mkdir(exist_ok=True)

    if not candidates_file.exists():
        print("No candidates found. Run retrieval first.")
        return

    with open(candidates_file, "r") as f:
        candidates = json.load(f)

    with open(approvals_log, "a") as log:
        for target_img, matches in candidates.items():
            best_match = matches[0]

            if best_match["score"] >= AUTO_APPROVE_THRESHOLD:
                ref_path = REFERENCES_DIR / best_match["reference"]
                tgt_path = IMAGES_DIR / target_img
                out_path = out_dir / target_img

                if not ref_path.exists() or not tgt_path.exists():
                    continue

                # Extract and apply
                prior = extract_color_prior(ref_path)
                apply_color_prior(tgt_path, prior, out_path)

                # Log success
                log.write(
                    json.dumps(
                        {
                            "image": target_img,
                            "status": "approved",
                            "reference": best_match["reference"],
                        }
                    )
                    + "\n"
                )
                print(f"Auto-approved and colorized: {target_img}")


if __name__ == "__main__":
    process_approvals()
