# core/feedback_store.py
import json
import os
from pathlib import Path

import imagehash
from PIL import Image


def compute_perceptual_hash(image_pil: Image.Image) -> str:
    """Computes a 64-bit perceptual hash for near-duplicate detection."""
    # Convert to grayscale and resize internally, returning a hex string
    return str(imagehash.phash(image_pil))


def save_feedback(
    feedback_dir: str,
    original_filename: str,
    record: dict,
    annotation_pil: Image.Image = None,
):
    """Appends operator review decisions to the JSONL audit ledger."""
    feedback_path = Path(feedback_dir)
    feedback_path.mkdir(parents=True, exist_ok=True)

    jsonl_file = feedback_path / "feedback.jsonl"

    # Save the annotated visual for the audit trail
    if annotation_pil:
        annotated_path = feedback_path / f"annotated_{original_filename}"
        annotation_pil.save(annotated_path)
        record["annotation_file"] = str(annotated_path.name)

    # Append to the continuous learning ledger
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def save_override(feedback_dir: str, image_phash: str, lab_shifts: dict):
    """Saves a proven, operator-approved mathematical fix to memory."""
    feedback_path = Path(feedback_dir)
    feedback_path.mkdir(parents=True, exist_ok=True)

    override_file = feedback_path / "image_overrides.json"

    overrides = {}
    if override_file.exists():
        with open(override_file, "r") as f:
            try:
                overrides = json.load(f)
            except json.JSONDecodeError:
                pass

    overrides[image_phash] = lab_shifts

    with open(override_file, "w") as f:
        json.dump(overrides, f, indent=4)
