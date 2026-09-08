# core/reviewer_agent.py
import json
from pathlib import Path

import imagehash

PHASH_THRESHOLD = 8  # Maximum Hamming distance to be considered a "match" (relaxed for robustness)


def hamming_distance(hash1_str: str, hash2_str: str) -> int:
    """Calculates the bitwise difference between two perceptual hashes."""
    if not hash1_str or not hash2_str:
        return 999
    h1 = imagehash.hex_to_hash(hash1_str)
    h2 = imagehash.hex_to_hash(hash2_str)
    return h1 - h2


def find_approved_fix(current_phash: str, feedback_dir: str) -> dict:
    """Scans historical overrides for a near-duplicate match (distance <= 3)."""
    override_file = Path(feedback_dir) / "image_overrides.json"

    if not override_file.exists():
        return None

    with open(override_file, "r") as f:
        try:
            overrides = json.load(f)
        except json.JSONDecodeError:
            return None

    best_match = None
    lowest_distance = 999

    for saved_hash, lab_shifts in overrides.items():
        distance = hamming_distance(current_phash, saved_hash)

        # If perfect match or near-duplicate within threshold
        if distance <= PHASH_THRESHOLD and distance < lowest_distance:
            lowest_distance = distance
            best_match = lab_shifts

    return best_match
