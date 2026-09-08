# tests/test_pipeline_e2e.py
import pytest

from core.reviewer_agent import hamming_distance


@pytest.mark.integration
def test_phash_near_duplicate_detection():
    """Validates that slight visual changes still trigger historical human fixes."""
    # Stub hashes mimicking imagehash outputs
    original_hash = "ffff81818181ffff"

    # 1 bit difference (e.g., slight lighting change on the factory floor)
    duplicate_hash = "ffff81818181fffe"

    # Completely distinct image
    distinct_hash = "0000ffff0000ffff"

    # The threshold must allow near-duplicates but reject distinct images
    assert hamming_distance(original_hash, duplicate_hash) <= 3
    assert hamming_distance(original_hash, distinct_hash) > 3
