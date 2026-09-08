# tests/test_segmentation_vlm_robustness.py
import pytest

from core.agent_controller import validate_adjust_lab_call


@pytest.mark.vlm_safety
def test_out_of_bounds_vlm_hallucination():
    """Ensures the schema validation blocks mathematically dangerous AI proposals."""
    hallucination = {
        "action": "extreme_color_cast_fix",
        "delta_l_ticks": -999,  # Dangerous! Way outside the +/- 40 limit.
        "delta_a_ticks": 0,
        "delta_b_ticks": 0,
        "strength_ticks": 10,
    }
    result = validate_adjust_lab_call(hallucination)
    assert result is None, "Out-of-bounds VLM ticks must be strictly rejected."


@pytest.mark.vlm_safety
def test_invalid_json_type_coercion():
    """Ensures the VLM cannot inject float values where integers are required."""
    bad_type_payload = {
        "action": "smooth_fix",
        "delta_l_ticks": 10.5,  # Invalid type
        "delta_a_ticks": 0,
        "delta_b_ticks": 0,
        "strength_ticks": 10,
    }
    result = validate_adjust_lab_call(bad_type_payload)
    assert (
        result is None
    ), "Float values must be rejected to prevent type coercion crashes."
