# core/agent_controller.py
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class AdjustLabCall:
    """Strict GBNF schema enforced on the VLM to prevent hallucinations."""

    action: str
    delta_l_ticks: int
    delta_a_ticks: int
    delta_b_ticks: int
    strength_ticks: int = 10
    reasoning: str = ""


def validate_adjust_lab_call(payload: dict) -> Optional[AdjustLabCall]:
    """Ensures VLM outputs are mathematically safe before execution."""
    try:
        # Prevent type coercion vulnerabilities (Ensure strict integers)
        if not isinstance(payload.get("delta_l_ticks"), int):
            return None
        if not isinstance(payload.get("delta_a_ticks"), int):
            return None
        if not isinstance(payload.get("delta_b_ticks"), int):
            return None

        # Enforce physical bounds (+/- 40 ticks max)
        if not (-40 <= payload["delta_l_ticks"] <= 40):
            return None
        if not (-40 <= payload["delta_a_ticks"] <= 40):
            return None
        if not (-40 <= payload["delta_b_ticks"] <= 40):
            return None

        clean_payload = {k: v for k, v in payload.items() if k not in ("confidence_boost", "confidence")}
        return AdjustLabCall(**clean_payload)
    except (KeyError, ValueError, TypeError):
        return None


class ProportionalController:
    def __init__(self, image_bgr: np.ndarray):
        self.original_bgr = image_bgr.copy()
        # Convert to CIELAB space
        self.current_lab = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2LAB).astype(
            np.float32
        )
        self.alpha_damping = 1.0
        self.event_history = []

    def apply_lab_shift(
        self,
        l_shift: float,
        a_shift: float,
        b_shift: float,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Applies shift using a Gaussian color-distance soft mask to prevent box artifacts."""

        # Enforce minimum damping to prevent infinite loops
        if self.alpha_damping < 0.001:
            return False

        l_shift *= self.alpha_damping
        a_shift *= self.alpha_damping
        b_shift *= self.alpha_damping

        target_lab = self.current_lab.copy()

        if region is None:
            target_lab[:, :, 0] += l_shift
            target_lab[:, :, 1] += a_shift
            target_lab[:, :, 2] += b_shift
        else:
            # 1. Extract the "seed" color from the user's box
            left, top, right, bottom = region
            box_pixels = self.current_lab[top:bottom, left:right]
            if box_pixels.size == 0:
                return False
            seed_color = box_pixels.mean(axis=(0, 1))

            # 2. Calculate Color Distance across the ENTIRE image
            dist = np.linalg.norm(self.current_lab - seed_color, axis=2)

            # 3. Create a seamless, feathered soft mask (sigma = 12.0 Lab units tolerance)
            sigma = 12.0
            soft_mask = np.exp(-(dist**2) / (2 * sigma**2))
            soft_mask_3d = np.stack([soft_mask] * 3, axis=2)

            # 4. Apply the shift globally, weighted by the material similarity mask
            shift_tensor = np.array([l_shift, a_shift, b_shift], dtype=np.float32)
            target_lab = self.current_lab + (shift_tensor * soft_mask_3d)

        # Clip to valid CIELAB bounds
        target_lab[:, :, 0] = np.clip(target_lab[:, :, 0], 0, 255)
        target_lab[:, :, 1] = np.clip(target_lab[:, :, 1], 0, 255)
        target_lab[:, :, 2] = np.clip(target_lab[:, :, 2], 0, 255)

        self.current_lab = target_lab

        self.event_history.append(
            {
                "type": "smart_propagated_shift",
                "l_shift": l_shift,
                "a_shift": a_shift,
                "b_shift": b_shift,
                "damping": self.alpha_damping,
            }
        )
        return True
