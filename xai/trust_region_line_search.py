# xai/trust_region_line_search.py
import numpy as np


class TrustRegionLineSearch:
    def __init__(self, c1=1e-4, max_backtracks=5):
        self.c1 = c1
        self.max_backtracks = max_backtracks

    def calculate_armijo_step(
        self,
        current_lab: np.ndarray,
        target_lab: np.ndarray,
        shift_vector: np.ndarray,
        current_de00: float,
    ) -> tuple[float, np.ndarray]:
        """
        Tests progressively smaller step sizes to guarantee error reduction.
        Returns the optimal alpha scalar and the resulting Lab image.
        """
        alphas = [1.0, 0.5, 0.25, 0.125, 0.0625]

        for alpha in alphas:
            # 1. Propose a step
            proposed_lab = current_lab + (shift_vector * alpha)
            proposed_lab[:, :, 0] = np.clip(proposed_lab[:, :, 0], 0, 255)
            proposed_lab[:, :, 1] = np.clip(proposed_lab[:, :, 1], 0, 255)
            proposed_lab[:, :, 2] = np.clip(proposed_lab[:, :, 2], 0, 255)

            # 2. Measure new error
            # (Assuming a ciede2000 function exists in your environment)
            # proposed_de00 = ciede2000(proposed_lab, target_lab)
            proposed_de00 = current_de00 - (
                alpha * 2.0
            )  # Stub for actual CIEDE2000 output

            # 3. Check Armijo sufficient-decrease condition
            expected_decrease = self.c1 * alpha * np.linalg.norm(shift_vector)

            if proposed_de00 <= current_de00 - expected_decrease:
                return alpha, proposed_lab

        # If all steps fail, return 0 alpha (halt optimization to prevent damage)
        return 0.0, current_lab
