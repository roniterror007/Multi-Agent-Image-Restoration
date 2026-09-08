# reference/quality_evaluator.py
import cv2
import numpy as np


def calculate_blur_laplacian(image_bgr: np.ndarray) -> float:
    """Calculates image blur using the variance of the Laplacian.
    Lower variance = more blur (fewer sharp edges).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_clipping_ratio(image_bgr: np.ndarray, mask: np.ndarray = None) -> float:
    """Returns the percentage of pixels that are completely blown out (>= 250)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = gray[mask > 0]

    if gray.size == 0:
        return 0.0

    clipped_pixels = np.sum(gray >= 250)
    return float(clipped_pixels / gray.size)


def calculate_mask_leakage(image_bgr: np.ndarray, mask: np.ndarray) -> float:
    """Checks if the mask accidentally included the white catalog background.
    High leakage means the background is incorrectly being treated as the object.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Area where mask is active, but the image is almost pure white
    leaked_pixels = np.sum((mask > 0) & (gray >= 245))
    mask_area = np.sum(mask > 0)

    if mask_area == 0:
        return 0.0
    return float(leaked_pixels / mask_area)


def get_quality_report(image_bgr: np.ndarray, mask: np.ndarray = None) -> dict:
    """Compiles a full diagnostic report for the internal critic."""
    return {
        "laplacian_variance": calculate_blur_laplacian(image_bgr),
        "clipping_ratio": calculate_clipping_ratio(image_bgr, mask),
        "mask_leakage": (
            calculate_mask_leakage(image_bgr, mask) if mask is not None else 0.0
        ),
    }
