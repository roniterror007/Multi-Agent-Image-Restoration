# core/degradation_classifier.py
"""
Centralized Degradation Detection Module.

Replaces scattered heuristics (smart_restore, mock_vlm_call, monochrome check)
with a single, deterministic classifier that outputs typed degradation reports.
Supports compound degradations (e.g., pink + underexposed).
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class DegradationType(Enum):
    CLEAN = "clean"
    PINK_CAST = "pink_cast"
    GREEN_CAST = "green_cast"
    YELLOW_CAST = "yellow_cast"
    BLUE_CAST = "blue_cast"
    GRAYSCALE_BW = "grayscale_bw"
    OVEREXPOSED = "overexposed"
    UNDEREXPOSED = "underexposed"
    WASHED_OUT = "washed_out"
    LOW_CONTRAST = "low_contrast"


@dataclass
class DegradationReport:
    """Typed degradation analysis for a single image."""
    primary: DegradationType
    secondary: List[DegradationType] = field(default_factory=list)
    severity: float = 0.0  # 0.0 = clean, 1.0 = completely destroyed
    metrics: dict = field(default_factory=dict)

    @property
    def is_degraded(self) -> bool:
        return self.primary != DegradationType.CLEAN

    @property
    def is_bw(self) -> bool:
        return self.primary == DegradationType.GRAYSCALE_BW

    @property
    def has_color_cast(self) -> bool:
        return self.primary in (
            DegradationType.PINK_CAST,
            DegradationType.GREEN_CAST,
            DegradationType.YELLOW_CAST,
            DegradationType.BLUE_CAST,
        )

    @property
    def has_exposure_issue(self) -> bool:
        return self.primary in (
            DegradationType.OVEREXPOSED,
            DegradationType.UNDEREXPOSED,
            DegradationType.WASHED_OUT,
            DegradationType.LOW_CONTRAST,
        )

    @property
    def all_degradations(self) -> List[DegradationType]:
        """Returns all detected degradation types (primary + secondary)."""
        return [self.primary] + self.secondary


class DegradationClassifier:
    """
    Multi-signal degradation classifier that examines Lab, HSV, and BGR
    statistics to produce a comprehensive DegradationReport.

    Replaces the fragmented detection logic previously scattered across
    smart_restore(), mock_vlm_call(), and inline monochrome checks.
    """

    # ------------- Thresholds (tuned for industrial part photography) --------

    # B/W detection: true grayscale has near-zero chroma variance
    BW_STD_A_THRESHOLD = 2.0
    BW_STD_B_THRESHOLD = 2.0
    BW_SATURATION_THRESHOLD = 8.0

    # Color cast detection thresholds (Lab a* and b*, centered at 128)
    PINK_A_THRESHOLD = 135.0
    GREEN_A_THRESHOLD = 120.0
    YELLOW_B_THRESHOLD = 140.0
    BLUE_B_THRESHOLD = 118.0

    # Alternative cast detection using BGR channel imbalance
    PINK_BGR_R_EXCESS = 30.0   # mean_r - mean_g
    PINK_BGR_B_EXCESS = 15.0   # mean_b_bgr - mean_g

    # Exposure thresholds
    OVEREXPOSED_MEAN_L = 210.0
    OVEREXPOSED_P95_L = 240.0
    UNDEREXPOSED_MEAN_L = 90.0
    WASHED_IQR_THRESHOLD = 30.0
    LOW_CONTRAST_STD_THRESHOLD = 25.0

    def classify(self, img_bgr: np.ndarray) -> DegradationReport:
        """
        Analyze an image and return a comprehensive DegradationReport.

        Args:
            img_bgr: Input image in BGR format (as loaded by cv2).

        Returns:
            DegradationReport with primary degradation, secondaries, and severity.
        """
        metrics = self._compute_metrics(img_bgr)
        degradations = []
        severity_scores = []

        # ---- Exposure Issues (check BEFORE B/W to disambiguate washed-out) ----
        exposure_type, exposure_severity = self._detect_exposure_issue(metrics)
        if exposure_type is not None:
            degradations.append(exposure_type)
            severity_scores.append(exposure_severity)

        # ---- B/W Detection (only if NOT already classified as washed-out) ----
        # Washed-out images can have near-zero chroma but they are NOT truly B/W
        already_washed = DegradationType.WASHED_OUT in degradations
        if self._is_grayscale(metrics) and not already_washed:
            degradations.append(DegradationType.GRAYSCALE_BW)
            severity_scores.append(self._bw_severity(metrics))

        # ---- Color Cast Detection ----
        if not self._is_grayscale(metrics):
            cast_type, cast_severity = self._detect_color_cast(metrics)
            if cast_type is not None:
                degradations.append(cast_type)
                severity_scores.append(cast_severity)

        # ---- Low Contrast (can co-occur with anything) ----
        if self._is_low_contrast(metrics):
            degradations.append(DegradationType.LOW_CONTRAST)
            severity_scores.append(self._low_contrast_severity(metrics))


        # Build report — sort by severity so the most impactful degradation is primary
        if not degradations:
            return DegradationReport(
                primary=DegradationType.CLEAN,
                secondary=[],
                severity=0.0,
                metrics=metrics
            )

        # Sort by severity (descending) to pick the most severe as primary
        paired = list(zip(degradations, severity_scores))
        paired.sort(key=lambda x: x[1], reverse=True)
        degradations = [p[0] for p in paired]
        severity_scores = [p[1] for p in paired]

        primary = degradations[0]
        secondary = degradations[1:]
        severity = severity_scores[0]

        return DegradationReport(
            primary=primary,
            secondary=secondary,
            severity=min(severity, 1.0),
            metrics=metrics
        )

    def _compute_metrics(self, img_bgr: np.ndarray) -> dict:
        """Compute all statistical metrics needed for classification."""
        img_f = img_bgr.astype(np.float32)

        # BGR channel statistics
        mean_b_bgr, mean_g, mean_r = cv2.mean(img_f)[:3]
        std_b_bgr = np.std(img_f[:, :, 0])
        std_g = np.std(img_f[:, :, 1])
        std_r = np.std(img_f[:, :, 2])

        # Lab statistics
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean_l, mean_a, mean_b = img_lab.mean(axis=(0, 1))
        std_a = img_lab[:, :, 1].std()
        std_b = img_lab[:, :, 2].std()

        # L channel distribution
        l_channel = img_lab[:, :, 0]
        l_p5 = np.percentile(l_channel, 5)
        l_p25 = np.percentile(l_channel, 25)
        l_p75 = np.percentile(l_channel, 75)
        l_p95 = np.percentile(l_channel, 95)
        luma_iqr = l_p75 - l_p25

        # HSV saturation
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mean_sat = hsv[:, :, 1].mean()
        std_sat = hsv[:, :, 1].std()

        # Edge complexity (Laplacian)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Overall standard deviation (for flat/constant detection)
        overall_std = np.std(img_f)

        return {
            # BGR
            "mean_r": float(mean_r),
            "mean_g": float(mean_g),
            "mean_b_bgr": float(mean_b_bgr),
            "std_r": float(std_r),
            "std_g": float(std_g),
            "std_b_bgr": float(std_b_bgr),
            # Lab
            "mean_l": float(mean_l),
            "mean_a": float(mean_a),
            "mean_b": float(mean_b),
            "std_a": float(std_a),
            "std_b": float(std_b),
            # L distribution
            "l_p5": float(l_p5),
            "l_p25": float(l_p25),
            "l_p75": float(l_p75),
            "l_p95": float(l_p95),
            "luma_iqr": float(luma_iqr),
            # HSV
            "mean_sat": float(mean_sat),
            "std_sat": float(std_sat),
            # Texture
            "laplacian_var": float(laplacian_var),
            "overall_std": float(overall_std),
        }

    # ---- Detection Methods ----

    def _is_grayscale(self, m: dict) -> bool:
        """True grayscale: near-zero chroma variance AND low saturation."""
        chroma_zero = m["std_a"] < self.BW_STD_A_THRESHOLD and m["std_b"] < self.BW_STD_B_THRESHOLD
        sat_zero = m["mean_sat"] < self.BW_SATURATION_THRESHOLD

        # Also check BGR channel alignment (all channels nearly identical)
        bgr_aligned = (
            abs(m["mean_r"] - m["mean_g"]) < 5.0 and
            abs(m["mean_b_bgr"] - m["mean_g"]) < 5.0
        )

        return chroma_zero or (sat_zero and bgr_aligned)

    def _bw_severity(self, m: dict) -> float:
        """How "deeply" B/W is this? 1.0 = pure grayscale, lower = slight tint."""
        # Max chroma std divided by a "slightly colored" threshold
        max_chroma_std = max(m["std_a"], m["std_b"])
        return max(0.0, 1.0 - (max_chroma_std / 5.0))

    def _detect_color_cast(self, m: dict) -> tuple:
        """Detect color cast from Lab and BGR statistics. Returns (type, severity) or (None, 0)."""

        casts = []

        # Pink / Magenta: high a* OR BGR red+blue excess over green
        pink_lab = m["mean_a"] > self.PINK_A_THRESHOLD
        pink_bgr = (
            m["mean_r"] > m["mean_g"] + self.PINK_BGR_R_EXCESS and
            m["mean_b_bgr"] > m["mean_g"] + self.PINK_BGR_B_EXCESS
        )
        if pink_lab or pink_bgr:
            severity = min(1.0, (m["mean_a"] - 128.0) / 30.0) if pink_lab else 0.6
            casts.append((DegradationType.PINK_CAST, severity))

        # Green: low a*
        if m["mean_a"] < self.GREEN_A_THRESHOLD:
            severity = min(1.0, (128.0 - m["mean_a"]) / 30.0)
            casts.append((DegradationType.GREEN_CAST, severity))

        # Yellow: high b*
        if m["mean_b"] > self.YELLOW_B_THRESHOLD:
            severity = min(1.0, (m["mean_b"] - 128.0) / 30.0)
            casts.append((DegradationType.YELLOW_CAST, severity))

        # Blue: low b*
        if m["mean_b"] < self.BLUE_B_THRESHOLD:
            severity = min(1.0, (128.0 - m["mean_b"]) / 30.0)
            casts.append((DegradationType.BLUE_CAST, severity))

        if not casts:
            return None, 0.0

        # Return the most severe cast
        casts.sort(key=lambda x: x[1], reverse=True)
        return casts[0]

    def _detect_exposure_issue(self, m: dict) -> tuple:
        """Detect exposure problems. Returns (type, severity) or (None, 0)."""

        # Overexposed: high mean L and high 95th percentile
        if m["mean_l"] > self.OVEREXPOSED_MEAN_L and m["l_p95"] > self.OVEREXPOSED_P95_L:
            severity = min(1.0, (m["mean_l"] - 200.0) / 55.0)
            return DegradationType.OVEREXPOSED, severity

        # Underexposed: very low mean L
        if m["mean_l"] < self.UNDEREXPOSED_MEAN_L:
            severity = min(1.0, (100.0 - m["mean_l"]) / 100.0)
            return DegradationType.UNDEREXPOSED, severity

        # Washed out: compressed dynamic range (small IQR)
        # OR: high brightness + near-zero saturation (desaturated but still has luma spread)
        washed_classic = m["luma_iqr"] < self.WASHED_IQR_THRESHOLD and m["mean_l"] > 120
        washed_desaturated = (
            m["mean_l"] > 160 and
            1.0 < m["mean_sat"] < 10.0 and  # Must have SOME saturation (true B/W has ~0)
            m["std_a"] < 2.0 and m["std_b"] < 2.0
        )
        if washed_classic or washed_desaturated:
            if washed_classic:
                severity = min(1.0, (30.0 - m["luma_iqr"]) / 25.0)
            else:
                severity = min(1.0, (10.0 - m["mean_sat"]) / 8.0)
            return DegradationType.WASHED_OUT, severity

        return None, 0.0

    def _is_low_contrast(self, m: dict) -> bool:
        """Overall image has very low global contrast."""
        return m["overall_std"] < self.LOW_CONTRAST_STD_THRESHOLD

    def _low_contrast_severity(self, m: dict) -> float:
        return max(0.0, (25.0 - m["overall_std"]) / 20.0)

    def format_report(self, report: DegradationReport) -> str:
        """Pretty-print a degradation report for logging."""
        lines = [
            f"[DegradationClassifier] Primary: {report.primary.value} "
            f"(severity: {report.severity:.2f})"
        ]
        if report.secondary:
            sec_names = [d.value for d in report.secondary]
            lines.append(f"[DegradationClassifier] Secondary: {', '.join(sec_names)}")

        key_metrics = {
            k: f"{v:.1f}" for k, v in report.metrics.items()
            if k in ("mean_a", "mean_b", "std_a", "std_b", "mean_sat", "mean_l", "luma_iqr")
        }
        lines.append(f"[DegradationClassifier] Key metrics: {key_metrics}")
        return "\n".join(lines)
