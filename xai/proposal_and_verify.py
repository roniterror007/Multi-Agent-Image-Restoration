# xai/proposal_and_verify.py
"""
Proposal Engine and Quality Gate for the Agent-Critic Architecture.

The SharedQualityGate now accepts a DegradationReport to make scoring
context-aware — it won't penalize legitimate color restoration for B/W images
or color cast removal.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from enum import Enum
from skimage.metrics import structural_similarity as ssim

# Import degradation types for context-aware scoring
from core.degradation_classifier import DegradationType, DegradationReport


class ProposalSource(Enum):
    PHASH_MEMORY = "phash_memory"
    VLM_REASONING = "vlm_reasoning"
    SPATIAL_FIELDS = "spatial_fields"
    COLOR_MLP = "color_mlp"
    PURE_MATH = "pure_math"
    REFERENCE_PRIOR = "reference_prior"
    EXEMPLAR_TRANSFER = "exemplar_transfer"
    DIRECT_REPLACEMENT = "direct_replacement"


@dataclass
class CorrectionProposal:
    source_name: str
    global_shifts: Optional[tuple] = None
    spatial_fields: Optional[np.ndarray] = None
    spatial_lab_image: Optional[np.ndarray] = None
    confidence: float = 0.0
    reasoning: str = ""
    ref_similarity: float = 0.0  # DINOv2 match score, if applicable


class SharedQualityGate:
    """
    Degradation-aware quality gate that scores proposals differently
    based on what's actually wrong with the image.

    Key changes from the original:
    - B/W→color: TV penalty disabled (adding color inherently increases TV)
    - Pink cast removal: rewards movement toward neutral a*≈128
    - High DINOv2 similarity: exemplar/direct proposals get a confidence boost
    - Structural preservation scored on L-channel only (not full Lab)
    """

    def __init__(self, target_srgb_bounds=(0.0, 1.0)):
        self.lower_bound, self.upper_bound = target_srgb_bounds

    def score_proposal(
        self,
        original_lab: np.ndarray,
        proposal: CorrectionProposal,
        degradation: Optional[DegradationReport] = None,
    ) -> float:
        """
        Scores a proposal on a [0, 1] scale.
        """
        # Manual user feedback is the absolute ground truth
        if proposal.source_name == ProposalSource.PHASH_MEMORY or proposal.source_name == "phash_memory":
            return 1.0

        import cv2

        # 1. Simulate the edit
        if proposal.spatial_lab_image is not None:
            test_lab = proposal.spatial_lab_image.copy()
            if test_lab.shape[:2] != original_lab.shape[:2]:
                test_lab = cv2.resize(test_lab, (original_lab.shape[1], original_lab.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
        else:
            test_lab = original_lab.copy()
            if proposal.global_shifts:
                dl, da, db = proposal.global_shifts
                test_lab[:, :, 0] += dl
                test_lab[:, :, 1] += da
                test_lab[:, :, 2] += db

        # 2. Gamut Safety Check (unchanged — always important)
        test_bgr = cv2.cvtColor(test_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        test_rgb_norm = test_bgr[..., ::-1].astype(np.float32) / 255.0

        clipping_violations = np.sum(
            (test_rgb_norm < self.lower_bound) | (test_rgb_norm > self.upper_bound)
        )
        total_pixels = test_rgb_norm.shape[0] * test_rgb_norm.shape[1] * 3
        clipping_ratio = clipping_violations / total_pixels

        gamut_score = max(0.0, 1.0 - (clipping_ratio * 10))

        # 3. Structural Preservation (L-channel SSIM only)
        ssim_score = 1.0
        if proposal.spatial_lab_image is not None or getattr(proposal, 'spatial_fields', None) is not None:
            orig_l = original_lab[:, :, 0].astype(np.float64)
            test_l = test_lab[:, :, 0].astype(np.float64)
            if orig_l.shape == test_l.shape:
                ssim_val = ssim(orig_l, test_l, data_range=255.0)
                ssim_score = max(0.0, ssim_val)
                # Only heavy penalty if L-channel structure is truly destroyed
                if ssim_val < 0.80:
                    ssim_score *= 0.3

        # 3.5 Chromatic Total Variation Penalty (Rainbow Exploit Fix)
        # Prevents injecting random high-frequency color noise.
        chroma_tv_score = 1.0
        if proposal.spatial_lab_image is not None or getattr(proposal, 'spatial_fields', None) is not None:
            orig_a, orig_b = original_lab[:,:,1].astype(np.float32), original_lab[:,:,2].astype(np.float32)
            test_a, test_b = test_lab[:,:,1].astype(np.float32), test_lab[:,:,2].astype(np.float32)
            
            # Simple Total Variation estimate
            tv_orig = np.mean(np.abs(orig_a[1:,:] - orig_a[:-1,:])) + np.mean(np.abs(orig_a[:,1:] - orig_a[:,:-1])) + \
                      np.mean(np.abs(orig_b[1:,:] - orig_b[:-1,:])) + np.mean(np.abs(orig_b[:,1:] - orig_b[:,:-1]))
            tv_test = np.mean(np.abs(test_a[1:,:] - test_a[:-1,:])) + np.mean(np.abs(test_a[:,1:] - test_a[:,:-1])) + \
                      np.mean(np.abs(test_b[1:,:] - test_b[:-1,:])) + np.mean(np.abs(test_b[:,1:] - test_b[:,:-1]))
            
            # If test TV explodes compared to original (and isn't just flat color being added), heavily penalize
            if tv_test > tv_orig * 2.0 and tv_test > 5.0:
                chroma_tv_score = 0.2 # Massive penalty for rainbow noise

        # 4. Degradation Correction Reward (NEW — context-aware)
        correction_reward = self._compute_correction_reward(
            original_lab, test_lab, degradation
        )

        # 5. Confidence boost for high-similarity exemplar matches
        confidence = proposal.confidence
        if proposal.ref_similarity > 0.85 and proposal.source_name in (
            ProposalSource.EXEMPLAR_TRANSFER,
            ProposalSource.DIRECT_REPLACEMENT,
        ):
            confidence = min(1.0, confidence * 1.15)

        # Combine scores with degradation-aware weights
        if degradation and degradation.is_bw:
            # B/W images: reward color restoration more, don't care about TV/smoothness
            final_score = (
                (confidence * 0.15) +
                (gamut_score * 0.20) +
                (ssim_score * 0.20) +
                (chroma_tv_score * 0.10) +
                (correction_reward * 0.35)
            )
        elif degradation and degradation.has_color_cast:
            # Color cast: reward neutralization heavily
            final_score = (
                (confidence * 0.20) +
                (gamut_score * 0.25) +
                (ssim_score * 0.15) +
                (chroma_tv_score * 0.10) +
                (correction_reward * 0.30)
            )
        else:
            # General case: balanced scoring
            final_score = (
                (confidence * 0.20) +
                (gamut_score * 0.25) +
                (ssim_score * 0.20) +
                (chroma_tv_score * 0.10) +
                (correction_reward * 0.25)
            )

        return final_score

    def _compute_correction_reward(
        self,
        original_lab: np.ndarray,
        corrected_lab: np.ndarray,
        degradation: Optional[DegradationReport],
    ) -> float:
        """
        Score how well a proposal corrects the detected degradation.

        For metallic/industrial parts, "correct" generally means:
        - a* ≈ 128 (neutral, no pink/green cast)
        - b* ≈ 128 (neutral, no yellow/blue cast)
        - Reasonable L distribution (not blown out or crushed)
        """
        if degradation is None:
            return 0.5  # No degradation info → neutral score

        # Ignore background pixels (L > 240 assumed white background)
        fg_mask = original_lab[:, :, 0] < 240
        if fg_mask.sum() < 100:
            return 0.5

        # Corrected foreground statistics
        corrected_fg_a = corrected_lab[:, :, 1][fg_mask].mean()
        corrected_fg_b = corrected_lab[:, :, 2][fg_mask].mean()

        # Original foreground statistics
        original_fg_a = original_lab[:, :, 1][fg_mask].mean()
        original_fg_b = original_lab[:, :, 2][fg_mask].mean()

        reward = 0.5  # Start at neutral

        if degradation.is_bw:
            # For B/W images, we WANT color to be added.
            # Reward proposals that add meaningful chromaticity
            corrected_std_a = corrected_lab[:, :, 1][fg_mask].std()
            corrected_std_b = corrected_lab[:, :, 2][fg_mask].std()
            original_std_a = original_lab[:, :, 1][fg_mask].std()

            # Did we add meaningful color variance? (not just flat tint)
            color_added = max(corrected_std_a, corrected_std_b) - max(original_std_a, 0.1)
            if color_added > 2.0:
                reward += 0.3  # Significant color restoration
            elif color_added > 0.5:
                reward += 0.15  # Some color restoration

            # Is the added color reasonable (not extreme)?
            if 100 < corrected_fg_a < 156 and 100 < corrected_fg_b < 156:
                reward += 0.2  # Within reasonable metallic color range

        elif degradation.has_color_cast:
            # For color casts, reward movement toward neutral
            original_deviation = abs(original_fg_a - 128.0) + abs(original_fg_b - 128.0)
            corrected_deviation = abs(corrected_fg_a - 128.0) + abs(corrected_fg_b - 128.0)

            if corrected_deviation < original_deviation:
                improvement_ratio = (original_deviation - corrected_deviation) / max(original_deviation, 1.0)
                reward += improvement_ratio * 0.5
            else:
                # Made the cast WORSE
                reward -= 0.2

        elif degradation.has_exposure_issue:
            # For exposure issues, check L-channel improvement
            corrected_fg_l = corrected_lab[:, :, 0][fg_mask].mean()
            original_fg_l = original_lab[:, :, 0][fg_mask].mean()

            # Target: mean L around 130-160 for well-exposed industrial parts
            target_l = 145.0
            original_l_error = abs(original_fg_l - target_l)
            corrected_l_error = abs(corrected_fg_l - target_l)

            if corrected_l_error < original_l_error:
                reward += 0.3
            else:
                reward -= 0.1

        return min(1.0, max(0.0, reward))


class ProposalEngine:
    def __init__(self):
        self.gate = SharedQualityGate()

    def select_best(
        self,
        original_lab: np.ndarray,
        proposals: list[CorrectionProposal],
        degradation: Optional[DegradationReport] = None,
    ) -> CorrectionProposal:
        """Select the best proposal using degradation-aware scoring."""
        best_score = -1.0
        best_proposal = proposals[0]

        for prop in proposals:
            score = self.gate.score_proposal(original_lab, prop, degradation)
            source_name = prop.source_name.value if hasattr(prop.source_name, 'value') else str(prop.source_name)
            print(f"  [Gate Score] {source_name}: {score:.3f} (conf={prop.confidence:.2f}, ref_sim={prop.ref_similarity:.2f})")
            if score > best_score:
                best_score = score
                best_proposal = prop

        return best_proposal
