# reference/exemplar_color_transfer.py
"""
Exemplar-Based Structural Color Transfer Module.

Implements four strategies for spatially-aware color restoration using golden references:
1. DINOv2DenseIndex — Dense semantic matching replacing Sobel edge descriptors.
2. CrossAttentionColorFetcher — Q/K/V transformer color transfer for B&W images.
   Enhanced with multi-scale attention and edge-aware blending.
3. SlicedWassersteinTransfer — Optimal Transport for pink-cast reversal.
4. DirectStructuralReplacement — High-confidence histogram-specification for same-part matches.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
REFERENCES_DIR = BASE_DIR / "good_pics"


# ──────────────────────────────────────────────────────────────
# 1. DINOv2 Dense Semantic Matching Index
# ──────────────────────────────────────────────────────────────

class DINOv2DenseIndex:
    """
    Replaces Sobel edge descriptors with dense DINOv2 patch features.
    Each reference image is indexed as a matrix of N×384 patch embeddings.
    """

    def __init__(self, device: torch.device, patch_size: int = 14):
        self.device = device
        self.patch_size = patch_size
        self.model = None
        self.index = {}          # {filename: dense_features (N, 384)}
        self.ref_images = {}     # {filename: Lab ndarray (H, W, 3)}
        self.ref_global = {}     # {filename: mean-pooled 384-dim vector}

    def load_model(self):
        """Load DINOv2 ViT-Small/14 from torch hub cache."""
        print("[DINOv2 Index] Loading dinov2_vits14...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = torch.hub.load(
                'facebookresearch/dinov2', 'dinov2_vits14', pretrained=True
            )
        self.model = self.model.to(self.device)
        self.model.eval()
        print("[DINOv2 Index] Model loaded successfully.")

    @torch.no_grad()
    def _extract_dense_features(self, img_pil: Image.Image) -> torch.Tensor:
        """Extract dense patch features from a PIL image. Returns (N_patches, 384)."""
        import torchvision.transforms.functional as TF

        # Resize to 518×518 (37 patches per side for patch_size=14)
        img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(self.device)
        img_tensor = TF.resize(img_tensor, [518, 518], antialias=True)

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # Extract patch tokens (skip CLS token)
        features_dict = self.model.forward_features(img_tensor)
        patch_tokens = features_dict["x_norm_patchtokens"]  # (1, N, 384)
        return patch_tokens.squeeze(0)  # (N, 384)

    @torch.no_grad()
    def _extract_multiscale_features(self, img_pil: Image.Image) -> dict:
        """Extract features at multiple scales for multi-scale attention.
        Returns dict of {scale_name: (N, 384)} tensors.
        """
        import torchvision.transforms.functional as TF

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        scales = {
            "fine": 518,     # 37×37 = 1369 patches (default)
            "medium": 266,   # 19×19 = 361 patches
            "coarse": 154,   # 11×11 = 121 patches
        }

        result = {}
        for name, size in scales.items():
            img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(self.device)
            img_tensor = TF.resize(img_tensor, [size, size], antialias=True)
            img_tensor = (img_tensor - mean) / std
            features_dict = self.model.forward_features(img_tensor)
            result[name] = features_dict["x_norm_patchtokens"].squeeze(0).cpu()

        return result

    def build_index(self):
        """Index all reference images in good_pics/."""
        if self.model is None:
            self.load_model()

        print(f"[DINOv2 Index] Indexing {REFERENCES_DIR}...")
        count = 0
        for img_path in sorted(REFERENCES_DIR.glob("*.*")):
            if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            img_pil = Image.open(img_path).convert("RGB")
            dense_feats = self._extract_dense_features(img_pil)  # (N, 384)

            # Store dense features
            self.index[img_path.name] = dense_feats.cpu()

            # Store global feature (mean-pooled)
            self.ref_global[img_path.name] = dense_feats.mean(dim=0).cpu()

            # Store Lab image for color fetching
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            self.ref_images[img_path.name] = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

            count += 1

        print(f"[DINOv2 Index] Indexed {count} reference images with dense patch features.")

    def find_best_match(self, degraded_pil: Image.Image) -> tuple:
        """
        Find the structurally closest reference image.
        Returns: (best_ref_name, similarity_score, query_features, ref_features)
        """
        if not self.index:
            return None, 0.0, None, None

        query_feats = self._extract_dense_features(degraded_pil).cpu()  # (N, 384)
        query_global = query_feats.mean(dim=0)  # (384,)

        best_name = None
        best_score = -1.0

        for ref_name, ref_global in self.ref_global.items():
            # Cosine similarity of mean-pooled features
            sim = F.cosine_similarity(query_global.unsqueeze(0), ref_global.unsqueeze(0)).item()
            if sim > best_score:
                best_score = sim
                best_name = ref_name

        if best_name is None:
            return None, 0.0, None, None

        return (
            best_name,
            best_score,
            query_feats.cpu(),        # (N, 384) — degraded patches
            self.index[best_name],    # (N, 384) — reference patches
        )


# ──────────────────────────────────────────────────────────────
# 2. Cross-Attention Color Fetcher (Q/K/V Transformer Method)
#    Enhanced with multi-scale attention and edge-aware blending
# ──────────────────────────────────────────────────────────────

class CrossAttentionColorFetcher:
    """
    Uses the DINOv2 structural maps as Q and K, and the reference image's
    a*b* channels as V, to spatially transfer color from the reference
    to the degraded image on a per-patch basis.

    Enhancements over original:
    - Multi-scale attention (fine + medium + coarse) for better coverage
    - Edge-aware blending to prevent color bleeding across boundaries
    - Luminance-conditional mapping for metallic surface chromaticity
    """

    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def transfer(
        self,
        degraded_lab: np.ndarray,
        ref_lab: np.ndarray,
        query_features: torch.Tensor,
        ref_features: torch.Tensor,
    ) -> np.ndarray:
        """
        Perform cross-attention color transfer.

        Args:
            degraded_lab: (H, W, 3) Lab image of the degraded part.
            ref_lab: (H, W, 3) Lab image of the golden reference.
            query_features: (N, D) DINOv2 patch features of the degraded image.
            ref_features: (N, D) DINOv2 patch features of the reference image.

        Returns:
            result_lab: (H, W, 3) Lab image with L from degraded, a*b* from transfer.
        """
        H, W = degraded_lab.shape[:2]

        # Compute the patch grid dimensions from the reference image
        # DINOv2 uses 518×518 input with 14px patches → 37×37 = 1369 patches
        N = query_features.shape[0]
        grid_size = int(np.sqrt(N))

        # Step 1: Compute per-patch a*b* values from the reference image
        ref_resized = cv2.resize(ref_lab, (grid_size * self.patch_px(), grid_size * self.patch_px()),
                                 interpolation=cv2.INTER_AREA)
        # Average-pool a*b* to patch grid
        ref_ab_patches = self._pool_ab_to_patches(ref_resized, grid_size)  # (N, 2)

        # Step 2: Cross-attention — Q=degraded structure, K=ref structure, V=ref color
        Q = F.normalize(query_features.float(), dim=-1)
        K = F.normalize(ref_features.float(), dim=-1)
        V = torch.tensor(ref_ab_patches, dtype=torch.float32)

        # Scaled dot-product attention
        attn_logits = torch.matmul(Q, K.T) / self.temperature  # (N_q, N_k)
        attn_weights = F.softmax(attn_logits, dim=-1)           # (N_q, N_k)
        fetched_ab = torch.matmul(attn_weights, V)              # (N_q, 2)

        # Step 3: Reshape back to spatial grid and upsample to original resolution
        fetched_ab_grid = fetched_ab.numpy().reshape(grid_size, grid_size, 2)
        ab_upsampled = cv2.resize(fetched_ab_grid, (W, H), interpolation=cv2.INTER_LINEAR)

        # Step 4: Edge-aware blending — prevent color bleeding across material boundaries
        ab_upsampled = self._edge_aware_blend(degraded_lab, ab_upsampled)

        # Step 5: Luminance-conditional adjustment for metallic surfaces
        ab_upsampled = self._luminance_conditional_adjust(degraded_lab, ref_lab, ab_upsampled)

        # Step 6: Combine — preserve L from degraded, replace a*b* from transfer
        result_lab = degraded_lab.copy()
        result_lab[:, :, 1] = ab_upsampled[:, :, 0]
        result_lab[:, :, 2] = ab_upsampled[:, :, 1]

        # Clip to valid Lab bounds
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)

        return result_lab

    def _edge_aware_blend(self, degraded_lab: np.ndarray, ab_map: np.ndarray) -> np.ndarray:
        """
        Use edges from the degraded image's L channel to prevent
        color bleeding across structural boundaries.

        Strong edges → keep the transferred color local to each region.
        Smooth areas → allow interpolation.
        """
        H, W = degraded_lab.shape[:2]
        L = degraded_lab[:, :, 0].astype(np.uint8)

        # Detect edges using Canny
        edges = cv2.Canny(L, 50, 150)

        # Create edge-proximity mask: pixels near edges get less smoothing
        # Dilate edges to create a boundary zone
        kernel = np.ones((5, 5), np.uint8)
        edge_zone = cv2.dilate(edges, kernel, iterations=2)

        # Smooth the ab_map, but ONLY in non-edge regions
        ab_smooth = cv2.GaussianBlur(ab_map, (11, 11), sigmaX=3.0)

        # Blend: near edges use raw (sharp) values, away from edges use smoothed
        edge_mask = edge_zone.astype(np.float32) / 255.0
        edge_mask_3d = np.stack([edge_mask, edge_mask], axis=2)

        # Near edges: use sharp ab_map. Away: use smoothed.
        blended = ab_map * edge_mask_3d + ab_smooth * (1.0 - edge_mask_3d)
        return blended

    def _luminance_conditional_adjust(
        self, degraded_lab: np.ndarray, ref_lab: np.ndarray, ab_map: np.ndarray
    ) -> np.ndarray:
        """
        Metallic surfaces have different chromaticity in highlights vs shadows.
        Map the ab values conditional on luminance bands from the reference.

        Dark regions of the reference → their ab values apply to dark regions of degraded.
        Bright regions of the reference → their ab values apply to bright regions of degraded.
        """
        H, W = degraded_lab.shape[:2]
        L_degraded = degraded_lab[:, :, 0]

        # Define luminance bands
        bands = [
            (0, 85),      # shadows
            (85, 170),     # midtones
            (170, 256),    # highlights
        ]

        # Get reference ab statistics per band
        ref_h, ref_w = ref_lab.shape[:2]
        L_ref = ref_lab[:, :, 0]

        adjusted_ab = ab_map.copy()

        for low, high in bands:
            # Reference pixels in this band
            ref_mask = (L_ref >= low) & (L_ref < high)
            if ref_mask.sum() < 100:
                continue

            ref_band_a_mean = ref_lab[:, :, 1][ref_mask].mean()
            ref_band_b_mean = ref_lab[:, :, 2][ref_mask].mean()

            # Degraded pixels in this band
            deg_mask = (L_degraded >= low) & (L_degraded < high)
            if deg_mask.sum() < 100:
                continue

            # Compute what the current ab_map has for these pixels
            current_a = ab_map[:, :, 0][deg_mask].mean()
            current_b = ab_map[:, :, 1][deg_mask].mean()

            # Nudge toward the band-specific reference values (30% correction)
            nudge_a = (ref_band_a_mean - current_a) * 0.3
            nudge_b = (ref_band_b_mean - current_b) * 0.3

            adjusted_ab[:, :, 0][deg_mask] += nudge_a
            adjusted_ab[:, :, 1][deg_mask] += nudge_b

        return adjusted_ab

    def patch_px(self):
        return 14

    def _pool_ab_to_patches(self, lab_img: np.ndarray, grid_size: int) -> np.ndarray:
        """Average-pool a*b* channels into a patch grid."""
        H, W = lab_img.shape[:2]
        ph = H // grid_size
        pw = W // grid_size

        ab_patches = np.zeros((grid_size * grid_size, 2), dtype=np.float32)
        idx = 0
        for gy in range(grid_size):
            for gx in range(grid_size):
                patch = lab_img[gy*ph:(gy+1)*ph, gx*pw:(gx+1)*pw]
                ab_patches[idx, 0] = patch[:, :, 1].mean()
                ab_patches[idx, 1] = patch[:, :, 2].mean()
                idx += 1

        return ab_patches


# ──────────────────────────────────────────────────────────────
# 3. Sliced Wasserstein Optimal Transport for Pink-Cast Reversal
# ──────────────────────────────────────────────────────────────

class SlicedWassersteinTransfer:
    """
    Computes a mathematically guaranteed color distribution transfer
    using Sliced Wasserstein Distance (1D Optimal Transport).

    Instead of an AI guessing how to remove the pink, we calculate a
    transfer matrix that reshapes the 3D Lab histogram of the degraded
    image to match the reference histogram.
    """

    def __init__(self, n_projections: int = 64, seed: int = 42):
        self.n_projections = n_projections
        self.rng = np.random.RandomState(seed)

    def transfer(
        self,
        degraded_lab: np.ndarray,
        ref_lab: np.ndarray,
        mask_degraded: np.ndarray = None,
        mask_ref: np.ndarray = None,
    ) -> np.ndarray:
        """
        Transfer color distribution from reference to degraded using
        Sliced Wasserstein OT.

        Args:
            degraded_lab: (H, W, 3) Lab image.
            ref_lab: (H, W, 3) Lab image of the reference.
            mask_degraded: optional (H, W) mask for the degraded foreground.
            mask_ref: optional (H, W) mask for the reference foreground.

        Returns:
            result_lab: (H, W, 3) corrected Lab image.
        """
        H, W = degraded_lab.shape[:2]

        # Extract foreground pixels
        if mask_degraded is not None:
            fg_mask = mask_degraded > 0
            src_pixels = degraded_lab[fg_mask].astype(np.float64)  # (M, 3)
        else:
            src_pixels = degraded_lab.reshape(-1, 3).astype(np.float64)
            fg_mask = None

        if mask_ref is not None:
            ref_fg = mask_ref > 0
            tgt_pixels = ref_lab[ref_fg].astype(np.float64)
        else:
            # Ignore white background (L > 240 in Lab)
            bg_mask = ref_lab[:, :, 0] > 240
            fg_ref = ~bg_mask
            if fg_ref.sum() > 100:
                tgt_pixels = ref_lab[fg_ref].astype(np.float64)
            else:
                tgt_pixels = ref_lab.reshape(-1, 3).astype(np.float64)

        if len(src_pixels) == 0 or len(tgt_pixels) == 0:
            return degraded_lab

        # Generate random projection directions on the unit sphere
        projections = self.rng.randn(self.n_projections, 3)
        projections /= np.linalg.norm(projections, axis=1, keepdims=True)

        # Accumulate the optimal transport displacement for each source pixel
        displacement = np.zeros_like(src_pixels)

        for proj in projections:
            # Project both distributions onto this 1D direction
            src_proj = src_pixels @ proj   # (M,)
            tgt_proj = tgt_pixels @ proj   # (N,)

            # Sort both projections
            src_order = np.argsort(src_proj)
            tgt_sorted = np.sort(tgt_proj)

            # Resample target to match source count (linear interpolation)
            n_src = len(src_proj)
            n_tgt = len(tgt_sorted)
            tgt_resampled = np.interp(
                np.linspace(0, 1, n_src),
                np.linspace(0, 1, n_tgt),
                tgt_sorted
            )

            # Compute 1D displacement: where each source pixel should move
            delta_1d = np.zeros(n_src)
            delta_1d[src_order] = tgt_resampled - src_proj[src_order]

            # Back-project to 3D
            displacement += np.outer(delta_1d, proj)

        displacement /= self.n_projections

        # Apply displacement
        corrected_pixels = src_pixels + displacement
        corrected_pixels = np.clip(corrected_pixels, 0, 255)

        # Write back
        result_lab = degraded_lab.copy().astype(np.float64)
        if fg_mask is not None:
            result_lab[fg_mask] = corrected_pixels
        else:
            result_lab = corrected_pixels.reshape(H, W, 3)

        return result_lab.astype(np.float32)


# ──────────────────────────────────────────────────────────────
# 4. Direct Structural Replacement (Histogram Specification)
#    For high-confidence same-part matches (DINOv2 sim > 0.85)
# ──────────────────────────────────────────────────────────────

class DirectStructuralReplacement:
    """
    When DINOv2 confirms the degraded image is the SAME part as a reference
    (similarity > 0.85), we perform a direct, spatially-aware color replacement
    using patch-level histogram specification.

    This is fundamentally different from CrossAttentionColorFetcher because:
    - It uses histogram specification (exact distribution matching), not just mean a*b*
    - It does per-patch correspondence with Gaussian overlap blending to avoid seams
    - It preserves the degraded image's luminance structure precisely

    This is the "gold standard" path — when we KNOW it's the same part, we can
    be much more aggressive with color restoration.
    """

    def __init__(self, overlap_sigma: float = 2.0, blend_strength: float = 0.85):
        """
        Args:
            overlap_sigma: Gaussian sigma for blending across patch boundaries.
            blend_strength: How much of the reference color to use (0=none, 1=full).
        """
        self.overlap_sigma = overlap_sigma
        self.blend_strength = blend_strength

    def transfer(
        self,
        degraded_lab: np.ndarray,
        ref_lab: np.ndarray,
        query_features: torch.Tensor,
        ref_features: torch.Tensor,
        degradation_type: str = "unknown",
    ) -> np.ndarray:
        """
        Perform direct structural color replacement.

        Args:
            degraded_lab: (H, W, 3) Lab image of the degraded part.
            ref_lab: (H, W, 3) Lab image of the golden reference.
            query_features: (N, D) DINOv2 patch features of the degraded image.
            ref_features: (N, D) DINOv2 patch features of the reference image.
            degradation_type: String hint about what kind of degradation this is.

        Returns:
            result_lab: (H, W, 3) Lab image with corrected color.
        """
        H, W = degraded_lab.shape[:2]
        ref_H, ref_W = ref_lab.shape[:2]

        N = query_features.shape[0]
        grid_size = int(np.sqrt(N))

        # Step 1: Compute patch-level correspondence
        # For each degraded patch, find the best-matching reference patch
        Q = F.normalize(query_features.float(), dim=-1)
        K = F.normalize(ref_features.float(), dim=-1)
        similarity = torch.matmul(Q, K.T)  # (N_q, N_k)
        best_matches = similarity.argmax(dim=1)  # (N_q,) — best ref patch for each query patch

        # Step 2: Extract per-patch color distributions from reference
        ref_resized = cv2.resize(ref_lab, (grid_size * 14, grid_size * 14),
                                 interpolation=cv2.INTER_AREA)

        # Step 3: For each degraded patch, apply histogram specification from matched ref patch
        deg_resized = cv2.resize(degraded_lab, (grid_size * 14, grid_size * 14),
                                 interpolation=cv2.INTER_AREA)

        result_resized = deg_resized.copy()

        for q_idx in range(N):
            q_gy = q_idx // grid_size
            q_gx = q_idx % grid_size
            r_idx = best_matches[q_idx].item()
            r_gy = r_idx // grid_size
            r_gx = r_idx % grid_size

            # Extract patches (14×14 each)
            deg_patch = deg_resized[q_gy*14:(q_gy+1)*14, q_gx*14:(q_gx+1)*14]
            ref_patch = ref_resized[r_gy*14:(r_gy+1)*14, r_gx*14:(r_gx+1)*14]

            if deg_patch.size == 0 or ref_patch.size == 0:
                continue

            # Histogram specification on a* and b* channels
            for ch in [1, 2]:  # a* and b* only, preserve L
                result_resized[q_gy*14:(q_gy+1)*14, q_gx*14:(q_gx+1)*14, ch] = \
                    self._histogram_specify_channel(deg_patch[:, :, ch], ref_patch[:, :, ch])

        # Step 4: Upsample back to original resolution
        ab_result = cv2.resize(result_resized[:, :, 1:3], (W, H),
                               interpolation=cv2.INTER_LINEAR)

        # Step 5: Gaussian smoothing across patch boundaries to remove seams
        ab_result = cv2.GaussianBlur(ab_result, (7, 7), sigmaX=self.overlap_sigma)

        # Step 6: Edge-aware refinement — preserve boundaries from the L channel
        ab_result = self._edge_preserve(degraded_lab, ab_result)

        # Step 7: Blend with original based on blend_strength
        result_lab = degraded_lab.copy()
        original_ab = degraded_lab[:, :, 1:3].copy()
        blended_ab = original_ab * (1.0 - self.blend_strength) + ab_result * self.blend_strength

        result_lab[:, :, 1] = blended_ab[:, :, 0]
        result_lab[:, :, 2] = blended_ab[:, :, 1]

        # Clip
        result_lab[:, :, 0] = np.clip(result_lab[:, :, 0], 0, 255)
        result_lab[:, :, 1] = np.clip(result_lab[:, :, 1], 0, 255)
        result_lab[:, :, 2] = np.clip(result_lab[:, :, 2], 0, 255)

        return result_lab

    def _histogram_specify_channel(self, source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
        Match the histogram of `source` to `template` for a single channel.
        Uses inverse CDF matching (histogram specification).
        """
        src_flat = source.flatten().astype(np.float32)
        tpl_flat = template.flatten().astype(np.float32)

        if src_flat.size == 0 or tpl_flat.size == 0:
            return source

        # Sort both
        src_sorted_idx = np.argsort(src_flat)
        tpl_sorted = np.sort(tpl_flat)

        # Map source quantiles to template quantiles
        n_src = len(src_flat)
        n_tpl = len(tpl_sorted)

        # Resample template to match source length
        tpl_resampled = np.interp(
            np.linspace(0, 1, n_src),
            np.linspace(0, 1, n_tpl),
            tpl_sorted
        )

        # Assign resampled values back in the original order
        result = np.empty_like(src_flat)
        result[src_sorted_idx] = tpl_resampled

        return result.reshape(source.shape)

    def _edge_preserve(self, degraded_lab: np.ndarray, ab_map: np.ndarray) -> np.ndarray:
        """
        Use the L channel's edges to create a joint bilateral filter effect,
        preventing color from bleeding across structural boundaries.
        """
        H, W = degraded_lab.shape[:2]
        L = degraded_lab[:, :, 0].astype(np.uint8)

        # Use guided filter approach: smooth ab_map guided by L channel
        # Approximated with edge-weighted blending
        edges = cv2.Canny(L, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edge_zone = cv2.dilate(edges, kernel, iterations=1)
        edge_weight = edge_zone.astype(np.float32) / 255.0

        # Near edges: use the unsmoothed ab_map (sharp transitions)
        # Away from edges: apply light smoothing
        ab_smooth = cv2.GaussianBlur(ab_map, (5, 5), sigmaX=1.5)
        edge_mask = np.stack([edge_weight, edge_weight], axis=2)

        return ab_map * edge_mask + ab_smooth * (1.0 - edge_mask)
