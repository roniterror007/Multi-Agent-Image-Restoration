import os
import argparse
import pickle
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import random

# Implemented Architecture Modules
from xai.birefnet_tensor_isolation import IsolatedBiRefNetProcessor
from xai.proposal_and_verify import ProposalEngine, CorrectionProposal, ProposalSource
from xai.spatial_correction_fields import SpatialCorrectionFieldGenerator
from xai.trust_region_line_search import TrustRegionLineSearch
from core.reviewer_agent import find_approved_fix
from core.feedback_store import save_feedback, compute_perceptual_hash
from core.agent_controller import validate_adjust_lab_call, ProportionalController
from core.degradation_classifier import DegradationClassifier, DegradationType, DegradationReport
from training.train_color_model import ColorMLP, extract_features_gpu
import torchvision.transforms.functional as F_vision
from core.vlm_reasoner import VLMReasoner
import math
from reference.reference_retrieval import build_index, compute_sobel_descriptor
from reference.exemplar_color_transfer import (
    DINOv2DenseIndex, CrossAttentionColorFetcher,
    SlicedWassersteinTransfer, DirectStructuralReplacement
)

BASE_DIR = Path(__file__).resolve().parent

def create_abb_canvas(img_bgr, mask_np):
    canvas = np.ones((1440, 1920, 3), dtype=np.uint8) * 255
    
    foreground = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_np)
    coords = cv2.findNonZero(mask_np)
    if coords is None:
        # Fallback: if mask is empty, place the entire image on canvas
        h, w = img_bgr.shape[:2]
        scale_w = (1920 * 0.75) / w
        scale_h = (1440 * 0.75) / h
        scale = min(scale_w, scale_h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        start_x = (1920 - new_w) // 2
        start_y = (1440 - new_h) // 2
        canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return canvas
        
    x, y, w, h = cv2.boundingRect(coords)
    cropped_fg = foreground[y:y+h, x:x+w]
    cropped_mask = mask_np[y:y+h, x:x+w]
    
    # Revert to prevent blurry upscaling of small images
    scale_w = (1920 * 0.75) / w
    scale_h = (1440 * 0.75) / h
    scale = min(scale_w, scale_h, 1.0)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_fg = cv2.resize(cropped_fg, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    
    start_x = (1920 - new_w) // 2
    start_y = (1440 - new_h) // 2
    
    roi = canvas[start_y:start_y+new_h, start_x:start_x+new_w]
    mask_inv = cv2.bitwise_not(resized_mask)
    
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = cv2.add(bg, resized_fg)
    
    return canvas


def classifier_restore(img_bgr, report: DegradationReport):
    """
    Targeted pre-restoration based on DegradationClassifier results.
    Replaces the old hardcoded smart_restore() heuristic.

    This applies ONLY basic, deterministic fixes for exposure issues.
    Color cast correction is handled by the full agent pipeline.
    """
    if not report.is_degraded:
        return img_bgr

    img_f = img_bgr.astype(np.float32)

    # Check for flat/constant image (degenerate case)
    if report.metrics.get("overall_std", 999) < 2.0:
        return img_bgr

    # Fix exposure issues (these are safe, deterministic operations)
    for deg_type in report.all_degradations:
        if deg_type == DegradationType.OVEREXPOSED:
            # Reduce brightness and expand contrast
            img_f = np.clip((img_f - 50) / 1.5, 0, 255)
            print("[ClassifierRestore] Applied overexposure correction: brightness -50, contrast /1.5")

        elif deg_type == DegradationType.UNDEREXPOSED:
            # Boost brightness
            target_l = 175.0
            mean_l = report.metrics["mean_l"]
            boost = (target_l - mean_l) * 0.6
            img_lab = cv2.cvtColor(img_f.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
            img_lab[:, :, 0] = np.clip(img_lab[:, :, 0] + boost, 0, 255)
            img_f = cv2.cvtColor(img_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
            print(f"[ClassifierRestore] Applied underexposure correction: L boost +{boost:.1f}")

        elif deg_type == DegradationType.WASHED_OUT:
            # Expand dynamic range (contrast stretch)
            img_f = np.clip((img_f - 100) * 2.0, 0, 255)
            print("[ClassifierRestore] Applied washed-out correction: contrast stretch ×2")

        elif deg_type == DegradationType.LOW_CONTRAST:
            # CLAHE on L-channel
            img_lab = cv2.cvtColor(img_f.astype(np.uint8), cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_lab[:, :, 0] = clahe.apply(img_lab[:, :, 0])
            img_f = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR).astype(np.float32)
            print("[ClassifierRestore] Applied CLAHE contrast enhancement")

    # NOTE: Color casts (pink, green, yellow, blue) and B/W are NOT fixed here.
    # They are handled by the full agent pipeline (DINOv2 → exemplar transfer → critic).

    return img_f.astype(np.uint8)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="images")
    parser.add_argument("--output-dir", default="abb_output_full")
    parser.add_argument("--feedback-dir", default="feedback_auto")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed (0 for no limit)")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    feed_dir = Path(args.feedback_dir).resolve()
    
    out_dir.mkdir(parents=True, exist_ok=True)
    feed_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading BiRefNet...")
    try:
        from transformers import AutoModelForImageSegmentation
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        birefnet_path = r"C:\Users\Ronit\.cache\huggingface\hub\models--ZhengPeng7--BiRefNet\snapshots\e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
        birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True, local_files_only=True).to(device)
        birefnet.eval()
    except Exception as e:
        print(f"Failed to load BiRefNet: {e}")
        birefnet = None

    print("Loading ColorMLP...")
    color_model_path = BASE_DIR / "models" / "color_predictor.pkl"
    color_model = None
    if color_model_path.exists():
        with open(color_model_path, "rb") as f:
            bundle = pickle.load(f)
        color_model = ColorMLP(input_dim=16).to(device)
        color_model.load_state_dict(bundle["model_state_dict"])
        color_model.eval()
        x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32, device=device)
        x_std = torch.tensor(bundle["x_std"], dtype=torch.float32, device=device)
        
    print("Loading Spatial Correction Fields...")
    spatial_path = BASE_DIR / "models" / "spatial_correction.pth"
    spatial_model = SpatialCorrectionFieldGenerator().to(device)
    if spatial_path.exists():
        spatial_model.load_state_dict(torch.load(spatial_path, map_location=device))
    spatial_model.eval()

    print("Loading Golden Reference Index...")
    ref_index = build_index()
    ref_names = list(ref_index.keys())
    if ref_names:
        ref_matrix = np.vstack(list(ref_index.values()))
    else:
        ref_matrix = None

    engine = ProposalEngine()
    trust_region = TrustRegionLineSearch()
    classifier = DegradationClassifier()
    vlm_reasoner = VLMReasoner()

    print("Loading DINOv2 Dense Semantic Index...")
    dino_index = DINOv2DenseIndex(device=device)
    try:
        dino_index.load_model()
        dino_index.build_index()
    except Exception as e:
        print(f"[DINOv2 Index] Failed to load: {e}. Exemplar transfer disabled.")
        dino_index = None

    cross_attn_fetcher = CrossAttentionColorFetcher(temperature=0.07)
    sw_transfer = SlicedWassersteinTransfer(n_projections=64)
    direct_replacer = DirectStructuralReplacement(overlap_sigma=1.5, blend_strength=0.90)
    
    print("\nBeginning Unified Full Architecture Processing...\n")
    processed_count = 0
    all_images = [p for p in in_dir.glob("*.*") if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    total_images = len(all_images)
    
    for img_path in all_images:
        print(f"\n" + "="*70)
        print(f"[{processed_count+1}/{total_images}] Processing Pipeline for: {img_path.name}")
        print("="*70)
        
        # ================================================================
        # PHASE 0: LOAD & CLASSIFY DEGRADATION
        # ================================================================
        raw_pil = Image.open(img_path).convert("RGB")
        raw_bgr = cv2.cvtColor(np.array(raw_pil), cv2.COLOR_RGB2BGR)

        # Classify degradation FIRST (replaces scattered heuristics)
        report = classifier.classify(raw_bgr)
        print(classifier.format_report(report))

        # Apply targeted pre-restoration (exposure fixes only)
        img_bgr = classifier_restore(raw_bgr, report)
        img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

        # Re-classify after pre-restoration (exposure fix may have changed metrics)
        if report.has_exposure_issue:
            report = classifier.classify(img_bgr)
            print(f"[Post-Restore] Re-classified: {report.primary.value}")

        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Early-Exit Fast Track for clean images
        if not report.is_degraded and report.severity < 0.2:
            print("[Pipeline] Image classified as CLEAN (Fast Track). Bypassing heavy networks.")
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), img_bgr)
            processed_count += 1
            if args.limit > 0 and processed_count >= args.limit:
                break
            continue
            
        else:
            # ================================================================
            # PHASE 1: AGENT DEBATE & PROPOSALS
            # ================================================================
            proposals = []
            print(f"\n--- AGENT DEBATE & PROPOSALS (degradation: {report.primary.value}) ---")
            
            # Log key metrics
            print(f"[Metrics] Mean_A: {report.metrics['mean_a']:.1f}, "
                  f"Luma_IQR: {report.metrics['luma_iqr']:.1f}, "
                  f"Laplacian: {report.metrics['laplacian_var']:.1f}, "
                  f"Saturation: {report.metrics['mean_sat']:.1f}")
            
            is_monochrome = report.is_bw
            if is_monochrome:
                print("[Gating] MONOCHROME detected. Global shifts locked to da=0, db=0.")
                
            # ---- DINOv2 Structural Matching (Primary Exemplar Path) ----
            ref_similarity = 0.0
            if dino_index is not None and dino_index.index:
                ref_name, ref_sim, q_feats, r_feats = dino_index.find_best_match(img_pil)
                ref_similarity = ref_sim
                
                if ref_name and ref_sim > 0.70:
                    ref_lab = dino_index.ref_images[ref_name]
                    print(f"[DINOv2 Index] Best structural match: {ref_name} (similarity: {ref_sim:.3f})")

                    # DINOv2 Color Sanity Gate to prevent color-invariant structural trap
                    ref_mean_a = ref_lab[:, :, 1].mean()
                    ref_mean_b = ref_lab[:, :, 2].mean()
                    img_mean_a = img_lab[:, :, 1].mean()
                    img_mean_b = img_lab[:, :, 2].mean()

                    if (abs(ref_mean_a - 128.0) > 15.0 or abs(ref_mean_b - 128.0) > 15.0):
                        # Divergent cast check
                        if np.sign(ref_mean_a - 128.0) != np.sign(img_mean_a - 128.0) or np.sign(ref_mean_b - 128.0) != np.sign(img_mean_b - 128.0):
                            print(f"[DINOv2 Sanity Gate] Match rejected! Reference {ref_name} has a divergent extreme color cast. Falling back to generic agents.")
                            ref_sim = 0.0  # Bypass exemplar matching

                    # ---- HIGH CONFIDENCE: Direct Structural Replacement ----
                    if ref_sim > 0.90:
                        print(f"[DirectReplacement] High-confidence match (sim={ref_sim:.3f}). "
                              f"Using histogram-specification color transfer.")
                        direct_lab = direct_replacer.transfer(
                            degraded_lab=img_lab, ref_lab=ref_lab,
                            query_features=q_feats, ref_features=r_feats,
                            degradation_type=report.primary.value
                        )
                        proposals.append(CorrectionProposal(
                            source_name=ProposalSource.DIRECT_REPLACEMENT,
                            spatial_lab_image=direct_lab,
                            confidence=min(0.98, ref_sim),
                            reasoning=f"Direct histogram replacement from {ref_name}",
                            ref_similarity=ref_sim
                        ))

                    # ---- MEDIUM CONFIDENCE: Exemplar Transfer ----
                    exemplar_lab_result = None

                    if is_monochrome:
                        # B&W image → Cross-Attention color fetch from reference
                        print(f"[Cross-Attention] Fetching a*b* from {ref_name} for B/W colorization...")
                        exemplar_lab_result = cross_attn_fetcher.transfer(
                            degraded_lab=img_lab, ref_lab=ref_lab,
                            query_features=q_feats, ref_features=r_feats
                        )
                        print(f"[Cross-Attention] Spatial a*b* transfer complete (edge-aware).")

                    elif report.has_color_cast and DegradationType.PINK_CAST in report.all_degradations:
                        # Pink cast → Sliced Wasserstein OT
                        print(f"[Sliced Wasserstein OT] Computing optimal transport from {ref_name}...")
                        exemplar_lab_result = sw_transfer.transfer(
                            degraded_lab=img_lab, ref_lab=ref_lab
                        )
                        print(f"[Sliced Wasserstein OT] Color distribution transfer complete.")

                    else:
                        # General degradation → Cross-attention
                        print(f"[Cross-Attention] Fetching a*b* from {ref_name}...")
                        exemplar_lab_result = cross_attn_fetcher.transfer(
                            degraded_lab=img_lab, ref_lab=ref_lab,
                            query_features=q_feats, ref_features=r_feats
                        )
                        print(f"[Cross-Attention] Spatial a*b* transfer complete.")

                    if exemplar_lab_result is not None:
                        proposals.append(CorrectionProposal(
                            source_name=ProposalSource.EXEMPLAR_TRANSFER,
                            spatial_lab_image=exemplar_lab_result,
                            confidence=min(0.95, ref_sim),
                            reasoning=f"DINOv2 structural match to {ref_name}",
                            ref_similarity=ref_sim
                        ))
                else:
                    print(f"[DINOv2 Index] No strong match (best: {ref_name}, sim: {ref_sim:.3f}). "
                          f"Falling back to computed agents.")

            # ---- Sobel-Based Golden Reference ----
            if ref_matrix is not None:
                desc = compute_sobel_descriptor(img_path)
                similarities = np.dot(ref_matrix, desc)
                best_idx = np.argmax(similarities)
                best_score = float(similarities[best_idx])
                if best_score > 0.8:
                    best_ref_name = ref_names[best_idx]
                    print(f"[Golden Reference] Matched {best_ref_name} with score {best_score:.2f}")
                    mean_a = report.metrics["mean_a"]
                    mean_b = report.metrics["mean_b"]
                    da_ref = 128.0 - mean_a
                    db_ref = 128.0 - mean_b
                    proposals.append(CorrectionProposal(
                        source_name=ProposalSource.REFERENCE_PRIOR,
                        global_shifts=(0.0, float(da_ref), float(db_ref)),
                        confidence=0.85
                    ))

            # ---- pHash Memory Agent ----
            phash = compute_perceptual_hash(img_pil)
            phash_fix = find_approved_fix(phash, str(feed_dir))
            if phash_fix:
                print(f"[Memory Agent] Found historical pHash match! Proposing exact recall {tuple(phash_fix)}.")
                proposals.append(CorrectionProposal(
                    source_name=ProposalSource.PHASH_MEMORY, 
                    global_shifts=tuple(phash_fix), 
                    confidence=0.99
                ))
            else:
                print(f"[Memory Agent] No historical override found for pHash {phash}.")
                
            # ---- VLM Reasoner (Real API or Expert System) ----
            vlm_payload = vlm_reasoner.analyze(img_pil, img_lab, phash, report)
            safe_vlm_cmd = validate_adjust_lab_call(vlm_payload)
            if safe_vlm_cmd:
                conf = vlm_payload.get("confidence", 0.70)
                print(f"[VLM Reasoner] Reasoning: \"{vlm_payload['reasoning']}\"")
                print(f"[VLM Reasoner] Proposal: ({safe_vlm_cmd.delta_l_ticks}, {safe_vlm_cmd.delta_a_ticks}, {safe_vlm_cmd.delta_b_ticks}) | Confidence: {conf:.2f}")
                proposals.append(CorrectionProposal(
                    source_name=ProposalSource.VLM_REASONING,
                    global_shifts=(safe_vlm_cmd.delta_l_ticks, safe_vlm_cmd.delta_a_ticks, safe_vlm_cmd.delta_b_ticks),
                    confidence=conf
                ))
                
            # ---- Spatial Field U-Net ----
            img_tensor = F_vision.to_tensor(img_pil).unsqueeze(0).to(device)
            img_resized_spatial = F_vision.resize(img_tensor, [256, 256], antialias=True)
            with torch.no_grad():
                shifts_field, uncertainty = spatial_model(img_resized_spatial)
                shifts_field_np = shifts_field[0].cpu().numpy().transpose(1, 2, 0) # [256, 256, 3]
                conf = round(1.0 - float(uncertainty.mean().item()), 2)
                print(f"[Spatial Field U-Net] Derived true dense spatial map [256x256x3] | Confidence: {conf:.2f}")
                proposals.append(CorrectionProposal(
                    source_name=ProposalSource.SPATIAL_FIELDS,
                    spatial_fields=shifts_field_np,
                    confidence=conf
                ))
                
            # ---- Color MLP ----
            if color_model:
                with torch.no_grad():
                    features = extract_features_gpu(img_resized_spatial)
                    features_norm = (features - x_mean) / x_std
                    out = color_model(features_norm)
                    l, a, b, sat = out[0].cpu().numpy()
                    
                    dist = torch.norm(features_norm).item()
                    conf = max(0.1, 0.85 * math.exp(-dist / 10.0))
                    
                    print(f"[Color MLP] Computed shift: ({l:.2f}, {a:.2f}, {b:.2f}) | Distance: {dist:.2f} | Confidence: {conf:.2f}")
                    proposals.append(CorrectionProposal(
                        source_name=ProposalSource.COLOR_MLP, 
                        global_shifts=(float(l), float(a), float(b)), 
                        confidence=conf
                    ))
            
            if not proposals:
                proposals.append(CorrectionProposal(
                    source_name=ProposalSource.PURE_MATH,
                    global_shifts=(0, 0, 0),
                    confidence=1.0
                ))
                
            # ================================================================
            # PHASE 2: CRITIC RESOLUTION (Degradation-Aware)
            # ================================================================
            print(f"\n--- CRITIC RESOLUTION (degradation-aware: {report.primary.value}) ---")
            
            # Monochrome lock: zero out color shifts for global-shift proposals
            if is_monochrome:
                for p in proposals:
                    # Don't zero out exemplar/direct transfer — they have spatially-mapped colors
                    if p.spatial_lab_image is not None:
                        continue
                    if p.global_shifts:
                        p.global_shifts = (p.global_shifts[0], 0.0, 0.0)
                        
            # Pass DegradationReport to the engine for context-aware scoring
            best_proposal = engine.select_best(img_lab, proposals, degradation=report)
            source_name = best_proposal.source_name.value if hasattr(best_proposal.source_name, 'value') else str(best_proposal.source_name)
            print(f"\n[Critic Gate] Selected {source_name} as the winning proposal.")

            # ================================================================
            # PHASE 3: APPLY CORRECTION
            # ================================================================
            if best_proposal.spatial_lab_image is not None:
                # Spatial exemplar/direct proposal — use the full Lab image directly
                corrected_lab = best_proposal.spatial_lab_image.copy()
                if corrected_lab.shape[:2] != img_lab.shape[:2]:
                    corrected_lab = cv2.resize(corrected_lab, (img_lab.shape[1], img_lab.shape[0]),
                                               interpolation=cv2.INTER_LINEAR)
                corrected_lab[:, :, 0] = np.clip(corrected_lab[:, :, 0], 0, 255)
                corrected_lab[:, :, 1] = np.clip(corrected_lab[:, :, 1], 0, 255)
                corrected_lab[:, :, 2] = np.clip(corrected_lab[:, :, 2], 0, 255)
                final_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                print(f"[Application] Applied spatial Lab image directly (bypassed ProportionalController).")
            elif getattr(best_proposal, 'spatial_fields', None) is not None:
                # Spatial U-Net proposal — resize dense map and apply pixel-wise shifts
                field = best_proposal.spatial_fields
                if field.shape[:2] != img_lab.shape[:2]:
                    field = cv2.resize(field, (img_lab.shape[1], img_lab.shape[0]), interpolation=cv2.INTER_LINEAR)
                corrected_lab = img_lab.copy()
                corrected_lab += field
                corrected_lab[:, :, 0] = np.clip(corrected_lab[:, :, 0], 0, 255)
                corrected_lab[:, :, 1] = np.clip(corrected_lab[:, :, 1], 0, 255)
                corrected_lab[:, :, 2] = np.clip(corrected_lab[:, :, 2], 0, 255)
                final_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                print(f"[Application] Applied true pixel-wise spatial fields (bypassed ProportionalController).")
            else:
                shifts = best_proposal.global_shifts if best_proposal.global_shifts else (0, 0, 0)
                shift_vector = np.array(shifts, dtype=np.float32)
                target_lab_simulated = img_lab + shift_vector
                current_de00 = 15.0
                alpha, corrected_lab = trust_region.calculate_armijo_step(
                    img_lab, target_lab_simulated, shift_vector, current_de00
                )
                print(f"[Trust-Region Line Search] Approved Armijo step size alpha={alpha}.")
                
                corrected_lab[:, :, 0] = np.clip(corrected_lab[:, :, 0], 0, 255)
                corrected_lab[:, :, 1] = np.clip(corrected_lab[:, :, 1], 0, 255)
                corrected_lab[:, :, 2] = np.clip(corrected_lab[:, :, 2], 0, 255)
                
                # Smart Proportional Controller
                controller = ProportionalController(img_bgr)
                controller.apply_lab_shift(shifts[0], shifts[1], shifts[2])
                final_bgr = cv2.cvtColor(controller.current_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # ================================================================
        # PHASE 4: SEGMENT & FORMAT (runs for ALL images, including clean)
        # ================================================================
        corrected_pil = Image.fromarray(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB))
        processor = IsolatedBiRefNetProcessor(birefnet)
        processor.set_clean_image(corrected_pil)
        mask_pil = processor.segment()
        print(f"[Core Architecture] Isolated foreground from corrected image.")
        
        # Threshold mask to clean binary
        mask_np = np.array(mask_pil.convert("L"))
        _, mask_binary = cv2.threshold(mask_np, 127, 255, cv2.THRESH_BINARY)
        
        fg_ratio = mask_binary.sum() / (255.0 * mask_binary.shape[0] * mask_binary.shape[1])
        print(f"[Segmentation] Foreground coverage: {fg_ratio*100:.1f}%")

        # ABB Standard Formatting
        canvas_bgr = create_abb_canvas(final_bgr, mask_binary)
        
        # ================================================================
        # PHASE 5: FINAL AUDIT (Edge Preservation Check)
        # ================================================================
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        edges_before = cv2.Canny(gray, 100, 200)
        edges_after = cv2.Canny(cv2.cvtColor(final_bgr, cv2.COLOR_BGR2GRAY), 100, 200)
        
        edge_count_before = np.count_nonzero(edges_before)
        edge_count_after = np.count_nonzero(edges_after)
        
        edge_loss = 0.0
        if edge_count_before > 0:
            edge_loss = max(0.0, (edge_count_before - edge_count_after) / float(edge_count_before))
            
        print(f"[Internal Critic] Edge loss: {edge_loss*100:.1f}%")
        
        if edge_loss > 0.15:
            print("[Internal Critic] WARNING: High edge loss (>15%). Routing to review queue.")
            bad_dir = BASE_DIR / "review_queue"
            bad_dir.mkdir(exist_ok=True)
            bad_file = bad_dir / img_path.name
            cv2.imwrite(str(bad_file), canvas_bgr)
        else:
            out_file = out_dir / img_path.name
            cv2.imwrite(str(out_file), canvas_bgr)
            print(f"[ABB Formatter] Canvas constructed & padded. => Saved to {out_file.name}")
        
        # ================================================================
        # PHASE 6: FEEDBACK STORE
        # ================================================================
        if report.is_degraded:
            shifts_for_log = best_proposal.global_shifts if best_proposal.global_shifts else (0, 0, 0)
            if best_proposal.source_name == ProposalSource.SPATIAL_FIELDS:
                shifts_for_log = tuple(best_proposal.spatial_fields.mean(axis=(0, 1)))
            record = {
                "image": img_path.name,
                "phash": phash,
                "degradation_type": report.primary.value,
                "degradation_severity": round(report.severity, 3),
                "applied_method": str(best_proposal.source_name),
                "shifts": [float(s) for s in shifts_for_log] if best_proposal.global_shifts else [0, 0, 0],
                "exemplar_transfer": best_proposal.spatial_lab_image is not None,
                "ref_similarity": round(best_proposal.ref_similarity, 3),
                "decision": "approved"
            }
        else:
            record = {
                "image": img_path.name,
                "phash": compute_perceptual_hash(img_pil),
                "degradation_type": "clean",
                "degradation_severity": 0.0,
                "applied_method": "passthrough",
                "shifts": [0, 0, 0],
                "exemplar_transfer": False,
                "ref_similarity": 0.0,
                "decision": "approved"
            }
        save_feedback(str(feed_dir), img_path.name, record)
        processed_count += 1
        
        if args.limit > 0 and processed_count >= args.limit:
            break

    print(f"\nUnified Pipeline execution complete! {processed_count} features successfully ran through the Critic pipeline.")

if __name__ == '__main__':
    main()
