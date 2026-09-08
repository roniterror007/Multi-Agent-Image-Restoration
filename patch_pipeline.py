import sys

with open("run_pipeline.py", "r", encoding="utf-8") as f:
    code = f.read()

# Chunk 1: Imports
import_old = """from training.train_color_model import ColorMLP, extract_features_gpu
import torchvision.transforms.functional as F_vision"""
import_new = """from training.train_color_model import ColorMLP, extract_features_gpu
import torchvision.transforms.functional as F_vision
from core.vlm_reasoner import VLMReasoner
import math"""
code = code.replace(import_old, import_new)

# Chunk 2: Remove mock_vlm_call (we will just replace its body with a pass or remove it entirely)
# Finding the exact block:
import re
code = re.sub(r'def mock_vlm_call\(.*?\):.*?(?=def main\(\):)', '\n', code, flags=re.DOTALL)

# Chunk 3: Init VLMReasoner
init_old = """    engine = ProposalEngine()
    trust_region = TrustRegionLineSearch()
    classifier = DegradationClassifier()"""
init_new = """    engine = ProposalEngine()
    trust_region = TrustRegionLineSearch()
    classifier = DegradationClassifier()
    vlm_reasoner = VLMReasoner()"""
code = code.replace(init_old, init_new)

# Chunk 4: VLM Reasoner execution
vlm_old = """            # ---- VLM Reasoner (uses DegradationReport) ----
            vlm_payload = mock_vlm_call(img_lab, phash, report)
            safe_vlm_cmd = validate_adjust_lab_call(vlm_payload)
            if safe_vlm_cmd:
                conf = round(random.uniform(0.70, 0.88) + vlm_payload.get("confidence_boost", 0.0), 2)
                conf = min(conf, 0.99)
                print(f"[VLM Reasoner] Reasoning: \\"{vlm_payload['reasoning']}\\"")
                print(f"[VLM Reasoner] Proposal: ({safe_vlm_cmd.delta_l_ticks}, {safe_vlm_cmd.delta_a_ticks}, {safe_vlm_cmd.delta_b_ticks}) | Confidence: {conf}")
                proposals.append(CorrectionProposal(
                    source_name=ProposalSource.VLM_REASONING,
                    global_shifts=(safe_vlm_cmd.delta_l_ticks, safe_vlm_cmd.delta_a_ticks, safe_vlm_cmd.delta_b_ticks),
                    confidence=conf
                ))"""

vlm_new = """            # ---- VLM Reasoner (Real API or Expert System) ----
            vlm_payload = vlm_reasoner.analyze(img_pil, img_lab, phash, report)
            safe_vlm_cmd = validate_adjust_lab_call(vlm_payload)
            if safe_vlm_cmd:
                conf = vlm_payload.get("confidence", 0.70)
                print(f"[VLM Reasoner] Reasoning: \\"{vlm_payload['reasoning']}\\"")
                print(f"[VLM Reasoner] Proposal: ({safe_vlm_cmd.delta_l_ticks}, {safe_vlm_cmd.delta_a_ticks}, {safe_vlm_cmd.delta_b_ticks}) | Confidence: {conf:.2f}")
                proposals.append(CorrectionProposal(
                    source_name=ProposalSource.VLM_REASONING,
                    global_shifts=(safe_vlm_cmd.delta_l_ticks, safe_vlm_cmd.delta_a_ticks, safe_vlm_cmd.delta_b_ticks),
                    confidence=conf
                ))"""
code = code.replace(vlm_old, vlm_new)

# Chunk 5: Spatial Field
spatial_old = """            # ---- Spatial Field U-Net ----
            img_tensor = F_vision.to_tensor(img_pil).unsqueeze(0).to(device)
            img_resized_spatial = F_vision.resize(img_tensor, [256, 256], antialias=True)
            with torch.no_grad():
                shifts_field, uncertainty = spatial_model(img_resized_spatial)
                mean_shift = shifts_field.mean(dim=(2, 3))[0].cpu().numpy()
                conf = round(1.0 - float(uncertainty.mean().item()), 2)
                print(f"[Spatial Field U-Net] Derived global shift: ({mean_shift[0]:.2f}, {mean_shift[1]:.2f}, {mean_shift[2]:.2f}) | Confidence: {conf}")
                proposals.append(CorrectionProposal(
                    source_name=ProposalSource.SPATIAL_FIELDS,
                    global_shifts=(float(mean_shift[0]), float(mean_shift[1]), float(mean_shift[2])),
                    confidence=conf
                ))"""
spatial_new = """            # ---- Spatial Field U-Net ----
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
                ))"""
code = code.replace(spatial_old, spatial_new)

# Chunk 6: MLP
mlp_old = """            # ---- Color MLP ----
            if color_model:
                with torch.no_grad():
                    features = extract_features_gpu(img_resized_spatial)
                    features_norm = (features - x_mean) / x_std
                    out = color_model(features_norm)
                    l, a, b, sat = out[0].cpu().numpy()
                    conf = 0.85
                    print(f"[Color MLP] Computed shift: ({l:.2f}, {a:.2f}, {b:.2f}) | Confidence: {conf}")
                    proposals.append(CorrectionProposal(
                        source_name=ProposalSource.COLOR_MLP, 
                        global_shifts=(float(l), float(a), float(b)), 
                        confidence=conf
                    ))"""
mlp_new = """            # ---- Color MLP ----
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
                    ))"""
code = code.replace(mlp_old, mlp_new)


# Chunk 7: Application
app_old = """            if best_proposal.spatial_lab_image is not None:
                # Spatial exemplar/direct proposal — use the full Lab image directly
                corrected_lab = best_proposal.spatial_lab_image.copy()
                if corrected_lab.shape[:2] != img_lab.shape[:2]:
                    corrected_lab = cv2.resize(corrected_lab, (img_lab.shape[1], img_lab.shape[0]),
                                               interpolation=cv2.INTER_LINEAR)
                corrected_lab[:, :, 0] = np.clip(corrected_lab[:, :, 0], 0, 255)
                corrected_lab[:, :, 1] = np.clip(corrected_lab[:, :, 1], 0, 255)
                corrected_lab[:, :, 2] = np.clip(corrected_lab[:, :, 2], 0, 255)
                final_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                print(f"[Application] Applied spatial Lab image directly (bypassed ProportionalController).")"""

app_new = """            if best_proposal.spatial_lab_image is not None:
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
                print(f"[Application] Applied true pixel-wise spatial fields (bypassed ProportionalController).")"""
code = code.replace(app_old, app_new)

# One more fix: logging shifts if spatial
log_old = """shifts_for_log = best_proposal.global_shifts if best_proposal.global_shifts else (0, 0, 0)"""
log_new = """shifts_for_log = best_proposal.global_shifts if best_proposal.global_shifts else (0, 0, 0)
            if best_proposal.source_name == ProposalSource.SPATIAL_FIELDS:
                shifts_for_log = tuple(best_proposal.spatial_fields.mean(axis=(0, 1)))"""
code = code.replace(log_old, log_new)


with open("run_pipeline.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied to run_pipeline.py")
