import os
import argparse
import pickle
import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from xai.birefnet_tensor_isolation import IsolatedBiRefNetProcessor
from xai.proposal_and_verify import ProposalEngine, CorrectionProposal, ProposalSource
from core.batch_process import HybridConceptMLP
from core.feedback_store import compute_perceptual_hash, save_feedback

# Setup paths
BASE_DIR = Path(__file__).resolve().parent

def apply_shifts_numpy(img_bgr, shifts):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] += shifts[0]
    lab[:, :, 1] += shifts[1]
    lab[:, :, 2] += shifts[2]
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def create_abb_canvas(img_bgr, mask_pil):
    canvas = np.ones((1440, 1920, 3), dtype=np.uint8) * 255
    mask_np = np.array(mask_pil)
    
    # Simple foreground extraction using mask
    foreground = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_np)
    
    # Find bounding box
    coords = cv2.findNonZero(mask_np)
    if coords is None:
        return canvas
        
    x, y, w, h = cv2.boundingRect(coords)
    cropped_fg = foreground[y:y+h, x:x+w]
    cropped_mask = mask_np[y:y+h, x:x+w]
    
    # Scale to fit standard ABB margin (e.g. 75% of canvas height/width max)
    scale_w = (1920 * 0.75) / w
    scale_h = (1440 * 0.75) / h
    scale = min(scale_w, scale_h, 1.0)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_fg = cv2.resize(cropped_fg, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_mask = cv2.resize(cropped_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center on canvas
    start_x = (1920 - new_w) // 2
    start_y = (1440 - new_h) // 2
    
    # Paste
    roi = canvas[start_y:start_y+new_h, start_x:start_x+new_w]
    mask_inv = cv2.bitwise_not(resized_mask)
    
    bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = cv2.add(bg, resized_fg)
    
    return canvas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="images")
    parser.add_argument("--output-dir", default="abb_output_auto")
    parser.add_argument("--feedback-dir", default="feedback_auto")
    parser.add_argument("--model", default="color_predictor.pkl")
    parser.add_argument("--review-mode", default="none")
    args = parser.parse_args()
    
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    feed_dir = Path(args.feedback_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    feed_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BiRefNet
    print("Loading BiRefNet from local cache...")
    try:
        from transformers import AutoModelForImageSegmentation
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        birefnet_path = r"C:\Users\Ronit\.cache\huggingface\hub\models--ZhengPeng7--BiRefNet\snapshots\e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"
        birefnet = AutoModelForImageSegmentation.from_pretrained(birefnet_path, trust_remote_code=True, local_files_only=True).to(device)
        birefnet.eval()
    except Exception as e:
        print(f"Failed to load BiRefNet: {e}")
        birefnet = None

    # Load Color Predictor
    color_model_path = Path(args.model)
    if not color_model_path.exists():
        color_model_path = BASE_DIR / "models" / "color_predictor.pkl"
        
    color_model = None
    if color_model_path.exists():
        with open(color_model_path, "rb") as f:
            bundle = pickle.load(f)
        from training.train_color_model import ColorMLP
        color_model = ColorMLP(input_dim=16).to(device)
        color_model.load_state_dict(bundle["model_state_dict"])
        color_model.eval()
        x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32, device=device)
        x_std = torch.tensor(bundle["x_std"], dtype=torch.float32, device=device)
    else:
        print(f"Warning: {color_model_path} not found.")

    engine = ProposalEngine()
    
    for img_path in in_dir.glob("*.*"):
        if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
            continue
            
        print(f"Processing {img_path.name}...")
        img_pil = Image.open(img_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # 1. Segmentation
        processor = IsolatedBiRefNetProcessor(birefnet)
        processor.set_clean_image(img_pil)
        mask_pil = processor.segment()
        
        # 2. Extract Features & Generate Proposals
        proposals = []
        
        # Fake features for ColorMLP (in full impl, use extract_features_gpu)
        if color_model:
            # We'll generate a dummy proposal using zero shifts if features fail
            import torchvision.transforms.functional as F_vision
            from training.train_color_model import extract_features_gpu
            img_tensor = F_vision.to_tensor(img_pil).unsqueeze(0).to(device)
            img_resized = F_vision.resize(img_tensor, [256, 256], antialias=True)
            with torch.no_grad():
                features = extract_features_gpu(img_resized)
                features_norm = (features - x_mean) / x_std
                out = color_model(features_norm)
                l, a, b, sat = out[0].cpu().numpy()
                proposals.append(CorrectionProposal(source_name=ProposalSource.COLOR_MLP, global_shifts=(l, a, b), confidence=0.9))
                
        if not proposals:
            proposals.append(CorrectionProposal(source_name=ProposalSource.PURE_MATH, global_shifts=(0, 0, 0), confidence=1.0))
            
        # 3. Select Best Proposal
        best_proposal = engine.select_best(img_lab, proposals)
        
        # 4. Apply Fix
        shifts = best_proposal.global_shifts if best_proposal.global_shifts else (0, 0, 0)
        corrected_bgr = apply_shifts_numpy(img_bgr, shifts)
        
        # 5. ABB Canvas Layout
        canvas_bgr = create_abb_canvas(corrected_bgr, mask_pil)
        
        # Output
        out_file = out_dir / img_path.name
        cv2.imwrite(str(out_file), canvas_bgr)
        print(f"Saved {out_file.name}")
        
        # Feedback Logging
        phash = compute_perceptual_hash(img_pil)
        record = {
            "image": img_path.name,
            "phash": phash,
            "applied_method": best_proposal.source_name.value if hasattr(best_proposal.source_name, 'value') else str(best_proposal.source_name),
            "shifts": [float(s) for s in shifts],
            "decision": "approved"
        }
        save_feedback(str(feed_dir), img_path.name, record)

if __name__ == '__main__':
    main()
