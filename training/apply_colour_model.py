# training/apply_colour_model.py
import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

# Adjust paths to the repository root
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = BASE_DIR / "color_predictor.pkl"
from core.batch_process import HybridConceptMLP  # Import the new architecture


def apply_model_to_directory(input_dir: str, output_dir: str, model_path: str):
    in_path, out_path = Path(input_dir), Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridConceptMLP(input_dim=16).to(device)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32, device=device)
    x_std = torch.tensor(bundle["x_std"], dtype=torch.float32, device=device)

    print(f"Applying color model to {in_path} -> {out_path} using {device}")

    # Needs the extract_features_gpu logic from train_color_model.py
    import torchvision.io as io
    import torchvision.transforms.functional as F_vision

    from training.train_color_model import extract_features_gpu

    for img_file in in_path.glob("*.*"):
        if img_file.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
            continue

        try:
            # Load and extract features
            img_tensor = io.read_image(str(img_file)).unsqueeze(0).to(device)
            img_resized = F_vision.resize(img_tensor, [256, 256], antialias=True)

            with torch.no_grad():
                features = extract_features_gpu(img_resized)
                features_norm = (features - x_mean) / x_std

                # Predict utilizing the HybridConceptMLP
                shifts, concepts = model(features_norm)

            l_shift, a_shift, b_shift, sat_gain = shifts[0].cpu().numpy()

            # Apply to original resolution image
            img_bgr = cv2.imread(str(img_file))
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

            lab[:, :, 0] += l_shift
            lab[:, :, 1] += a_shift
            lab[:, :, 2] += b_shift
            lab = np.clip(lab, 0, 255).astype(np.uint8)

            corrected_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(out_path / img_file.name), corrected_bgr)
            print(f"Processed: {img_file.name}")

        except Exception as e:
            print(f"Failed {img_file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--model", default=str(DEFAULT_MODEL), help="Path to .pkl model"
    )
    args = parser.parse_args()

    apply_model_to_directory(args.input, args.output, args.model)
