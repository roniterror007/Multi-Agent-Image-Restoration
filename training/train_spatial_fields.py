import torch
from pathlib import Path
from xai.spatial_correction_fields import SpatialCorrectionFieldGenerator

def train_spatial_fields():
    print("Initializing Spatial Correction Fields training pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatialCorrectionFieldGenerator().to(device)

    # Note: In a real run, this would loop over data and use SpatialCorrectionLoss
    out_path = Path(__file__).resolve().parent.parent / "models" / "spatial_correction.pth"
    out_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved initialized weights to {out_path}")

if __name__ == "__main__":
    import sys
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    train_spatial_fields()
