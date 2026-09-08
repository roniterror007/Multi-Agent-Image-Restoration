# training/train_fusion_unet.py
from pathlib import Path

import torch
import torch.nn as nn


# We reuse the U-Net structure from our spatial correction fields,
# but adapted for 1-channel binary mask output instead of 3-channel color shifts.
class MaskFusionUNet(nn.Module):
    def __init__(
        self, in_channels=4, base_channels=16
    ):  # Input: RGB (3) + Initial Mask (1)
        super().__init__()

        self.enc1 = self._block(in_channels, base_channels)
        self.enc2 = self._block(base_channels, base_channels * 2)
        self.dec1 = self._block(base_channels * 2 + base_channels, base_channels)
        self.out = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Sigmoid(),  # Output a soft probability mask [0, 1]
        )
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        import torch.nn.functional as F

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))

        d1 = F.interpolate(e2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def train_mask_refiner():
    print("Initializing Mask Fusion U-Net training pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskFusionUNet().to(device)

    # Standard training loop stub
    # 1. Load dataset containing raw image + BiRefNet mask + Ground Truth manual mask
    # 2. Optimize using Binary Cross Entropy (BCE) Loss
    # criterion = nn.BCELoss()
    # ...

    out_path = Path(__file__).resolve().parent.parent / "models" / "fusion_unet.pth"
    out_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved initialized weights to {out_path}")


if __name__ == "__main__":
    train_mask_refiner()
