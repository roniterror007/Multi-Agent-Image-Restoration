# xai/spatial_correction_fields.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialCorrectionFieldGenerator(nn.Module):
    """U-Net architecture predicting [H, W, 3] Lab shifts and [H, W, 1] uncertainty."""

    def __init__(self, in_channels=3, base_channels=16):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)

        # Decoder
        self.dec2 = self._conv_block(
            base_channels * 4 + base_channels * 2, base_channels * 2
        )
        self.dec1 = self._conv_block(base_channels * 2 + base_channels, base_channels)

        # Output Heads
        self.shift_head = nn.Conv2d(base_channels, 3, kernel_size=3, padding=1)
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Constrain uncertainty to [0.0, 1.0]
        )
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Downsample
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Upsample with skip connections
        d2 = F.interpolate(e3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Predict fields
        shifts = self.shift_head(d1)
        uncertainty = self.uncertainty_head(d1)
        return shifts, uncertainty


class SpatialCorrectionLoss(nn.Module):
    """Loss function with integrated Gamut Penalty to ensure physical realism."""

    def __init__(self, tv_weight=0.1, gamut_weight=1.0):
        super().__init__()
        self.tv_weight = tv_weight
        self.gamut_weight = gamut_weight

    def forward(self, pred_shifts, target_shifts, current_lab):
        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(pred_shifts, target_shifts)

        # 2. Smoothness Loss (Total Variation) to prevent noisy/blotchy edits
        tv_loss = torch.mean(
            torch.abs(pred_shifts[:, :, :, :-1] - pred_shifts[:, :, :, 1:])
        ) + torch.mean(torch.abs(pred_shifts[:, :, :-1, :] - pred_shifts[:, :, 1:, :]))

        # 3. Gamut Penalty (Differentiable constraint)
        simulated_lab = current_lab + pred_shifts
        # Penalize L outside [0, 255]
        l_penalty = torch.mean(F.relu(simulated_lab[:, 0] - 255.0)) + torch.mean(
            F.relu(-simulated_lab[:, 0])
        )
        # Penalize a, b outside [-128, 127]
        a_penalty = torch.mean(F.relu(simulated_lab[:, 1] - 127.0)) + torch.mean(
            F.relu(-simulated_lab[:, 1] - 128.0)
        )
        b_penalty = torch.mean(F.relu(simulated_lab[:, 2] - 127.0)) + torch.mean(
            F.relu(-simulated_lab[:, 2] - 128.0)
        )

        gamut_loss = l_penalty + a_penalty + b_penalty

        return (
            recon_loss + (self.tv_weight * tv_loss) + (self.gamut_weight * gamut_loss)
        )
