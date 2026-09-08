import argparse
import json
import os
import pickle
from datetime import datetime

import torch
import torch.nn.functional as F
import torchvision.io as io
import torchvision.transforms.functional as F_vision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DEFAULT_INPUT_FOLDER = os.path.join(PROJECT_ROOT, "good_pics")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_ROOT, "reference_report.json")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "color_predictor.pkl")

INPUT_FEATURES = [
    "V_P50",
    "Entropy",
    "Edge Density (Sobel)",
    "White BG Ratio",
    "Mean_A",
    "Mean_B",
    "Mean_Sat",
    "Laplacian_Var_Q1",
    "Laplacian_Var_Q2",
    "Laplacian_Var_Q3",
    "Laplacian_Var_Q4",
    "Glare_Ratio",
    "Channel_Gap",
    "Sat_P95",
    "Low_Sat_Ratio",
    "Luma_IQR",
]
OUTPUT_FEATURES = ["L_Shift", "A_Shift", "B_Shift", "Sat_Gain"]
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def print_status(message):
    print(message, flush=True)


# ---------------------------------------------------------
# GPU TENSOR MATH: Color Spaces & Edge Detection
# ---------------------------------------------------------


def rgb_to_lab_gpu(rgb_tensor):
    """Converts a batch of RGB tensors [B, 3, H, W] to CIELAB purely on the GPU."""
    # Normalize RGB to 0-1
    rgb = rgb_tensor.float() / 255.0

    # sRGB to Linear RGB
    mask = rgb > 0.04045
    rgb[mask] = torch.pow((rgb[mask] + 0.055) / 1.055, 2.4)
    rgb[~mask] = rgb[~mask] / 12.92

    # Linear RGB to XYZ
    xyz = torch.zeros_like(rgb)
    xyz[:, 0, :, :] = (
        rgb[:, 0] * 0.4124564 + rgb[:, 1] * 0.3575761 + rgb[:, 2] * 0.1804375
    )
    xyz[:, 1, :, :] = (
        rgb[:, 0] * 0.2126729 + rgb[:, 1] * 0.7151522 + rgb[:, 2] * 0.0721750
    )
    xyz[:, 2, :, :] = (
        rgb[:, 0] * 0.0193339 + rgb[:, 1] * 0.1191920 + rgb[:, 2] * 0.9503041
    )

    # Normalize by D65 White Point
    xyz[:, 0, :, :] /= 0.95047
    xyz[:, 1, :, :] /= 1.00000
    xyz[:, 2, :, :] /= 1.08883

    # XYZ to LAB
    mask = xyz > 0.008856
    xyz[mask] = torch.pow(xyz[mask], 1 / 3)
    xyz[~mask] = (7.787 * xyz[~mask]) + (16.0 / 116.0)

    lab = torch.zeros_like(rgb)
    lab[:, 0, :, :] = (116.0 * xyz[:, 1, :, :]) - 16.0  # L
    lab[:, 1, :, :] = 500.0 * (xyz[:, 0, :, :] - xyz[:, 1, :, :])  # a
    lab[:, 2, :, :] = 200.0 * (xyz[:, 1, :, :] - xyz[:, 2, :, :])  # b
    return lab


def calculate_edges_gpu(gray_tensor):
    """Calculates Edge Density using a Sobel Filter entirely on CUDA."""
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=torch.float32,
        device=gray_tensor.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=torch.float32,
        device=gray_tensor.device,
    ).view(1, 1, 3, 3)

    edge_x = F.conv2d(gray_tensor, sobel_x, padding=1)
    edge_y = F.conv2d(gray_tensor, sobel_y, padding=1)
    magnitude = torch.sqrt(edge_x**2 + edge_y**2)
    return (magnitude > 0.5).float().mean(dim=(1, 2, 3))  # Edge density per image


def rgb_to_hsv_gpu(rgb_tensor):
    """Converts a batch of RGB tensors [B, 3, H, W] to HSV purely on the GPU."""
    r = rgb_tensor[:, 0:1, :, :]
    g = rgb_tensor[:, 1:2, :, :]
    b = rgb_tensor[:, 2:3, :, :]

    cmax, max_idx = torch.max(rgb_tensor, dim=1, keepdim=True)
    cmin, _ = torch.min(rgb_tensor, dim=1, keepdim=True)
    delta = cmax - cmin

    h = torch.zeros_like(cmax)
    s = torch.zeros_like(cmax)
    v = cmax * 255.0  # Scale to match OpenCV 0-255 format

    # Avoid division by zero
    delta_safe = delta.clone()
    delta_safe[delta_safe == 0] = 1.0

    # Saturation (scaled 0-255 to match OpenCV)
    s[cmax > 0] = (delta[cmax > 0] / cmax[cmax > 0]) * 255.0

    # Hue
    idx = (max_idx == 0) & (delta > 0)
    h[idx] = ((g[idx] - b[idx]) / delta_safe[idx]) % 6.0

    idx = (max_idx == 1) & (delta > 0)
    h[idx] = ((b[idx] - r[idx]) / delta_safe[idx]) + 2.0

    idx = (max_idx == 2) & (delta > 0)
    h[idx] = ((r[idx] - g[idx]) / delta_safe[idx]) + 4.0

    h = (h / 6.0) % 1.0

    return torch.cat([h, s, v], dim=1)


def extract_features_gpu(batch_rgb):
    """Takes a batch of images [B, 3, H, W] on the GPU and extracts features in parallel."""
    B = batch_rgb.shape[0]
    batch_rgb_f = batch_rgb.float()

    # Grayscale
    gray = (
        batch_rgb_f[:, 0] * 0.299
        + batch_rgb_f[:, 1] * 0.587
        + batch_rgb_f[:, 2] * 0.114
    ).unsqueeze(1) / 255.0

    # HSV & LAB
    hsv = rgb_to_hsv_gpu(batch_rgb_f / 255.0)
    lab = rgb_to_lab_gpu(batch_rgb_f)

    # 1. Structural/Luma
    v_channel = hsv[:, 2, :, :].view(B, -1)
    p50 = torch.median(v_channel, dim=1).values
    white_bg_ratio = (gray >= 0.98).float().mean(dim=(1, 2, 3))
    edge_density = calculate_edges_gpu(gray)

    # Entropy (Simplified batch approximation)
    entropy = torch.zeros(B, device=batch_rgb.device)
    for i in range(B):
        hist = torch.histc(gray[i], bins=256, min=0, max=1)
        prob = hist[hist > 0] / hist.sum()
        entropy[i] = -(prob * torch.log2(prob)).sum()

    # 2. Color/Chroma
    mean_a = lab[:, 1, :, :].mean(dim=(1, 2))
    mean_b = lab[:, 2, :, :].mean(dim=(1, 2))
    mean_sat = hsv[:, 1, :, :].mean(dim=(1, 2))

    laplacian_kernel = torch.tensor(
        [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
        dtype=torch.float32,
        device=batch_rgb.device,
    ).view(1, 1, 3, 3)
    laplacian = F.conv2d(gray, laplacian_kernel, padding=1)
    h_half, w_half = gray.shape[2] // 2, gray.shape[3] // 2
    laplacian_var_q1 = laplacian[:, :, :h_half, :w_half].var(dim=(2, 3)).squeeze(1)
    laplacian_var_q2 = laplacian[:, :, :h_half, w_half:].var(dim=(2, 3)).squeeze(1)
    laplacian_var_q3 = laplacian[:, :, h_half:, :w_half].var(dim=(2, 3)).squeeze(1)
    laplacian_var_q4 = laplacian[:, :, h_half:, w_half:].var(dim=(2, 3)).squeeze(1)
    glare_ratio = ((hsv[:, 1] > 200) & (hsv[:, 2] > 200)).float().mean(dim=(1, 2))

    r = batch_rgb_f[:, 0]
    g = batch_rgb_f[:, 1]
    b = batch_rgb_f[:, 2]
    channel_gap = (
        (r - g).abs().mean(dim=(1, 2))
        + (g - b).abs().mean(dim=(1, 2))
        + (r - b).abs().mean(dim=(1, 2))
    ) / 3.0
    sat_flat = hsv[:, 1, :, :].view(B, -1)
    sat_p95 = torch.quantile(sat_flat, 0.95, dim=1)
    low_sat_ratio = (hsv[:, 1] < 20).float().mean(dim=(1, 2))
    v_q75 = torch.quantile(v_channel, 0.75, dim=1)
    v_q25 = torch.quantile(v_channel, 0.25, dim=1)
    luma_iqr = v_q75 - v_q25

    return torch.stack(
        [
            p50,
            entropy,
            edge_density,
            white_bg_ratio,
            mean_a,
            mean_b,
            mean_sat,
            laplacian_var_q1,
            laplacian_var_q2,
            laplacian_var_q3,
            laplacian_var_q4,
            glare_ratio,
            channel_gap,
            sat_p95,
            low_sat_ratio,
            luma_iqr,
        ],
        dim=1,
    )


def synthesize_factory_degradations(clean_batch):
    """Create reversible lighting and color failures seen in factory photography."""
    batch_size, _, height, width = clean_batch.shape
    device = clean_batch.device
    clean = clean_batch.float()

    # Cool/warm/pink/green cast: independent channel gains and sensor offsets.
    channel_gain = 0.76 + torch.rand(batch_size, 3, 1, 1, device=device) * 0.52
    channel_offset = (torch.rand(batch_size, 3, 1, 1, device=device) - 0.5) * 30.0
    cast = clean * channel_gain + channel_offset

    # Under/overexposure and contrast loss, without changing object geometry.
    exposure = 0.62 + torch.rand(batch_size, 1, 1, 1, device=device) * 0.68
    contrast = 0.70 + torch.rand(batch_size, 1, 1, 1, device=device) * 0.55
    exposure_contrast = ((clean - 127.5) * contrast + 127.5) * exposure

    # Low-color/near-monochrome camera failures. This teaches conservative recovery
    # of known source colors, but is not semantic colorization of unknown grayscale input.
    luma = clean[:, 0:1] * 0.299 + clean[:, 1:2] * 0.587 + clean[:, 2:3] * 0.114
    retained_color = 0.03 + torch.rand(batch_size, 1, 1, 1, device=device) * 0.52
    desaturated = luma + (clean - luma) * retained_color
    desaturated = desaturated * (
        0.78 + torch.rand(batch_size, 1, 1, 1, device=device) * 0.35
    )

    # Smooth uneven illumination represents directional workshop lighting and vignetting.
    x = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width)
    y = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1)
    slope_x = (torch.rand(batch_size, 1, 1, 1, device=device) - 0.5) * 0.30
    slope_y = (torch.rand(batch_size, 1, 1, 1, device=device) - 0.5) * 0.30
    vignette = 1.0 + slope_x * x + slope_y * y
    uneven_light = clean * vignette

    return torch.cat(
        [
            torch.clamp(cast, 0, 255),
            torch.clamp(exposure_contrast, 0, 255),
            torch.clamp(desaturated, 0, 255),
            torch.clamp(uneven_light, 0, 255),
        ],
        dim=0,
    )


# ---------------------------------------------------------
# GPU BATCH PROCESSOR
# ---------------------------------------------------------


def process_dataset_on_gpu(input_folder, device, batch_size=32):
    files = [
        f for f in os.listdir(input_folder) if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    print_status(
        f"Pushing {len(files)} raw images to RTX A2000 for batched CUDA processing..."
    )

    all_x, all_y = [], []

    for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        tensors = []

        # CPU only reads the raw bytes, nothing else
        for f in batch_files:
            try:
                img = io.read_image(os.path.join(input_folder, f))
                # Resize and ensure 3 channels
                img = img[:3, :, :]
                img = F_vision.resize(img, [256, 256], antialias=True)
                tensors.append(img)
            except Exception:
                continue

        if not tensors:
            continue

        # BLAST to GPU VRAM
        batch_rgb = torch.stack(tensors).to(device)

        with torch.no_grad():  # No gradients needed for preprocessing
            # 1. Extract Clean Features
            clean_features = extract_features_gpu(batch_rgb)

            # Dummy target logic for now (Target would normally come from the JSON report)
            # We are teaching it the identity function for clean images
            clean_labels = torch.zeros((len(batch_rgb), 4), device=device)
            clean_labels[:, 3] = 1.0  # Saturation gain = 1.0 (no change)

            # Create four independent photometric failures per clean image. Geometric
            # transforms are excluded because ColorMLP predicts only color controls.
            bad_batch = synthesize_factory_degradations(batch_rgb)

            # Extract features of the newly ruined images
            bad_features = extract_features_gpu(bad_batch)

            # THE DYNAMIC ANSWER KEY:
            # Instead of guessing, we mathematically calculate the exact shift required
            # to return the 'bad' features back to the 'clean' features.

            repeated_clean_features = clean_features.repeat(4, 1)
            l_shift = repeated_clean_features[:, 0] - bad_features[:, 0]
            a_shift = repeated_clean_features[:, 4] - bad_features[:, 4]
            b_shift = repeated_clean_features[:, 5] - bad_features[:, 5]
            sat_gain = repeated_clean_features[:, 6] / (bad_features[:, 6] + 1e-5)
            sat_gain = torch.clamp(sat_gain, 0.50, 1.75)

            # Stack into a [B, 4] tensor for the neural network
            bad_labels = torch.stack([l_shift, a_shift, b_shift, sat_gain], dim=1)

            all_x.append(clean_features)
            all_x.append(bad_features)
            all_y.append(clean_labels)
            all_y.append(bad_labels)
        if i % (batch_size * 2) == 0:
            print_status(f"GPU Preprocessed {i}/{len(files)} images...")

    # Flatten the results
    x_tensor = torch.cat(all_x, dim=0)
    y_tensor = torch.cat(all_y, dim=0)
    return x_tensor, y_tensor


# ---------------------------------------------------------
# MLP & TRAINING
# ---------------------------------------------------------


class ColorMLP(nn.Module):
    def __init__(self, input_dim=len(INPUT_FEATURES), output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def train_model(x_tensor, y_tensor, device):
    print_status(f"\nTraining on {len(x_tensor)} rows entirely inside VRAM...")

    # Normalize purely on GPU
    x_mean, x_std = x_tensor.mean(dim=0), x_tensor.std(dim=0).clamp(min=1e-6)
    y_mean, y_std = y_tensor.mean(dim=0), y_tensor.std(dim=0).clamp(min=1e-6)

    x_norm = (x_tensor - x_mean) / x_std
    y_norm = (y_tensor - y_mean) / y_std

    feature_idx = {name: idx for idx, name in enumerate(INPUT_FEATURES)}
    mean_sat = x_tensor[:, feature_idx["Mean_Sat"]]
    channel_gap = x_tensor[:, feature_idx["Channel_Gap"]]
    low_sat_ratio = x_tensor[:, feature_idx["Low_Sat_Ratio"]]
    mono_like = (
        (mean_sat < 22.0)
        | (low_sat_ratio > 0.82)
        | ((mean_sat < 35.0) & (channel_gap < 8.0))
    ).float()
    sample_weight = 1.0 + (1.6 * mono_like)

    dataset = TensorDataset(x_norm, y_norm, sample_weight)
    train_loader = DataLoader(dataset, batch_size=512, shuffle=True)

    model = ColorMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(1, 251):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y, batch_w in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            mse_per_row = ((pred - batch_y) ** 2).mean(dim=1)
            loss = (mse_per_row * batch_w).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 50 == 0:
            print_status(
                f"Epoch {epoch:03d} | train_loss={epoch_loss/len(train_loader):.6f}"
            )

    return model, (
        x_mean.cpu().numpy(),
        x_std.cpu().numpy(),
        y_mean.cpu().numpy(),
        y_std.cpu().numpy(),
    )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_status(f"HPC Pure GPU Pipeline Initiated -> {device}")

    x_tensor, y_tensor = process_dataset_on_gpu(DEFAULT_INPUT_FOLDER, device)
    model, norms = train_model(x_tensor, y_tensor, device)

    bundle = {
        "framework": "pytorch_native",
        "model_state_dict": model.state_dict(),
        "feature_names": INPUT_FEATURES,
        "x_mean": norms[0],
        "x_std": norms[1],
        "y_mean": norms[2],
        "y_std": norms[3],
    }

    with open(DEFAULT_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print_status(f"Saved Native GPU color model to {DEFAULT_MODEL_PATH}")


if __name__ == "__main__":
    main()
