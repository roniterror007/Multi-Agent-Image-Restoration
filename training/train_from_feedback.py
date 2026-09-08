# training/train_from_feedback.py
import json
import os
import pickle
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Adjust paths based on the new repository structure
BASE_DIR = Path(__file__).resolve().parent.parent
FEEDBACK_FILE = BASE_DIR / "data" / "feedback" / "feedback.jsonl"
BASE_MODEL_PATH = BASE_DIR / "color_predictor.pkl"
OUTPUT_MODEL_PATH = BASE_DIR / "color_predictor_feedback.pkl"

# Hardcoded target mappings for categorical issues
TARGETS = {
    "yellow_cast": [0.0, 0.0, -10.0, 1.0],  # Reduce yellow (b*)
    "red_cast": [0.0, -10.0, 0.0, 1.0],  # Reduce red (a*)
    "too_dark": [15.0, 0.0, 0.0, 1.0],  # Increase Lightness (L*)
    "too_bright": [-15.0, 0.0, 0.0, 1.0],  # Decrease Lightness (L*)
    "under_saturated": [0.0, 0.0, 0.0, 1.3],  # Boost saturation gain
}


class ColorMLP(nn.Module):
    """Must match the architecture defined in core/batch_process.py"""

    def __init__(self, input_dim=16, output_dim=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.network(x)


def load_and_sanitize_feedback():
    """Reads feedback.jsonl and backfills legacy feature vectors."""
    if not FEEDBACK_FILE.exists():
        print(f"No feedback found at {FEEDBACK_FILE}")
        return [], []

    x_data, y_data = [], []
    valid_count = 0

    with open(FEEDBACK_FILE, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("decision") != "rejected":
                    continue

                features = record.get("features", [])
                issue = record.get("issues", ["other"])[0]  # Grab primary issue

                if issue not in TARGETS:
                    continue

                # Backfill legacy 7-feature or 12-feature vectors to 16 dimensions
                if len(features) < 16:
                    features = features + [0.0] * (16 - len(features))
                elif len(features) > 16:
                    features = features[:16]

                # Sanitize NaNs
                features = [
                    0.0 if str(v).lower() == "nan" else float(v) for v in features
                ]

                x_data.append(features)
                y_data.append(TARGETS[issue])
                valid_count += 1
            except Exception as e:
                continue

    print(f"Loaded {valid_count} valid rejected feedback rows.")
    return x_data, y_data


def train_feedback_model():
    x_data, y_data = load_and_sanitize_feedback()
    if len(x_data) < 20:
        print("Need at least 20 feedback rows to train. Aborting.")
        return

    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    y_tensor = torch.tensor(y_data, dtype=torch.float32)

    # Load baseline normalizations
    with open(BASE_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    x_mean = torch.tensor(bundle["x_mean"], dtype=torch.float32)
    x_std = torch.tensor(bundle["x_std"], dtype=torch.float32)

    x_norm = (x_tensor - x_mean) / x_std
    dataset = TensorDataset(x_norm, y_tensor)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = ColorMLP(input_dim=16)
    model.load_state_dict(bundle["model_state_dict"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(120):  # Fine-tune for 120 epochs
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    # Save the fine-tuned bundle
    bundle["model_state_dict"] = model.state_dict()
    with open(OUTPUT_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"Successfully saved fine-tuned model to {OUTPUT_MODEL_PATH}")


if __name__ == "__main__":
    train_feedback_model()
