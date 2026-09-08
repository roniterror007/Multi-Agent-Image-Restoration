# training/train_safety_valve.py
from pathlib import Path

import torch
import torch.nn as nn


class SafetyValveClassifier(nn.Module):
    """A lightweight CNN to detect catastrophic failures (e.g., completely black image, massive artifacts)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 1), nn.Sigmoid()  # 1.0 = Catastrophic failure, 0.0 = Safe
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def train_safety_valve():
    print("Initializing Safety Valve Classifier training pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SafetyValveClassifier().to(device)

    out_path = Path(__file__).resolve().parent.parent / "models" / "safety_valve.pth"
    out_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"Saved initialized weights to {out_path}")


if __name__ == "__main__":
    train_safety_valve()
