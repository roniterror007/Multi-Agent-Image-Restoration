# core/batch_process.py
import numpy as np
import torch
from torch import nn


# --- 1. THE HYBRID CONCEPT BOTTLENECK MODEL ---
class HybridConceptMLP(nn.Module):
    def __init__(self, input_dim=16):
        super().__init__()

        # Step 1: Predict 8 Human-Readable Visual Concepts
        self.concept_encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Sigmoid(),  # Constrain concepts to [0.0, 1.0]
        )

        # Step 2: Constrained Shift Calculator
        self.shift_head = nn.Sequential(
            nn.Linear(8, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [L_shift, a_shift, b_shift, sat_gain]
        )

    def forward(self, x):
        # 1. Extract visual concepts
        concepts = self.concept_encoder(x)

        # Concept Index Mapping:
        # 0: warm_cast, 1: cool_cast, 2: green_cast, 3: magenta_cast
        # 4: underexposure, 5: highlight_clipping, 6: low_contrast
        # 7: metallic_neutrality

        # 2. Predict base shifts
        raw_shifts = self.shift_head(concepts)
        l_shift, a_shift, b_shift, sat_gain = (
            raw_shifts[:, 0],
            raw_shifts[:, 1],
            raw_shifts[:, 2],
            raw_shifts[:, 3],
        )

        # 3. Apply Hard Monotonic Constraints (Physics Overrides)
        metallic_neutrality = concepts[:, 7]
        chroma_dampening = 1.0 - metallic_neutrality  # 0.0 when fully metallic

        # Force chroma shifts to 0 if the part is identified as metal
        a_shift = a_shift * chroma_dampening
        b_shift = b_shift * chroma_dampening

        # 4. Apply safety clipping boundaries
        l_shift = torch.clamp(l_shift, -40.0, 40.0)
        a_shift = torch.clamp(a_shift, -25.0, 25.0)
        b_shift = torch.clamp(b_shift, -25.0, 25.0)
        sat_gain = torch.clamp(sat_gain, 0.5, 1.5)

        final_shifts = torch.stack([l_shift, a_shift, b_shift, sat_gain], dim=1)
        return final_shifts, concepts
