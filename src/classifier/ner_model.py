"""Micro NER model: 384‑dim input → 3‑class output with hidden layers."""

import torch
import torch.nn as nn

class MicroNER(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 384 = the slice384 MRL prefix of the native embedding (C17) —
            # an A9-gated retrain (or GLiNER swap) retires the slice.
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.net(x)