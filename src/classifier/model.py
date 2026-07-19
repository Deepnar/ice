# src/classifier/model.py

import torch
import torch.nn as nn

class ICEClassifier(nn.Module):
    def __init__(self):
        super(ICEClassifier, self).__init__()
        # 384 = the slice384 MRL prefix of the native embedding (C17) —
        # retrained at native width by B1, which retires the slice.
        self.fc1 = nn.Linear(384, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 25)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x