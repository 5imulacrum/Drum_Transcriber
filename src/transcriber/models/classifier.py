from typing import List
import torch
import torch.nn as nn

class OnsetClassifier(nn.Module):
    def __init__(self, n_mels: int = 128, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, n_mels, T_window)
        x = x.unsqueeze(1)
        return self.net(x)
