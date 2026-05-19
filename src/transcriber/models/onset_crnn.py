from typing import Optional
import torch
import torch.nn as nn

class OnsetCRNN(nn.Module):
    def __init__(self, n_mels: int = 128, hidden: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(16, 32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((1,2)),
        )
        conv_out = (n_mels // 4) * 32
        self.rnn = nn.GRU(input_size=conv_out, hidden_size=hidden, num_layers=1, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden*2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, n_mels, T)
        x = x.unsqueeze(1)  # (B,1,n_mels,T)
        x = self.conv(x)    # (B, C, n_mels', T')
        b, c, m, t = x.shape
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T', C, m)
        x = x.view(b, t, c * m)
        out, _ = self.rnn(x)
        out = self.fc(out)  # (B, T', 1)
        return out.squeeze(-1)
