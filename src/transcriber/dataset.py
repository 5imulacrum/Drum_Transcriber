from typing import List, Optional, Tuple
import numpy as np
import os
from .preprocess import load_audio, compute_mel, onset_envelope
import torch
from torch.utils.data import Dataset

class DrumDataset(Dataset):
    """
    Minimal dataset that expects pairs of (audio_path, annotations)
    annotations: list of (time_seconds, class_name)
    This class focuses on preprocessing and framewise label generation.
    """
    def __init__(self, items: List[Tuple[str, List[Tuple[float, str]]]], classes: List[str], sr: int = 22050, hop_length: int = 512, n_mels: int = 128):
        self.items = items
        self.classes = classes
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, annotations = self.items[idx]
        y, sr = load_audio(audio_path, sr=self.sr)
        mel = compute_mel(y, sr, hop_length=self.hop_length, n_mels=self.n_mels)
        # framewise labels
        n_frames = mel.shape[1]
        labels = np.zeros((len(self.classes), n_frames), dtype=np.float32)
        for t, cls in annotations:
            frame = int(round(t * sr / self.hop_length))
            if 0 <= frame < n_frames:
                if cls in self.classes:
                    labels[self.classes.index(cls), frame] = 1.0
        return torch.from_numpy(mel).float(), torch.from_numpy(labels).float()
