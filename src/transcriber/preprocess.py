from typing import Tuple, Optional
import numpy as np
import librosa
import yaml
import os

def load_config(path: Optional[str] = None) -> dict:
    cfg_path = path or os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yaml')
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def load_audio(path: str, sr: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=sr, mono=mono)
    if y.ndim > 1:
        y = librosa.to_mono(y)
    return y, sr

def compute_mel(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    return S_db

def onset_envelope(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

def pick_onsets(onset_env: np.ndarray, sr: int, hop_length: int, threshold: float = 0.3, merge_ms: int = 40, prominence: float = 0.1) -> np.ndarray:
    from scipy.signal import find_peaks
    peaks, props = find_peaks(onset_env, height=threshold, prominence=prominence)
    times = peaks * hop_length / sr
    merged = []
    merge_s = merge_ms / 1000.0
    for t in times:
        if not merged or t - merged[-1] > merge_s:
            merged.append(t)
    return np.array(merged)

def estimate_velocity(y: np.ndarray, sr: int, times: np.ndarray, window_ms: int = 50) -> np.ndarray:
    velocities = []
    half = int(sr * window_ms / 1000 / 2)
    for t in times:
        center = int(round(t * sr))
        start = max(0, center - half)
        end = min(len(y), center + half)
        energy = float(np.sqrt(np.mean(y[start:end] ** 2) + 1e-9))
        vel = int(np.clip((energy / 0.1) * 127, 20, 127))
        velocities.append(vel)
    return np.array(velocities, dtype=int)
