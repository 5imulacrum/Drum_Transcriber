"""
src/transcriber/infer.py

Fixed inference entrypoint:
- Module-level imports for heavy libs (torch, librosa, numpy).
- No rebinding of module names inside functions.
- Proper indentation so velocities are computed.
- Logging and safe fallbacks for model loading.
- Returns the output MIDI path.
"""

from typing import Optional, List
import argparse
import os
import yaml
import logging

import numpy as np
import torch
import librosa

from .preprocess import load_audio, compute_mel, onset_envelope, pick_onsets, estimate_velocity, load_config
from .midi_utils import write_drum_midi, load_config as load_midi_config
from .models.onset_crnn import OnsetCRNN
from .models.classifier import OnsetClassifier

import traceback
from fastapi import HTTPException

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _select_device(cfg_device: Optional[str] = None) -> torch.device:
    if cfg_device:
        try:
            return torch.device(cfg_device)
        except Exception:
            logger.warning("Invalid device in config '%s', falling back to cuda/cpu auto", cfg_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def infer(audio_path: str, out_midi: str, cfg_path: str = None):
    """
    Run onset detection, classification (optional), and write a drum MIDI file.

    Parameters
    ----------
    audio_path : str
        Path to input audio file.
    out_midi : str
        Path to write output MIDI file.
    cfg_path : str, optional
        Path to YAML config used by load_config.
    """
    try:
        # Load configs
        cfg = load_config(cfg_path)
        midi_cfg = load_midi_config(cfg_path)

        sr = cfg['audio']['sr']
        hop = cfg['audio']['hop_length']

        # Load audio
        y, sr = load_audio(audio_path, sr=sr)

        # Compute features
        mel = compute_mel(y, sr, n_fft=cfg['audio']['n_fft'], hop_length=hop, n_mels=cfg['audio']['n_mels'])
        onset_env = onset_envelope(y, sr, hop_length=hop)

        # Device selection
        device = _select_device(cfg['model'].get('device', None))
        logger.info("Using device: %s", device)

        # Try model-based onset detection if checkpoint provided
        times: List[float] = []
        onset_ckpt = cfg['model'].get('onset_checkpoint')
        if onset_ckpt:
            try:
                model = OnsetCRNN(n_mels=cfg['audio']['n_mels']).to(device)
                model.load_state_dict(torch.load(onset_ckpt, map_location=device))
                model.eval()
                with torch.no_grad():
                    # mel expected shape (n_mels, T) -> add batch dim
                    inp = torch.from_numpy(mel).unsqueeze(0).float().to(device)
                    pred = model(inp)  # expected (1, T) or similar
                    pred = pred.squeeze(0).cpu().numpy()
                    times = pick_onsets(pred, sr, hop, threshold=cfg['onset']['threshold'],
                                        merge_ms=cfg['onset']['merge_ms'],
                                        prominence=cfg['onset']['peak_prominence'])
                    logger.info("Model-based onset detection found %d onsets", len(times))
            except Exception:
                logger.exception("Onset model failed; falling back to librosa onset detection")
                times = []

        # Fallback to librosa/onset_env based detection
        if not times:
            try:
                times = pick_onsets(onset_env, sr, hop, threshold=cfg['onset']['threshold'],
                                    merge_ms=cfg['onset']['merge_ms'],
                                    prominence=cfg['onset']['peak_prominence'])
                logger.info("Librosa-based onset detection found %d onsets", len(times))
            except Exception:
                logger.exception("Fallback onset detection failed; no onsets detected")
                times = []

        # Estimate velocities (ensure this runs after times is set)
        try:
            velocities = estimate_velocity(y, sr, times)
        except Exception:
            logger.exception("Velocity estimation failed; using default velocity 100")
            velocities = [100] * len(times)

        # Classification: try model-based classifier, otherwise heuristic
        classes: List[str] = []
        mapping = midi_cfg['midi']['mapping']  # dict mapping class name -> midi note

        cls_ckpt = cfg['model'].get('classifier_checkpoint')
        if cls_ckpt:
            try:
                cls_model = OnsetClassifier(n_mels=cfg['audio']['n_mels'], n_classes=len(mapping)).to(device)
                cls_model.load_state_dict(torch.load(cls_ckpt, map_location=device))
                cls_model.eval()

                # prepare windows around onsets (mel is (n_mels, T))
                windows = []
                win_frames = max(1, int(0.1 * sr / hop))  # 100ms window in frames
                for t in times:
                    center = int(round(t * sr / hop))
                    start = max(0, center - win_frames // 2)
                    end = start + win_frames
                    spec = mel[:, start:end]
                    if spec.shape[1] < win_frames:
                        pad = np.zeros((mel.shape[0], win_frames - spec.shape[1]))
                        spec = np.concatenate([spec, pad], axis=1)
                    windows.append(spec)

                if windows:
                    inp = torch.from_numpy(np.stack(windows)).float().to(device)
                    with torch.no_grad():
                        out = cls_model(inp)
                        out = out.cpu().numpy()
                    class_names = list(mapping.keys())
                    for row in out:
                        idx = int(row.argmax())
                        classes.append(class_names[idx])
                    logger.info("Classifier assigned %d classes", len(classes))
                else:
                    logger.info("No windows to classify; skipping classifier")
            except Exception:
                logger.exception("Classifier failed; falling back to heuristic")
                classes = []

        # Heuristic classification if classifier not used or failed
        if not classes:
            try:
                # compute spectral centroid per frame and map to classes
                centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]
                for t in times:
                    frame = int(round(t * sr / hop))
                    frame = min(frame, len(centroids) - 1) if len(centroids) > 0 else 0
                    c = float(centroids[frame]) if len(centroids) > 0 else 0.0
                    if c < 1500:
                        classes.append('kick')
                    elif c < 3000:
                        classes.append('snare')
                    else:
                        classes.append('hihat_closed')
                logger.info("Heuristic classification produced %d classes", len(classes))
            except Exception:
                logger.exception("Heuristic classification failed; defaulting to snare")
                classes = ['snare'] * len(times)

        # Build events and write MIDI
        events = []
        for t, cls, vel in zip(times, classes, velocities):
            note = mapping.get(cls, 35)
            events.append((float(t), int(note), int(vel)))

        try:
            write_drum_midi(events, out_midi, quantize=cfg['midi']['quantize'], grid=cfg['midi']['quantize_grid'])
            logger.info("Wrote %d events to %s", len(events), out_midi)
        except Exception:
            logger.exception("Failed to write MIDI file")
            raise

        return out_midi
    except Exception as exc:
        # log full traceback to server console / logs
        tb = traceback.format_exc()
        logger.error("infer() raised an exception:\n%s", tb)
        # re-raise so the FastAPI endpoint can capture it
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--config', default=None)
    args = parser.parse_args()
    infer(args.audio, args.out, args.config)
