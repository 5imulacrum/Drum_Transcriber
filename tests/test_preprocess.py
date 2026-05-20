import numpy as np
from src.transcriber.preprocess import pick_onsets, estimate_velocity

def test_pick_onsets_merge():
    # synthetic onset envelope with two close peaks
    env = np.zeros(1000)
    env[100] = 1.0
    env[110] = 0.9
    sr = 1000
    hop = 1
    times = pick_onsets(env, sr, hop, threshold=0.5, merge_ms=20)
    assert len(times) == 1

def test_estimate_velocity():
    y = np.zeros(1000)
    y[500] = 1.0
    sr = 1000
    times = [0.5]
    vel = estimate_velocity(y, sr, times)
    assert vel.shape[0] == 1
    assert vel[0] >= 20
