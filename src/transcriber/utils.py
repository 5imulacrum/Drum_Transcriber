from typing import Iterable, List
import numpy as np

def seconds_to_frames(times: Iterable[float], sr: int, hop_length: int) -> List[int]:
    return [int(round(t * sr / hop_length)) for t in times]

def frames_to_seconds(frames: Iterable[int], sr: int, hop_length: int) -> List[float]:
    return [f * hop_length / sr for f in frames]

def ensure_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]
