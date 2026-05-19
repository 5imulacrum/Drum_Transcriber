from typing import List, Dict, Tuple
import mido
import yaml
import os

def load_config(path: str = None) -> dict:
    cfg_path = path or os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'default.yaml')
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)

def class_to_midi_note(class_name: str, mapping: Dict[str, int]) -> int:
    return mapping.get(class_name, 35)

def write_drum_midi(events: List[Tuple[float, int, int]], out_path: str, tempo_bpm: int = 120, quantize: bool = False, grid: float = 0.025):
    """
    events: list of (time_seconds, midi_note, velocity)
    """
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo_bpm)))
    ticks_per_beat = mid.ticks_per_beat
    last_tick = 0
    for t, note, vel in sorted(events, key=lambda x: x[0]):
        if quantize:
            t = round(t / grid) * grid
        tick = mido.second2tick(t, ticks_per_beat, mido.bpm2tempo(tempo_bpm))
        delta = int(round(tick - last_tick))
        if delta < 0:
            delta = 0
        track.append(mido.Message('note_on', channel=9, note=int(note), velocity=int(vel), time=delta))
        track.append(mido.Message('note_off', channel=9, note=int(note), velocity=0, time=int(round(ticks_per_beat * 0.05))))
        last_tick = tick + int(round(ticks_per_beat * 0.05))
    mid.save(out_path)
