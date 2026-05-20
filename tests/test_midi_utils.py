import os
from src.transcriber.midi_utils import write_drum_midi

def test_write_midi(tmp_path):
    out = tmp_path / "test.mid"
    events = [(0.0, 36, 100), (0.5, 38, 90)]
    write_drum_midi(events, str(out))
    assert out.exists()
    assert out.stat().st_size > 0
