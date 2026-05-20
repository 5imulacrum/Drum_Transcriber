# debug_infer.py
from src.transcriber.infer import infer
import sys, traceback

audio = r"C:\path\to\small.wav"   # replace with a small test WAV
out = r"C:\path\to\out.mid"
cfg = None                        # or path to your config

try:
    infer(audio, out, cfg)
    print("Success, wrote:", out)
except Exception:
    traceback.print_exc()
    sys.exit(1)
