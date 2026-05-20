# app.py
import shutil
import tempfile
import os
from pathlib import Path
from typing import Optional
import traceback
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import uvicorn

# Import your transcriber inference function
from src.transcriber.infer import infer as transcribe_infer
from src.transcriber.preprocess import load_config as load_preprocess_config

logger = logging.getLogger("drum_transcriber")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Drum Transcriber API")

# Serve a simple upload page
TEMPLATES_DIR = Path(__file__).parent / "templates"
if TEMPLATES_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")

# Load config once
CFG_PATH = None  # or path to your configs/default.yaml
CFG = load_preprocess_config(CFG_PATH)

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
MAX_UPLOAD_MB = 200  # adjust as needed

# Debug toggle: when True the endpoint returns full traceback in JSON (only use locally)
DEBUG_SHOW_TRACEBACK = True


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = TEMPLATES_DIR / "upload.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h3>Drum Transcriber</h3><p>Use POST /upload to send audio.</p>")


@app.post("/upload")
async def upload_and_transcribe(
    file: UploadFile = File(...),
    quantize: Optional[bool] = False,
    background_tasks: BackgroundTasks = None,
):
    # Basic validation
    filename = Path(file.filename)
    ext = filename.suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save to a secure temporary directory
    tmp_dir = Path(tempfile.mkdtemp(prefix="drum_trans_"))
    audio_path = tmp_dir / f"upload{ext}"
    out_midi = tmp_dir / (filename.stem + ".mid")

    try:
        # Stream upload to disk
        with audio_path.open("wb") as out_f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)

        # Run blocking inference in a threadpool to avoid blocking the event loop
        try:
            await run_in_threadpool(transcribe_infer, str(audio_path), str(out_midi), CFG_PATH)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Transcription failed:\n%s", tb)
            # schedule cleanup
            if background_tasks is not None:
                background_tasks.add_task(shutil.rmtree, tmp_dir, True)
            else:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            if DEBUG_SHOW_TRACEBACK:
                return JSONResponse(status_code=500, content={"detail": str(e), "traceback": tb})
            raise HTTPException(status_code=500, detail="Transcription failed")

        if not out_midi.exists():
            # schedule cleanup
            if background_tasks is not None:
                background_tasks.add_task(shutil.rmtree, tmp_dir, True)
            else:
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail="Transcription did not produce a MIDI file.")

        # Schedule cleanup after response is sent
        if background_tasks is not None:
            background_tasks.add_task(shutil.rmtree, tmp_dir, True)
        else:
            # If no background tasks provided, remove after returning (best-effort)
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass

        # Return the MIDI file as an attachment
        return FileResponse(path=str(out_midi), filename=out_midi.name, media_type="audio/midi")

    except HTTPException:
        # Re-raise HTTP exceptions unchanged
        raise
    except Exception as e:
        tb = traceback.format_exc()
        logger.exception("Unexpected error in upload endpoint:\n%s", tb)
        # ensure cleanup
        if background_tasks is not None:
            background_tasks.add_task(shutil.rmtree, tmp_dir, True)
        else:
            try:
                shutil.rmtree(tmp_dir)
            except Exception:
                pass
        if DEBUG_SHOW_TRACEBACK:
            return JSONResponse(status_code=500, content={"detail": str(e), "traceback": tb})
        raise HTTPException(status_code=500, detail="Internal server error")


if __name__ == "__main__":
    # Run with: python app.py
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
