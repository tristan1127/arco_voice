"""
main.py — Arco FastAPI backend

Endpoints:
  GET  /            -> redirect to /ui
  GET  /ui          -> serve frontend/index.html
  POST /transcribe  -> stream transcript segments as Server-Sent Events
  GET  /health      -> status check
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse

try:
    from .transcriber import Transcriber
except ImportError:  # Allows `cd backend && uvicorn main:app` during local debugging.
    from transcriber import Transcriber


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".mp4", ".webm"}
ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3", "turbo", "qwen3-asr-0.6b"}
ALLOWED_TASKS = {"transcribe", "translate"}
ALLOWED_FORMATS = {"txt", "srt", "vtt", "json"}
ALLOWED_DEVICES = {"auto", "cpu", "cuda", "mps"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_size = os.environ.get("ARCO_MODEL", "large-v3")
    device = os.environ.get("ARCO_DEVICE", "auto")
    app.state.transcriber = Transcriber(model_size=model_size, device=device)
    app.state.transcriber_lock = asyncio.Lock()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, app.state.transcriber.load)
    yield


app = FastAPI(title="Arco", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/ui")


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    html_path = FRONTEND_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>frontend/index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/health")
def health(request: Request):
    transcriber: Transcriber = request.app.state.transcriber
    return {
        "status": "ok",
        "backend": transcriber.backend or "not loaded",
        "model": transcriber.model_size,
        "device": transcriber.device,
    }


def _sse(payload: dict) -> dict:
    return {"data": json.dumps(payload, ensure_ascii=False)}


def _clean_form_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


async def _get_transcriber(
    request: Request,
    model_size: str,
    device: str,
) -> Transcriber:
    async with request.app.state.transcriber_lock:
        transcriber: Transcriber = request.app.state.transcriber
        if transcriber.model_size == model_size and transcriber.device == Transcriber(model_size, device)._resolve_device(device):
            return transcriber

        transcriber = Transcriber(model_size=model_size, device=device)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, transcriber.load)
        request.app.state.transcriber = transcriber
        return transcriber


@app.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile,
    language: Optional[str] = Form(None),
    model: str = Form("large-v3"),
    device: str = Form("auto"),
    task: str = Form("transcribe"),
    output_format: str = Form("txt"),
):
    ext = Path(file.filename or "audio.mp3").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return JSONResponse({"error": f"Unsupported file type: {ext}"}, status_code=400)

    model_size = _clean_form_value(model) or "large-v3"
    device_name = _clean_form_value(device) or "auto"
    task_name = _clean_form_value(task) or "transcribe"
    requested_format = _clean_form_value(output_format) or "txt"
    lang = _clean_form_value(language)
    if lang == "auto":
        lang = None

    if model_size not in ALLOWED_MODELS:
        return JSONResponse({"error": f"Unsupported model: {model_size}"}, status_code=400)
    if device_name not in ALLOWED_DEVICES:
        return JSONResponse({"error": f"Unsupported device: {device_name}"}, status_code=400)
    if task_name not in ALLOWED_TASKS:
        return JSONResponse({"error": f"Unsupported task: {task_name}"}, status_code=400)
    if requested_format not in ALLOWED_FORMATS:
        return JSONResponse({"error": f"Unsupported output_format: {requested_format}"}, status_code=400)

    transcriber = await _get_transcriber(request, model_size, device_name)

    suffix = ext or ".mp3"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        await file.seek(0)
        shutil.copyfileobj(file.file, tmp)

    audio_size = os.path.getsize(tmp_path)

    async def event_generator():
        started_at = time.monotonic()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        segment_count = 0
        total_end = 0.0
        detected_language = lang or "auto"

        def run_transcription():
            try:
                for seg in transcriber.transcribe(tmp_path, language=lang, task=task_name):
                    loop.call_soon_threadsafe(queue.put_nowait, ("segment", seg))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))

        worker = loop.run_in_executor(None, run_transcription)

        try:
            yield _sse({
                "type": "progress",
                "percent": 1,
                "message": f"loaded {audio_size // 1024 // 1024} MB, preparing model...",
            })

            while True:
                if await request.is_disconnected():
                    break

                try:
                    kind, payload = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    elapsed = int(time.monotonic() - started_at)
                    yield _sse({
                        "type": "progress",
                        "percent": min(95, 5 + elapsed // 20),
                        "message": "transcribing long audio...",
                    })
                    continue

                if kind == "segment":
                    segment_count += 1
                    total_end = max(total_end, float(payload.end))
                    detected_language = payload.language or detected_language

                    yield _sse({
                        "type": "segment",
                        "start": round(float(payload.start), 2),
                        "end": round(float(payload.end), 2),
                        "text": payload.text,
                        "language": detected_language,
                    })

                    # Without decoding duration up front, progress is an honest soft estimate.
                    percent = min(95, 5 + segment_count)
                    yield _sse({
                        "type": "progress",
                        "percent": percent,
                        "message": f"transcribing segment {segment_count}...",
                    })

                elif kind == "done":
                    yield _sse({
                        "type": "done",
                        "total_segments": segment_count,
                        "language": detected_language,
                        "duration_seconds": round(total_end),
                    })
                    break

                elif kind == "error":
                    yield _sse({"type": "error", "message": payload})
                    break

            await worker
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return EventSourceResponse(event_generator())
