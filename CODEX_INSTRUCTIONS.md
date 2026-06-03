# Arco — Speech-to-Text App: Codex Build Instructions

## Project Overview

Build a local speech-to-text web application called **Arco** that runs on macOS (Apple Silicon M-series).
- **Model**: `faster-whisper` with `large-v3` (best quality, supports Chinese/English/French + 99 languages)
- **Backend**: Python 3.11+ with FastAPI, streaming via Server-Sent Events
- **Frontend**: Single HTML file (no build step), dark minimal UI with Sora + DM Mono fonts
- **Target**: Apple M4 Mac, uses CoreML / Metal acceleration via `mlx-whisper` fallback

---

## Task 1 — Project Scaffold

Create the following directory structure exactly:

```
arco/
├── backend/
│   ├── main.py          # FastAPI app
│   ├── transcriber.py   # Whisper wrapper
│   └── requirements.txt
├── frontend/
│   └── index.html       # Complete UI (single file)
├── scripts/
│   └── run.sh           # One-command launcher
├── .gitignore
└── README.md
```

---

## Task 2 — Backend: `backend/requirements.txt`

```
fastapi==0.111.0
uvicorn[standard]==0.29.0
faster-whisper==1.0.3
python-multipart==0.0.9
sse-starlette==2.1.0
torch==2.3.0
```

> Note: On Apple Silicon, `faster-whisper` uses CPU with OpenBLAS. For best M4 performance,
> also install `mlx-whisper` as an optional accelerator (see transcriber.py).

---

## Task 3 — Backend: `backend/transcriber.py`

Create a `Transcriber` class with these exact specs:

```python
"""
transcriber.py — Whisper transcription engine

Supports:
  - faster-whisper (primary, CPU/CUDA)
  - mlx-whisper (optional, Apple Silicon Metal acceleration)

Auto-detects Apple Silicon and prefers mlx-whisper if installed.
"""

import os
import platform
from dataclasses import dataclass
from typing import Generator, Optional

@dataclass
class Segment:
    start: float      # seconds
    end: float        # seconds
    text: str
    language: str

class Transcriber:
    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        """
        Args:
            model_size: "tiny" | "base" | "small" | "medium" | "large-v3" | "turbo"
            device: "auto" | "cpu" | "cuda" | "mps"
        """
        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.model = None
        self.backend = None  # "faster-whisper" | "mlx-whisper"

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        is_apple_silicon = (
            platform.system() == "Darwin" and
            platform.machine() == "arm64"
        )
        return "mps" if is_apple_silicon else "cpu"

    def load(self):
        """Load model. Call once before transcribing."""
        # Try mlx-whisper first on Apple Silicon for Metal GPU acceleration
        if self.device == "mps":
            try:
                import mlx_whisper
                self.mlx_whisper = mlx_whisper
                self.backend = "mlx-whisper"
                # mlx-whisper downloads model on first use
                return
            except ImportError:
                pass  # Fall through to faster-whisper

        # faster-whisper (CPU with OpenBLAS, or CUDA)
        from faster_whisper import WhisperModel
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            self.model_size,
            device="cpu" if self.device == "mps" else self.device,
            compute_type=compute_type,
            num_workers=4,
        )
        self.backend = "faster-whisper"

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe",  # "transcribe" | "translate"
    ) -> Generator[Segment, None, None]:
        """
        Yields Segment objects as they are transcribed.
        language: ISO 639-1 code like "zh", "en", "fr", or None for auto-detect.
        task: "transcribe" keeps original language; "translate" converts to English.
        """
        if self.model is None and self.backend != "mlx-whisper":
            self.load()

        if self.backend == "mlx-whisper":
            yield from self._transcribe_mlx(audio_path, language, task)
        else:
            yield from self._transcribe_faster(audio_path, language, task)

    def _transcribe_faster(self, audio_path, language, task):
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=5,
            vad_filter=True,          # skip silence
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
            condition_on_previous_text=True,
        )
        detected_lang = info.language
        for seg in segments:
            yield Segment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip(),
                language=detected_lang,
            )

    def _transcribe_mlx(self, audio_path, language, task):
        result = self.mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=f"mlx-community/whisper-{self.model_size}-mlx",
            language=language,
            task=task,
        )
        detected_lang = result.get("language", "unknown")
        for seg in result.get("segments", []):
            yield Segment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                language=detected_lang,
            )
```

---

## Task 4 — Backend: `backend/main.py`

Create a FastAPI app with these endpoints:

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Redirect to `/ui` |
| GET | `/ui` | Serve `frontend/index.html` |
| POST | `/transcribe` | Upload audio, stream SSE transcript |
| GET | `/health` | Return `{"status": "ok", "backend": "..."}` |

### `/transcribe` SSE stream format

Each event is JSON on a single line:
```
data: {"type": "segment", "start": 3.2, "end": 7.1, "text": "你好世界", "language": "zh"}
data: {"type": "progress", "percent": 42, "message": "transcribing segment 8/19..."}
data: {"type": "done", "total_segments": 87, "language": "zh", "duration_seconds": 5497}
data: {"type": "error", "message": "..."}
```

### Request form fields

- `file`: audio file (multipart upload)
- `language`: string, e.g. `"zh"` or `"auto"` (optional, default `null`)
- `model`: string, e.g. `"large-v3"` (optional, default `"large-v3"`)
- `task`: `"transcribe"` | `"translate"` (optional, default `"transcribe"`)
- `output_format`: `"txt"` | `"srt"` | `"vtt"` | `"json"` (optional, default `"txt"`)

### CORS

Allow all origins (localhost development).

### Temporary file handling

Save uploaded file to `tempfile.NamedTemporaryFile` with correct extension, delete after transcription.

### Model caching

Keep a single `Transcriber` instance in app state (`app.state.transcriber`), loaded on startup.

---

## Task 5 — Frontend: `frontend/index.html`

**Design spec** — implement exactly as described:

### Visual style
- Dark background `#0E0E11`
- Warm accent `#C4A882` (sand/terracotta)
- Font: `Sora` (UI) + `DM Mono` (code/stats/timestamps) from Google Fonts
- No gradients, no heavy shadows — flat surfaces with subtle borders
- Two-panel layout: left = controls, right = transcript

### Layout (two columns, 50/50)

**Left panel:**
1. Upload dropzone (drag-and-drop + click to browse)
2. File info card (name, duration, size)
3. Language chips — multi-select: 中文, English, Français, Auto + others
4. Config dropdowns: Model, Device, Task, Output format
5. Animated waveform bars (CSS animation, decorative)
6. "Start transcription" button
7. Progress bar with step label + percentage

**Right panel:**
1. Section label "Transcript" + action buttons (copy, download, clear)
2. Stats row: word count, processed duration, detected language
3. Scrollable transcript area — monospace, segments appear streaming in real time
4. Each segment prefixed with timestamp `[HH:MM:SS]`

### JavaScript behavior

**Upload:**
- Drag-and-drop onto dropzone
- Click to open file picker
- Show file name + duration + size after selection

**Transcription:**
- POST file to `/transcribe` with form data
- Read SSE stream from response
- Append each `segment` event to transcript area with timestamp
- Update progress bar from `progress` events
- On `done`: show stats, enable copy/download

**Copy button:** Copy full transcript text to clipboard

**Download button:** Download as `.txt` file named `[original_filename]_transcript.txt`

**Output format selector:** When `srt` is selected, format segments as:
```
1
00:00:03,200 --> 00:00:07,100
你好世界

```

### No build step
Single HTML file. Load all libraries via CDN:
- Google Fonts (Sora, DM Mono)
- Tabler Icons webfont from cdnjs

---

## Task 6 — `scripts/run.sh`

```bash
#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "🎙 Arco — Speech to Text"
echo "========================"

# Check Python 3.11+
python3 --version

# Create venv if missing
if [ ! -d ".venv" ]; then
  echo "→ Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

# Install dependencies
echo "→ Installing dependencies..."
pip install -q -r backend/requirements.txt

# Optional: mlx-whisper for Apple Silicon acceleration
if [[ "$(uname -m)" == "arm64" ]]; then
  echo "→ Apple Silicon detected. Installing mlx-whisper for Metal GPU acceleration..."
  pip install -q mlx-whisper 2>/dev/null || echo "  (mlx-whisper install failed, falling back to CPU)"
fi

echo ""
echo "✓ Starting server at http://localhost:8000"
echo "  Open http://localhost:8000/ui in your browser"
echo ""

uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Make it executable: `chmod +x scripts/run.sh`

---

## Task 7 — `.gitignore`

```
.venv/
__pycache__/
*.pyc
*.pyo
tmp/
uploads/
*.mp3
*.wav
*.m4a
*.flac
.DS_Store
*.egg-info/
dist/
build/
```

---

## Task 8 — `README.md`

Include:
1. One-line description
2. Requirements (Python 3.11+, ffmpeg)
3. Quick start: `bash scripts/run.sh` then open `http://localhost:8000/ui`
4. Model size comparison table (tiny/small/medium/large-v3/turbo) with speed and quality ratings
5. Supported languages list (highlight Chinese, English, French)
6. Apple Silicon note: auto-uses Metal GPU via mlx-whisper if available
7. ffmpeg install: `brew install ffmpeg`

---

## Task 9 — Testing

After implementation:

1. Run `bash scripts/run.sh`
2. Open `http://localhost:8000/ui`
3. Upload a 5-minute MP3 with Chinese speech
4. Verify transcript appears streaming in real time
5. Verify copy and download buttons work
6. Test with `?model=medium` to confirm model switching

---

## Implementation Notes

- **Audio chunking**: `faster-whisper` handles long audio natively via its internal VAD — no manual chunking needed
- **Chinese accuracy**: `large-v3` with `language="zh"` is best; set `condition_on_previous_text=True`
- **VAD filter**: always enable `vad_filter=True` to skip silence, dramatically speeds up 90-min files
- **File size**: ffmpeg is NOT required in backend (faster-whisper uses ctranslate2's built-in ffmpeg bindings); just ensure `ffmpeg` is in PATH on macOS (`brew install ffmpeg`)
- **CORS**: must allow `*` for local dev since browser and server are both localhost
- **Streaming**: use `sse-starlette` for clean SSE, not raw StreamingResponse
- **Model download**: first run downloads ~3GB for large-v3 to `~/.cache/huggingface/hub/`
