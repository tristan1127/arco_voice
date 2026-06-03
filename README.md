# Arco

Arco is a local speech-to-text web app for high-quality Chinese, English, French, and multilingual transcription on macOS Apple Silicon.

## Requirements

- Python 3.11+
- ffmpeg
- macOS Apple Silicon recommended for Metal acceleration
- About 5 GB free disk space for the current local model cache

Install ffmpeg:

```bash
brew install ffmpeg
```

## Quick Start

```bash
bash scripts/run.sh
```

Then open:

```text
http://localhost:8000/ui
```

The first time you select a model, it will automatically download to the `checkpoints/huggingface/hub/` folder inside this project directory.

Do not commit `checkpoints/` to GitHub. It is intentionally ignored so the repository stays code-only and each machine can download its own local models.

## Model Sizes

| Model | Speed | Quality | Best for |
| --- | --- | --- | --- |
| `tiny` / `base` | Fastest | Low | Smoke tests and very rough drafts |
| `small` | Fast | Medium | Quick notes when speed matters |
| `medium` | Balanced | High | Faster everyday transcription |
| `large-v3` | Slower | Highest | Best multilingual accuracy |
| `turbo` | Very fast | High | Strong quality with lower latency |
| `qwen3-asr-0.6b`| Fast | High | Specially optimized for Chinese |

For Chinese recordings, try `qwen3-asr-0.6b` first. Keep `large-v3` as the stable multilingual fallback.

Current local cache examples:

| Cached model | Approx size |
| --- | ---: |
| `models--mlx-community--whisper-large-v3-mlx` | 2.9 GB |
| `models--Qwen--Qwen3-ASR-0.6B` | 1.8 GB |
| `models--Systran--faster-whisper-base` | 141 MB |

## Adding A Model

To add another model option, update:

1. `frontend/index.html` for the model dropdown option.
2. `backend/main.py` for `ALLOWED_MODELS`.
3. `backend/transcriber.py` for the loading/transcription branch.
4. `backend/requirements.txt` if the model needs another Python package.

Models from Hugging Face will use the project-local cache configured by `scripts/run.sh`.

## Supported Languages

Whisper supports 99 languages, including:

- **Chinese / 中文**: Mandarin, simplified and traditional text
- **English**
- **French / Français**
- Japanese, Korean, Spanish, German, Russian, Arabic, Italian, Portuguese, Hindi, and many more

## Apple Silicon

On M-series Macs, Arco auto-detects Apple Silicon and prefers `mlx-whisper` if it is installed. `scripts/run.sh` attempts to install it automatically, so transcription can use Metal GPU acceleration. If `mlx-whisper` is unavailable, Arco falls back to `faster-whisper` on CPU.

## Project Structure

```text
arco/
├── backend/
│   ├── main.py
│   ├── transcriber.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── scripts/
│   └── run.sh
├── .gitignore
└── README.md
```
