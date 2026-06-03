#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "🎙 Arco — Speech to Text"
echo "========================"

ARCO_CONDA_ENV="${ARCO_CONDA_ENV:-get-note}"
ARCO_HOST="${ARCO_HOST:-127.0.0.1}"
ARCO_PORT="${ARCO_PORT:-8000}"
ARCO_MODEL_CACHE_DIR="${ARCO_MODEL_CACHE_DIR:-$PWD/checkpoints/huggingface}"

export HF_HOME="$ARCO_MODEL_CACHE_DIR"
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
mkdir -p "$HF_HUB_CACHE"

echo "→ Model cache: $HF_HUB_CACHE"

if command -v conda >/dev/null 2>&1 && conda env list | awk '{print $1}' | grep -qx "$ARCO_CONDA_ENV"; then
  echo "→ Using conda environment: $ARCO_CONDA_ENV"
  PYTHON_CMD=(conda run -n "$ARCO_CONDA_ENV" python)
  PIP_CMD=(conda run -n "$ARCO_CONDA_ENV" python -m pip)
  UVICORN_CMD=(conda run -n "$ARCO_CONDA_ENV" python -m uvicorn)
else
  echo "→ Conda environment '$ARCO_CONDA_ENV' not found. Using local .venv."

  if command -v python3.11 >/dev/null 2>&1; then
    SYSTEM_PYTHON="python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    SYSTEM_PYTHON="python3"
  else
    echo "Python 3.11+ is required."
    exit 1
  fi

  "$SYSTEM_PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required. Install Python 3.11 or set ARCO_CONDA_ENV=get-note.")
PY

  if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    "$SYSTEM_PYTHON" -m venv .venv
  fi

  source .venv/bin/activate
  PYTHON_CMD=(python)
  PIP_CMD=(python -m pip)
  UVICORN_CMD=(python -m uvicorn)
fi

"${PYTHON_CMD[@]}" --version

echo "→ Checking dependencies..."
MISSING=$("${PYTHON_CMD[@]}" - <<'PY'
import importlib.util

modules = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "faster-whisper": "faster_whisper",
    "mlx-qwen3-asr": "mlx_qwen3_asr",
    "python-multipart": "multipart",
    "sse-starlette": "sse_starlette",
    "torch": "torch",
}

missing = [package for package, module in modules.items() if importlib.util.find_spec(module) is None]
print(" ".join(missing))
PY
)

if [ -n "$MISSING" ]; then
  echo "→ Installing missing dependencies: $MISSING"
  "${PIP_CMD[@]}" install -q -r backend/requirements.txt
else
  echo "✓ Dependencies ready"
fi

if [[ "$(uname -m)" == "arm64" ]]; then
  if "${PYTHON_CMD[@]}" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('mlx_whisper') else 1)" >/dev/null 2>&1; then
    echo "✓ mlx-whisper ready for Metal GPU acceleration"
  else
    echo "→ Apple Silicon detected. Installing mlx-whisper for Metal GPU acceleration..."
    "${PIP_CMD[@]}" install -q mlx-whisper 2>/dev/null || echo "  (mlx-whisper install failed, falling back to CPU)"
  fi
fi

echo ""
echo "✓ Starting server at http://${ARCO_HOST}:${ARCO_PORT}"
echo "  Open http://localhost:${ARCO_PORT}/ui in your browser"
echo ""

"${UVICORN_CMD[@]}" backend.main:app --host "$ARCO_HOST" --port "$ARCO_PORT" --reload
