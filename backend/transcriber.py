"""
transcriber.py — Whisper transcription engine

Supports:
  - faster-whisper (primary, CPU/CUDA)
  - mlx-whisper (optional, Apple Silicon Metal acceleration)

Auto-detects Apple Silicon and prefers mlx-whisper if installed.
"""

import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional


@dataclass
class Segment:
    start: float      # seconds
    end: float        # seconds
    text: str
    language: str


class Transcriber:
    QWEN3_ASR_06B = "qwen3-asr-0.6b"
    QWEN3_ASR_06B_REPO = "Qwen/Qwen3-ASR-0.6B"

    def __init__(self, model_size: str = "large-v3", device: str = "auto"):
        """
        Args:
            model_size: "tiny" | "base" | "small" | "medium" | "large-v3" | "turbo" | "qwen3-asr-0.6b"
            device: "auto" | "cpu" | "cuda" | "mps"
        """
        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.model = None
        self.backend = None  # "faster-whisper" | "mlx-whisper" | "mlx-qwen3-asr"
        self.mlx_whisper = None
        self.qwen_session = None

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
        if self.backend is not None:
            return

        if self.model_size == self.QWEN3_ASR_06B:
            from mlx_qwen3_asr import Session
            self.qwen_session = Session(model=self.QWEN3_ASR_06B_REPO)
            self.backend = "mlx-qwen3-asr"
            return

        # Try mlx-whisper first on Apple Silicon for Metal GPU acceleration.
        if self.device == "mps":
            try:
                import mlx_whisper
                self.mlx_whisper = mlx_whisper
                self.backend = "mlx-whisper"
                # mlx-whisper downloads model on first use.
                return
            except ImportError:
                pass

        # faster-whisper (CPU with OpenBLAS, or CUDA).
        from faster_whisper import WhisperModel
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = WhisperModel(
            self.model_size,
            device="cpu" if self.device == "mps" else self.device,
            compute_type=compute_type,
            num_workers=4,
            cpu_threads=8,
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

        if self.backend == "mlx-qwen3-asr":
            yield from self._transcribe_qwen3(audio_path, language)
        elif self.backend == "mlx-whisper":
            yield from self._transcribe_mlx(audio_path, language, task)
        else:
            yield from self._transcribe_faster(audio_path, language, task)

    def _transcribe_faster(self, audio_path, language, task):
        segments, info = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
            condition_on_previous_text=True,
        )
        detected_lang = info.language
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue
            yield Segment(
                start=seg.start,
                end=seg.end,
                text=text,
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
            text = seg["text"].strip()
            if not text:
                continue
            yield Segment(
                start=seg["start"],
                end=seg["end"],
                text=text,
                language=detected_lang,
            )

    def _transcribe_qwen3(self, audio_path, language):
        result = self.qwen_session.transcribe(
            audio_path,
            language=language,
            return_timestamps=False,
        )
        text = result.text.strip()
        if not text:
            return
        yield Segment(
            start=0.0,
            end=_probe_duration(audio_path),
            text=text,
            language=result.language,
        )


def _probe_duration(audio_path: str) -> float:
    path = Path(audio_path)
    if path.suffix.lower() == ".wav":
        try:
            import wave
            with wave.open(str(path), "rb") as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                return frames / rate if rate else 0.0
        except (OSError, wave.Error):
            pass

    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            check=True,
            text=True,
        )
        return float(result.stdout.strip() or 0.0)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0.0
