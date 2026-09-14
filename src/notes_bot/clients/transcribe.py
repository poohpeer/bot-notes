"""Speech-to-text — see docs/architecture/03-ingest.md, "Ограничения
тяжёлого пути": model `small`, `compute_type=int8` (CPU-only).

A real `FasterWhisperClient` loads once per worker process and is reused
across jobs — same rationale as queue/tasks.py's cached engine/session
factory. Import of faster_whisper is deferred into first use so importing
this module (and everything that transitively imports it) never pays for
torch unless a transcription actually runs — mirrors
services/embeddings/model.py's SentenceTransformerModel.
"""

from __future__ import annotations

from typing import Protocol


class TranscriptionClient(Protocol):
    def transcribe(self, audio_path: str) -> str: ...


class FasterWhisperClient:
    def __init__(self, *, model_size: str = "small", compute_type: str = "int8") -> None:
        self._model_size = model_size
        self._compute_type = compute_type
        self._model = None

    def transcribe(self, audio_path: str) -> str:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self._model_size, compute_type=self._compute_type)

        segments, _info = self._model.transcribe(audio_path)
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())
