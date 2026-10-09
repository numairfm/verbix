from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

PROVISION_CMD = "uv run scripts/fetch_voice_models.py"

REQUIRED_FILES = (
    "duration_predictor.int8.onnx",
    "text_encoder.int8.onnx",
    "vector_estimator.int8.onnx",
    "vocoder.int8.onnx",
    "tts.json",
    "unicode_indexer.bin",
    "voice.bin",
)

VOICE_MAP = {
    "M1": 0,
    "M2": 1,
    "M3": 2,
    "M4": 3,
    "M5": 4,
    "F1": 5,
    "F2": 6,
    "F3": 7,
    "F4": 8,
    "F5": 9,
}


def _resolve_threads() -> int:
    n = os.cpu_count() or 1
    return min(4, max(1, n))


def _resample_f32(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Resamples float32 mono audio using linear interpolation."""
    if from_rate == to_rate or samples.size == 0:
        return samples
    n_out = round(len(samples) * to_rate / from_rate)
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    pos = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(pos, np.arange(len(samples)), samples).astype(np.float32)


def _trim_silence(
    samples: np.ndarray,
    rate: int = 24000,
    threshold: float = 0.01,
    pad_ms: int = 30,
) -> np.ndarray:
    """Trims leading and trailing silence from synthesized audio, keeping a small natural pad."""
    if samples.size == 0:
        return samples
    mask = np.abs(samples) >= threshold
    if not np.any(mask):
        return samples
    pad = int(rate * (pad_ms / 1000.0))
    start_idx = max(0, int(np.argmax(mask)) - pad)
    end_idx = min(len(samples), len(samples) - int(np.argmax(mask[::-1])) + pad)
    return samples[start_idx:end_idx]


class SherpaTTS:
    """In-process Text-to-Speech engine using sherpa-onnx and Supertonic INT8."""

    def __init__(
        self,
        model_dir: Path | None = None,
        voice_id: str | int = "M1",
        threads: int | None = None,
        speed: float = 1.0,
    ) -> None:
        self.model_dir = model_dir
        self.voice_id = voice_id
        self.threads = threads if threads is not None else _resolve_threads()
        self.speed = speed
        self.sample_rate = 24000
        self.channels = 1
        self.audio_format = "f32"
        self._lock = asyncio.Lock()
        self._tts: Any = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._tts is not None

    async def load(self) -> None:
        """Loads the Supertonic TTS model into memory."""
        async with self._lock:
            await self._load_locked()

    async def _load_locked(self) -> None:
        if self.is_loaded:
            return

        from scripts.fetch_voice_models import default_root

        root = default_root()
        target_dir = self.model_dir or (root / "models" / "supertonic-int8")

        missing = [f for f in REQUIRED_FILES if not (target_dir / f).is_file()]
        if missing:
            raise RuntimeError(
                f"TTS model files missing under {target_dir}: {missing}. "
                f"Run '{PROVISION_CMD}' to download them."
            )

        try:
            import sherpa_onnx
        except (ImportError, OSError) as e:
            raise RuntimeError(
                f"sherpa-onnx is not installed or failed to import: {e}. "
                "Ensure sherpa-onnx is in your virtual environment."
            ) from e

        cfg = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                provider="cpu",
                num_threads=self.threads,
                debug=False,
                supertonic=sherpa_onnx.OfflineTtsSupertonicModelConfig(
                    duration_predictor=str(target_dir / "duration_predictor.int8.onnx"),
                    text_encoder=str(target_dir / "text_encoder.int8.onnx"),
                    vector_estimator=str(target_dir / "vector_estimator.int8.onnx"),
                    vocoder=str(target_dir / "vocoder.int8.onnx"),
                    tts_json=str(target_dir / "tts.json"),
                    unicode_indexer=str(target_dir / "unicode_indexer.bin"),
                    voice_style=str(target_dir / "voice.bin"),
                ),
            )
        )

        def _init() -> Any:
            return sherpa_onnx.OfflineTts(cfg)

        self._tts = await asyncio.to_thread(_init)
        self._loaded = True
        logger.info("SherpaTTS loaded successfully from %s", target_dir)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Synthesizes text into 24 kHz f32 mono PCM chunks."""
        clean = text.strip()
        if not clean:
            return

        if not self.is_loaded:
            await self.load()

        sid = (
            VOICE_MAP.get(str(self.voice_id).upper(), 0)
            if isinstance(self.voice_id, str)
            else int(self.voice_id)
        )

        def _gen() -> tuple[np.ndarray, int]:
            audio = self._tts.generate(clean, sid=sid, speed=self.speed)
            samples = np.asarray(audio.samples, dtype=np.float32)
            return samples, audio.sample_rate

        samples, native_rate = await asyncio.to_thread(_gen)

        # Resample from native rate (typically 44100) to 24000 f32
        resampled = _resample_f32(samples, native_rate, self.sample_rate)

        # Trim model-generated leading and trailing dead silence
        trimmed = _trim_silence(resampled, rate=self.sample_rate, pad_ms=30)

        # Yield raw float32 PCM bytes
        yield trimmed.tobytes()

    async def aclose(self) -> None:
        """Cleans up resources."""
        async with self._lock:
            self._tts = None
            self._loaded = False
