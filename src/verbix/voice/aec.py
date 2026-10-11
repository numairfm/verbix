from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FRAME_SIZE = 160  # 10ms at 16000 Hz


class EchoCanceller:
    """Acoustic echo canceller backed by WebRTC AudioProcessor.

    Subtracts outgoing speaker reference audio from incoming microphone audio
    in 10ms (160-sample) chunks at 16kHz so user speech is cleanly isolated.
    """

    def __init__(self, sample_rate: int = 16000, stream_delay_ms: int = 0) -> None:
        self.sample_rate = sample_rate
        self._ref_buffer: bytearray = bytearray()
        self._mic_pending: bytearray = bytearray()
        self._processor: Any = None
        try:
            from pywebrtc_audio import AudioProcessor

            self._processor = AudioProcessor(
                sample_rate=sample_rate,
                num_channels=1,
                echo_cancellation=True,
                noise_suppression=True,
                high_pass_filter=True,
                stream_delay_ms=stream_delay_ms,
            )
        except (ImportError, OSError, RuntimeError) as e:
            logger.warning("WebRTC AudioProcessor unavailable (%s), AEC bypassed.", e)

    @property
    def is_available(self) -> bool:
        return self._processor is not None

    def feed_reference(self, pcm_bytes: bytes, source_sample_rate: int = 24000) -> None:
        """Enqueues reference speaker audio for echo cancellation."""
        if not pcm_bytes or self._processor is None:
            return

        if source_sample_rate != self.sample_rate:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            if len(samples) > 0:
                n_out = int(len(samples) * self.sample_rate / source_sample_rate)
                if n_out > 0:
                    resampled = np.interp(
                        np.linspace(0, len(samples), n_out, endpoint=False),
                        np.arange(len(samples)),
                        samples,
                    ).astype(np.int16)
                    self._ref_buffer.extend(resampled.tobytes())
        else:
            self._ref_buffer.extend(pcm_bytes)

    def process_mic(self, mic_bytes: bytes) -> bytes:
        """Processes mic audio in 160-sample chunks, subtracting speaker echo."""
        if not mic_bytes and not self._mic_pending:
            return b""
        if self._processor is None or (not self._ref_buffer and not self._mic_pending):
            return mic_bytes

        combined = bytes(self._mic_pending) + mic_bytes
        frame_bytes = FRAME_SIZE * 2  # 320 bytes per 10ms frame @ 16kHz int16
        n_frames = len(combined) // frame_bytes
        if n_frames == 0:
            self._mic_pending = bytearray(combined)
            return b""

        full_len = n_frames * frame_bytes
        mic_part = combined[:full_len]
        self._mic_pending = bytearray(combined[full_len:])

        mic_samples = np.frombuffer(mic_part, dtype=np.int16).reshape(n_frames, FRAME_SIZE)

        # Slice reference audio or pad with zeros if exhausted
        ref_slice_len = min(len(self._ref_buffer), full_len)
        ref_raw = bytes(self._ref_buffer[:ref_slice_len])
        del self._ref_buffer[:ref_slice_len]

        if len(ref_raw) < full_len:
            ref_raw = ref_raw + (b"\x00" * (full_len - len(ref_raw)))

        ref_samples = np.frombuffer(ref_raw, dtype=np.int16).reshape(n_frames, FRAME_SIZE)

        out_chunks = []
        for i in range(n_frames):
            chunk = mic_samples[i]
            ref_chunk = ref_samples[i]
            try:
                cleaned = self._processor.process(chunk, ref_chunk)
                out_chunks.append(np.asarray(cleaned, dtype=np.int16))
            except Exception as e:  # noqa: BLE001
                logger.debug("AEC processing error: %s", e)
                out_chunks.append(chunk)

        if not out_chunks:
            return b""
        return np.concatenate(out_chunks).tobytes()

    def reset(self) -> None:
        """Clears reference and pending buffers and resets internal filter state."""
        self._ref_buffer.clear()
        self._mic_pending.clear()
        if self._processor is not None:
            try:
                self._processor.reset()
            except Exception as e:  # noqa: BLE001
                logger.debug("AEC reset error: %s", e)
