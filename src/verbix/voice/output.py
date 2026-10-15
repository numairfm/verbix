"""Non-blocking speech output: one utterance streams through one sink stream.

`emit()` only enqueues epoch-tagged frames and returns; a background pump task
synthesizes each text frame (`normalize_for_tts` + per-chunk `synthesize`) and
yields audio into a single `play_stream` generator call. `interrupt()` bumps
the epoch, silences the sink, and drops stale frames synchronously.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

from verbix.core.events import AgentEvent, TextDelta, TurnComplete
from verbix.core.telemetry import record_latency
from verbix.voice.segments import normalize_for_tts, split_chunks

logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TERMINAL = (".", "!", "?")
_INTERNAL_CONTROL_TOKENS = frozenset({
    "RepeatedToolCall",
    "LimitsExceeded",
    "TimeExceeded",
    "RepeatedFormatError",
    "FormatError",
})

# Prefetch depth: synth tasks kept warm while earlier frames play.
_WARM_DEPTH = 2
# Breath between sentences so seamless playback never sounds rushed.
_GAP_SECONDS = 0.15


@dataclass
class InterruptionSnapshot:
    """Exact boundary snapshot of an interrupted utterance."""

    confirmed_heard: str = ""
    interrupted_sentence: str = ""
    unplayed_text: str = ""
    timestamp: float = field(default_factory=time.time)

    def __init__(
        self,
        confirmed_heard: str = "",
        interrupted_sentence: str = "",
        unplayed_text: str = "",
        timestamp: float | None = None,
        heard_text: str | None = None,
        unspoken_draft: str | None = None,
    ) -> None:
        if heard_text is not None and not confirmed_heard:
            confirmed_heard = heard_text
        if unspoken_draft is not None and not unplayed_text and not interrupted_sentence:
            unplayed_text = unspoken_draft
        self.confirmed_heard = confirmed_heard
        self.interrupted_sentence = interrupted_sentence
        self.unplayed_text = unplayed_text
        self.timestamp = time.time() if timestamp is None else timestamp

    @property
    def heard_text(self) -> str:
        return self.confirmed_heard

    @property
    def unspoken_draft(self) -> str:
        parts = [p for p in (self.interrupted_sentence, self.unplayed_text) if p]
        return " ".join(parts).strip()



def _sentences(text: str) -> list[str]:
    """Split text at sentence boundaries so sub-220 tails keep TTS pauses."""
    return [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]


class SpeechOutput:
    """TurnOutput that speaks agent text without blocking the turn loop."""

    def __init__(
        self,
        tts: Any,
        sink: Any,
        printer: Any = None,
    ) -> None:
        self.tts = tts
        self.sink = sink
        self.printer = printer
        self.on_event: Any = None
        self._buf = ""
        self._seen = ""
        self._delivered: list[str] = []
        self._epoch = 0
        self._frames: asyncio.Queue[tuple[int, str, Any]] = asyncio.Queue()
        self._synth_tasks: set[asyncio.Task] = set()
        self._pump: asyncio.Task[None] | None = None
        self._warm: list[list] = []  # [tag, kind, payload, task|None] in frame order
        self.last_interruption: InterruptionSnapshot | None = None
        self._sent_spans: list[tuple[str, int, int]] = []
        self._enqueued_samples: int = 0
        raw_played = getattr(sink, "played_samples", None)
        self._start_played_samples: int = raw_played if isinstance(raw_played, int) else 0


    @property
    def is_playing(self) -> bool:
        return self._pump is not None and not self._pump.done()

    async def emit(self, event: AgentEvent) -> None:
        if self.printer and hasattr(self.printer, "emit"):
            res = self.printer.emit(event)
            if asyncio.iscoroutine(res):
                await res
        if callable(self.on_event):
            if isinstance(event, TurnComplete) and (event.final_text or "").strip() in _INTERNAL_CONTROL_TOKENS:
                pass
            else:
                res_ev = self.on_event(event)
                if asyncio.iscoroutine(res_ev):
                    await res_ev

        if isinstance(event, TextDelta):
            self._buf += event.text
            self._seen += event.text
            chunks = split_chunks(self._buf)
            if len(chunks) > 1:
                ready, self._buf = chunks[:-1], chunks[-1]
            elif self._buf.strip().endswith(_TERMINAL):
                ready, self._buf = chunks, ""
            else:
                return
            for c in ready:
                for s in _sentences(c) or [c]:
                    self._frames.put_nowait((self._epoch, "text", s))
            self._ensure_pump()
            self._top_up()
        elif isinstance(event, TurnComplete):
            buf, self._buf = self._buf, ""
            if buf.strip() and buf.strip() not in _INTERNAL_CONTROL_TOKENS:
                for s in _sentences(buf) or [buf]:
                    self._frames.put_nowait((self._epoch, "text", s))
            final_text = (event.final_text or "").strip()
            if final_text in _INTERNAL_CONTROL_TOKENS:
                final_text = ""
            rest = self._tail(final_text)
            if rest:
                for s in _sentences(rest) or [rest]:
                    self._frames.put_nowait((self._epoch, "text", s))
            self._frames.put_nowait((self._epoch, "end", ""))
            self._ensure_pump()
            self._top_up()

    def _tail(self, final_text: str) -> str:
        f = (final_text or "").strip()
        if not f or not self._seen:
            return f
        if f.startswith(self._seen.strip()):
            return f[len(self._seen.strip()):].strip()
        if self._seen.strip().startswith(f) or f in self._seen:
            return ""
        return f

    def interrupt(self) -> InterruptionSnapshot | None:
        """Barge-in: silence audio now, capture boundary snapshot, and discard pending."""
        playing = self.is_playing
        if playing:
            record_latency("interrupt_called")

        # Snapshot unvoiced frames before purging
        unplayed_tail: list[str] = []
        for _, k, p, _ in self._warm:
            if k == "text" and p:
                unplayed_tail.append(p)
        while not self._frames.empty():
            try:
                _, k, p = self._frames.get_nowait()
                if k == "text" and p:
                    unplayed_tail.append(p)
            except asyncio.QueueEmpty:
                break
        if self._buf.strip():
            unplayed_tail.extend(_sentences(self._buf))

        played_raw = getattr(self.sink, "played_samples", None)
        played_samples: int | None = None
        if isinstance(played_raw, int):
            played_samples = max(0, played_raw - self._start_played_samples)
        confirmed_heard_sents: list[str] = []
        interrupted_sent = ""
        unplayed_span_sents: list[str] = []

        if isinstance(played_samples, int) and self._sent_spans:
            for sent, start, end in self._sent_spans:
                if end <= played_samples:
                    confirmed_heard_sents.append(sent)
                elif start <= played_samples < end:
                    interrupted_sent = sent
                else:
                    unplayed_span_sents.append(sent)
        elif playing:
            confirmed_heard_sents = list(self._delivered)

        confirmed_heard = " ".join(confirmed_heard_sents).strip()
        unplayed_text = " ".join(unplayed_span_sents + unplayed_tail).strip()

        snapshot: InterruptionSnapshot | None = None
        if confirmed_heard or interrupted_sent or unplayed_text:
            snapshot = InterruptionSnapshot(
                confirmed_heard=confirmed_heard,
                interrupted_sentence=interrupted_sent,
                unplayed_text=unplayed_text,
                timestamp=time.time(),
            )
            self.last_interruption = snapshot

        self._epoch += 1
        self.sink.stop()
        if playing:
            record_latency("audio_silent")
        if self._pump and not self._pump.done():
            self._pump.cancel()
        self._pump = None
        for task in list(self._synth_tasks):
            if not task.done():
                task.cancel()
        self._synth_tasks.clear()
        self._drop_warm()
        self._buf = ""
        self._seen = ""
        self._delivered.clear()
        self._sent_spans.clear()
        self._enqueued_samples = 0
        raw_played = getattr(self.sink, "played_samples", None)
        self._start_played_samples = raw_played if isinstance(raw_played, int) else 0
        return snapshot

    def delivered_text(self) -> str:
        if self._delivered:
            return " ".join(self._delivered).strip()
        if self.last_interruption:
            return self.last_interruption.heard_text
        return ""

    def reset(self) -> None:
        """Clean reset between turns: halts playback and clears buffers without setting last_interruption."""
        self._epoch += 1
        self.sink.stop()
        if self._pump and not self._pump.done():
            self._pump.cancel()
        self._pump = None
        for task in list(self._synth_tasks):
            if not task.done():
                task.cancel()
        self._synth_tasks.clear()
        self._drop_warm()
        while not self._frames.empty():
            try:
                self._frames.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._buf = ""
        self._seen = ""
        self._delivered.clear()
        self._sent_spans.clear()
        self._enqueued_samples = 0
        raw_played = getattr(self.sink, "played_samples", None)
        self._start_played_samples = raw_played if isinstance(raw_played, int) else 0

    def reset_delivered(self) -> None:  # compat for WakeOutputFanOut/printer protocol
        self.reset()

    async def speak_text(self, text: str) -> None:
        """Enqueues and speaks an utterance directly without a prior AgentEvent."""
        text = text.strip()
        if not text:
            return
        for s in _sentences(text) or [text]:
            self._frames.put_nowait((self._epoch, "text", s))
        self._frames.put_nowait((self._epoch, "end", ""))
        self._ensure_pump()
        self._top_up()


    async def play_cue(self, freq: float = 880.0) -> None:
        """Earcon that never kills speech: direct play only when idle."""
        try:
            import numpy as np

            sample_rate = getattr(self.sink, "sample_rate", 24000)
            fmt = getattr(self.sink, "audio_format", "f32")
            n = int(sample_rate * 0.05)
            t = np.linspace(0, 0.05, n, endpoint=False)
            envelope = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
            wave = (np.sin(2 * np.pi * float(freq) * t) * envelope * 0.25).astype(np.float32)
            cue = wave.tobytes() if fmt == "f32" else (wave * 32767).astype(np.int16).tobytes()
        except (ImportError, OSError, RuntimeError, ValueError):
            logger.debug("Failed to build cue sound", exc_info=True)
            return
        if self.is_playing:
            self._frames.put_nowait((self._epoch, "cue", cue))
        else:
            await self.sink.play(cue)

    def _ensure_pump(self) -> None:
        if self._pump is None or self._pump.done():
            self._pump = asyncio.create_task(self._run())

    async def _run(self) -> None:
        current = asyncio.current_task()
        epoch = self._epoch
        try:
            while self._epoch == epoch and (not self._frames.empty() or self._warm):
                frames_before = self._frames.qsize()
                await self.sink.play_stream(self._feed(epoch))
                if self._frames.qsize() >= frames_before and not self._frames.empty():
                    while not self._frames.empty():
                        try:
                            self._frames.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    break
        except asyncio.CancelledError:
            if self._epoch == epoch:
                self.sink.stop()
            raise
        finally:
            if self._pump is current:
                self._pump = None
                if self._epoch == epoch:
                    self._sent_spans.clear()
                    self._enqueued_samples = 0

    def _spawn_synth(
        self, payload: str, epoch: int
    ) -> asyncio.Task[list[bytes] | None]:
        """Synthesizes one text frame to buffered parts without playing them.

        Returns None when the epoch moved mid-synthesis (stale: discard).
        Playing still happens only in _feed, so prefetched audio is never
        heard unless the pump yields it — barge-in stays exact.
        """

        async def _run() -> list[bytes] | None:
            parts: list[bytes] = []
            async for audio in self.tts.synthesize(normalize_for_tts(payload)):
                if self._epoch != epoch:
                    return None
                parts.append(audio)
            return parts

        task = asyncio.create_task(_run())
        self._synth_tasks.add(task)
        task.add_done_callback(self._synth_tasks.discard)
        return task

    def _gap_bytes(self) -> bytes:
        """Short silence between sentences; zeros in the sink's format."""
        rate = int(getattr(self.sink, "sample_rate", 24000) or 24000)
        fmt = getattr(self.sink, "audio_format", "f32")
        return bytes(int(rate * _GAP_SECONDS) * (2 if fmt != "f32" else 4))

    def _drop_warm(self) -> None:
        for _, _, _, t in self._warm:
            if t is not None and not t.done():
                t.cancel()
        self._warm.clear()

    def _top_up(self) -> None:
        """Move queued frames into the warm list, starting synth at once.

        Called on every enqueue (emit) and every feed iteration, so a frame
        that arrives mid-playback starts synthesizing immediately instead of
        waiting for the current frame to finish. Bounded to _WARM_DEPTH
        concurrent synth tasks; order is always frame order.
        """
        epoch = self._epoch
        running = sum(
            1 for _, k, _, t in self._warm if k == "text" and t is not None and not t.done()
        )
        while running < _WARM_DEPTH:
            try:
                tag, kind, payload = self._frames.get_nowait()
            except asyncio.QueueEmpty:
                return
            if tag != epoch:
                return
            if kind == "end":
                self._warm.append([tag, kind, payload, None])
                continue
            if kind == "cue":
                self._warm.append([tag, kind, payload, None])
                return
            self._warm.append([tag, kind, payload, self._spawn_synth(payload, epoch)])
            running += 1

    async def _feed(self, epoch: int):
        first_write = True
        first_pcm = True
        spoken_any = False
        while True:
            self._top_up()
            if not self._warm:
                tag, kind, payload = await self._frames.get()
                if tag != self._epoch or tag != epoch:
                    return
                if kind == "end":
                    self._top_up()
                    if self._frames.empty() and not self._warm:
                        return
                    continue
                if kind == "cue":
                    if first_write:
                        record_latency("sink_first_write")
                        first_write = False
                    yield payload
                    self._top_up()
                    if self._frames.empty() and not self._warm:
                        return
                    continue
                self._warm.append([tag, kind, payload, self._spawn_synth(payload, epoch)])
                continue
            tag, kind, payload, task = self._warm.pop(0)
            if tag != self._epoch or tag != epoch:
                self._drop_warm()
                return
            if kind == "cue":
                if first_write:
                    record_latency("sink_first_write")
                    first_write = False
                yield payload
                self._top_up()
                if self._frames.empty() and not self._warm:
                    return
                continue
            if kind == "end":
                self._top_up()
                if self._frames.empty() and not self._warm:
                    return
                continue
            if task is None:
                task = self._spawn_synth(payload, epoch)
            try:
                parts = await task
            except (asyncio.CancelledError, Exception):
                self._drop_warm()
                raise
            if parts is None or tag != self._epoch or tag != epoch:
                self._drop_warm()
                return
            if first_pcm:
                record_latency("tts_first_pcm")
                first_pcm = False
            if parts:
                fmt = getattr(self.sink, "audio_format", "f32")
                sample_width = 4 if fmt in ("f32", "float", "f32le") else 2
                if spoken_any:
                    gap = self._gap_bytes()
                    self._enqueued_samples += len(gap) // sample_width
                    yield gap
                spoken_any = True
                if not self._sent_spans:
                    raw_played = getattr(self.sink, "played_samples", None)
                    if isinstance(raw_played, int):
                        self._start_played_samples = raw_played
                    self._delivered.clear()
                self._delivered.append(payload)
                chunk_samples = sum(len(p) // sample_width for p in parts)
                start_sample = self._enqueued_samples
                end_sample = start_sample + chunk_samples
                self._enqueued_samples = end_sample
                self._sent_spans.append((payload, start_sample, end_sample))
                for audio in parts:
                    if first_write:
                        record_latency("sink_first_write")
                        first_write = False
                    yield audio
