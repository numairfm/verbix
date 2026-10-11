"""Span-driven sessions: bytes in, Silero span boundaries out, spans decode."""

import asyncio

import numpy as np
import pytest

from verbix.voice.sherpa_stt import SegmentCommittedSTT, STTSession


class FakeSegmenter:
    """Scripted spans: feed() buffers, pop_segments() emits queued scripts."""

    def __init__(self, scripts: list[list[bytes]] | None = None) -> None:
        self.scripts = list(scripts or [])
        self.fed: list[bytes] = []
        self.calls = 0

    def feed(self, pcm: bytes) -> None:
        self.fed.append(pcm)

    def pop_segments(self) -> list[bytes]:
        self.calls += 1
        if self.scripts:
            return self.scripts.pop(0)
        return []

    def flush(self) -> list[bytes]:
        rest = self.scripts
        self.scripts = []
        return [b for spans in rest for b in spans]

    def reset(self) -> None:
        self.fed = []


def _loud(n_samples: int, seed: int = 1, amp: float = 4000.0) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.normal(0, amp, n_samples)).astype(np.int16).tobytes()


@pytest.mark.asyncio
async def test_pop_audio_never_decodes():
    """The mic-open pop never becomes a span, so it never reaches the model."""
    calls = []

    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "."

    seg = FakeSegmenter(scripts=[[]])  # pop yields no spans
    sess = STTSession(fake_decode, seg)
    sess.feed(_loud(960, seed=3))  # 60 ms loud pop
    assert await sess.finalize() == ""
    assert calls == []


@pytest.mark.asyncio
async def test_speech_spans_decode_and_join_in_order():
    calls = []

    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return f"w{len(calls)}"

    seg = FakeSegmenter(scripts=[[b"a" * 6400], [b"b" * 6400]])
    sess = STTSession(fake_decode, seg)
    sess.feed(b"\x01\x02" * 320)
    assert await sess.finalize() == "w1 w2"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_flush_spans_decode_at_finalize():
    """Spans emitted by flush() (trailing speech) decode like any other."""
    calls = []

    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "tail"

    seg = FakeSegmenter(scripts=[[], [b"t" * 12800]])
    sess = STTSession(fake_decode, seg)
    sess.feed(b"\x01\x02" * 320)
    assert await sess.finalize() == "tail"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_blip_turn_finalizes_empty():
    """A transient that yields no spans returns ''."""
    calls = []

    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "blip"

    rng = np.random.default_rng(2)
    blip = (rng.normal(0, 4000, 1600)).astype(np.int16).tobytes()  # 100 ms
    seg = FakeSegmenter(scripts=[[]])
    sess = STTSession(fake_decode, seg)
    sess.feed(blip)
    assert await sess.finalize() == ""
    assert calls == []


@pytest.mark.asyncio
async def test_fragmented_speech_passes_turn_gate():
    """Speech across several spans finalizes: the gate judges the turn total."""

    calls = []

    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "frag"

    rng = np.random.default_rng(1)
    frags = [
        (rng.normal(0, 4000, 4800)).astype(np.int16).tobytes() for _ in range(2)
    ]
    seg = FakeSegmenter(scripts=[[frags[0]], [frags[1]]])
    sess = STTSession(fake_decode, seg)
    for f in frags:
        sess.feed(f[:640])
    assert await sess.finalize() == "frag frag"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_cancel_abandons_pending_decodes():
    """After cancel(), finalize returns '' without awaiting the killed task."""
    import numpy as np

    started = []

    async def slow_decode(pcm, prompt=""):
        started.append(pcm)
        await asyncio.sleep(60)

    rng = np.random.default_rng(4)
    loud = (rng.normal(0, 4000, 6400)).astype(np.int16).tobytes()
    seg = FakeSegmenter(scripts=[[loud]])
    sess = STTSession(slow_decode, seg)
    sess.feed(loud[:640])
    await asyncio.sleep(0)  # let the eager decode task start
    sess.cancel()
    assert await sess.finalize() == ""
    assert started == [loud]  # task began, never completed into text


@pytest.mark.asyncio
async def test_endpoint_callbacks_fire_per_span():
    endpoints = []
    partials = []

    async def fake_decode(pcm, prompt=""):
        return "seg"

    seg = FakeSegmenter(scripts=[[b"a" * 6400], [b"b" * 6400]])
    sess = STTSession(
        fake_decode,
        seg,
        on_partial=lambda t: partials.append(t),
        on_endpoint=lambda t: endpoints.append(t),
    )
    sess.feed(b"\x01\x02" * 320)
    await sess.finalize()
    assert endpoints == ["seg", "seg"]
    assert partials[-1] == "seg seg"


def test_factory_builds_sessions_with_segmenter():
    async def fake_decode(pcm, prompt=""):
        return "x"

    seg = FakeSegmenter()
    stt = SegmentCommittedSTT(decode_fn=fake_decode, segmenter=seg)
    sess = stt.open_session()
    assert isinstance(sess, STTSession)
    assert sess._segmenter is seg


def test_factory_without_segmenter_raises_loudly():
    stt = SegmentCommittedSTT(decode_fn=lambda pcm, prompt="": "")
    with pytest.raises(RuntimeError, match="segmenter"):
        stt.open_session()


@pytest.mark.asyncio
async def test_empty_spans_yields_empty_committed_text():
    """When the segmenter yields no spans, committed_text must be empty and stay empty."""
    rng = np.random.default_rng(2)
    blip = (rng.normal(0, 4000, 1600)).astype(np.int16).tobytes()  # 100 ms

    async def fake_decode(pcm, prompt=""):
        return "blip"

    seg = FakeSegmenter(scripts=[[]])
    sess = STTSession(fake_decode, seg)
    sess.feed(blip)

    # First finalize returns ""
    assert await sess.finalize() == ""
    # committed_text property must be empty
    assert sess.committed_text == ""
    # Subsequent finalize must also return ""
    assert await sess.finalize() == ""


@pytest.mark.asyncio
async def test_finalize_cancellation_propagates():
    """If finalize task itself is cancelled, CancelledError must propagate."""
    async def slow_decode(pcm, prompt=""):
        await asyncio.sleep(10)
        return "slow"

    seg = FakeSegmenter(scripts=[[b"a" * 6400]])
    sess = STTSession(slow_decode, seg)
    sess.feed(b"\x01\x02" * 320)

    task = asyncio.create_task(sess.finalize())
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

