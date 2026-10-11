import pytest

from verbix.voice.sherpa_stt import SegmentCommittedSTT


def _speech_frames(n):  # fake VAD: nonzero energy
    return [b"\x01\x02" * 320] * n

@pytest.mark.asyncio
async def test_pause_commits_segment_exactly_once():
    calls = []
    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "hello"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, vad_fn=lambda pcm: True,
                              clock=lambda: now[0])
    sess = stt.open_session()
    speech = _speech_frames(20)
    for chunk in speech:
        sess.feed(chunk)
    now[0] += 0.3  # 300ms of silence elapses on the fake clock
    sess.feed(b"\x00\x00" * 320)  # silence: VAD-false, triggers commit check
    await sess.finalize()
    assert len(calls) == 1  # committed audio never decoded twice
    assert calls[0] == b"".join(speech)

@pytest.mark.asyncio
async def test_force_commit_bounds_long_segments():
    seen = []
    async def fake_decode(pcm, prompt=""):
        seen.append(len(pcm))
        return "x"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, vad_fn=lambda pcm: True,
                              clock=lambda: now[0])
    sess = stt.open_session()
    for chunk in _speech_frames(2000):  # ~20s of "speech"
        sess.feed(chunk)
        now[0] += 0.01
    await sess.finalize()
    assert len(seen) >= 2
    assert all(n <= 12 * 16000 * 2 + 16000 * 2 for n in seen)

@pytest.mark.asyncio
async def test_20ms_capture_chunks_still_pause_commit():
    """Sub-32ms chunks still commit properly on pause."""
    calls = []
    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "hello"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, vad_fn=lambda pcm: any(pcm), clock=lambda: now[0])
    sess = stt.open_session()
    chunk = b"\x01\x02" * 320  # 20ms @16kHz, nonzero energy
    for _ in range(20):
        sess.feed(chunk)
    now[0] += 0.3
    sess.feed(b"\x00\x00" * 320)
    assert await sess.finalize() == "hello"
    assert len(calls) == 1 and calls[0] == chunk * 20


@pytest.mark.asyncio
async def test_fragmented_speech_passes_turn_gate():
    """Speech split across pause-committed fragments still finalizes: the gate
    judges the whole turn, not each fragment."""
    import numpy as np

    calls = []
    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "frag"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, vad_fn=lambda pcm: any(pcm), clock=lambda: now[0])
    sess = stt.open_session()
    rng = np.random.default_rng(1)
    # Two 300 ms loud fragments separated by a committing pause
    for _ in range(2):
        frag = (rng.normal(0, 4000, 4800)).astype(np.int16).tobytes()
        for i in range(0, len(frag), 640):
            sess.feed(frag[i:i + 640])
        now[0] += 0.3
        sess.feed(b"\x00\x00" * 320)
    text = await sess.finalize()
    assert "frag" in text


@pytest.mark.asyncio
async def test_blip_turn_finalizes_empty():
    """A turn of only short blips returns '' at finalize: sub-threshold total."""
    import numpy as np

    calls = []
    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "blip"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, vad_fn=lambda pcm: any(pcm), clock=lambda: now[0])
    sess = stt.open_session()
    rng = np.random.default_rng(2)
    blip = (rng.normal(0, 4000, 1600)).astype(np.int16).tobytes()  # 100 ms
    sess.feed(blip)
    now[0] += 0.3
    sess.feed(b"\x00\x00" * 320)
    assert await sess.finalize() == ""
