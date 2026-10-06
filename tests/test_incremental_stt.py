import pytest

from verbix.voice.stt import SegmentCommittedSTT


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
async def test_20ms_capture_chunks_still_pause_commit(monkeypatch):
    """Silero rejects sub-32ms windows; 20ms capture chunks must still commit."""
    import faster_whisper.vad as vad_mod
    import numpy as np

    def silero_like(audio, sampling_rate=16000):
        return [] if len(np.asarray(audio)) < 512 else [{"start": 0, "end": 512}]

    monkeypatch.setattr(vad_mod, "get_speech_timestamps", silero_like)
    calls = []
    async def fake_decode(pcm, prompt=""):
        calls.append(pcm)
        return "hello"
    now = [0.0]
    stt = SegmentCommittedSTT(decode_fn=fake_decode, clock=lambda: now[0])
    sess = stt.open_session()
    chunk = b"\x01\x02" * 320  # 20ms @16kHz, nonzero energy
    for _ in range(20):
        sess.feed(chunk)
    now[0] += 0.3
    sess.feed(b"\x00\x00" * 320)
    assert await sess.finalize() == "hello"
    assert len(calls) == 1 and calls[0] == chunk * 20
