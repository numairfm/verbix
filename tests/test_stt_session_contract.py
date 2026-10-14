"""S2 session contract: every STT session double implements the real surface."""

from unittest.mock import MagicMock

from tests.test_conversational_voice_loop import FakeSTT as ConvFakeSTT
from tests.test_draft_shelf import FakeSTT as ShelfFakeSTT
from verbix.voice.base import STTSessionProtocol
from verbix.voice.transcribe_stt import STTSession, TranscribeSession


def _assert_surface(sess, name):
    assert isinstance(sess, STTSessionProtocol), f"{name} breaks STTSessionProtocol"
    sess.feed(b"\x00" * 320)
    assert sess.feed_barge_in(b"\x00" * 320) is False
    assert isinstance(sess.speech_duration, float)
    sess.reset()
    sess.cancel()


def _marker_segmenter():
    seg = MagicMock()
    seg.pop_segments.return_value = []
    seg.flush.return_value = []
    seg.is_speech_detected = False
    return seg


def test_real_sessions_satisfy_protocol():
    async def _decode(pcm, prompt=""):
        return ""

    _assert_surface(STTSession(_decode, _marker_segmenter()), "STTSession")
    _assert_surface(
        TranscribeSession(stt=MagicMock(), segmenter=None), "TranscribeSession"
    )


def test_fake_sessions_satisfy_protocol():
    for name, stt in (("conv", ConvFakeSTT()), ("shelf", ShelfFakeSTT(["hi"]))):
        sess = stt.open_session(sample_rate=16000, on_endpoint=lambda: None)
        _assert_surface(sess, f"FakeLiveSession[{name}]")
