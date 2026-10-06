from typing import Any

from verbix.voice.base import (
    AudioSink,
    MockAudioSink,
    MockSpeechToText,
    MockTextToSpeech,
    SpeechToText,
    TextToSpeech,
)
from verbix.voice.callback_sink import CallbackAudioSink, select_sink
from verbix.voice.capture import PTTCapture
from verbix.voice.output import SpeechOutput
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer
from verbix.voice.sinks import SubprocessAudioSink


def __getattr__(name: str) -> Any:
    if name == "PocketTTSEngine":
        from verbix.voice.pocket_tts import PocketTTSEngine

        return PocketTTSEngine
    if name == "FasterWhisperSTT":
        from verbix.voice.stt import FasterWhisperSTT

        return FasterWhisperSTT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AudioPlayer",
    "AudioSink",
    "CallbackAudioSink",
    "FasterWhisperSTT",
    "MockAudioSink",
    "MockSpeechToText",
    "MockTextToSpeech",
    "PTTCapture",
    "PocketTTSEngine",
    "SpeechOutput",
    "SpeechToText",
    "SubprocessAudioSink",
    "TextToSpeech",
    "VoicePipeline",
    "select_sink",
]
