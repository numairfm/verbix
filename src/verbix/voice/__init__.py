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
    if name == "SherpaTTS":
        from verbix.voice.sherpa_tts import SherpaTTS

        return SherpaTTS
    if name == "SherpaSTT":
        from verbix.voice.sherpa_stt import SherpaSTT

        return SherpaSTT
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AudioPlayer",
    "AudioSink",
    "CallbackAudioSink",
    "MockAudioSink",
    "MockSpeechToText",
    "MockTextToSpeech",
    "PTTCapture",
    "SherpaSTT",
    "SherpaTTS",
    "SpeechOutput",
    "SpeechToText",
    "SubprocessAudioSink",
    "TextToSpeech",
    "VoicePipeline",
    "select_sink",
]
