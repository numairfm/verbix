import pytest

from verbix.runtime.floor import FloorArbiter, FloorState


def test_floor_states_and_speaker_mute_semantics():
    events = []
    floor = FloorArbiter(on_change=lambda prev, nxt: events.append((prev, nxt)))
    assert floor.state == FloorState.IDLE
    assert floor.is_idle is True

    # User presses PTT: enters LISTENING (speakers must be muted)
    floor.set_state(FloorState.LISTENING)
    assert floor.state == FloorState.LISTENING
    assert floor.is_listening is True

    # Assistant enters SPEAKING (covers generation, tool thinking, and audio playback)
    floor.set_state(FloorState.SPEAKING)
    assert floor.state == FloorState.SPEAKING
    assert floor.is_speaking is True

    # Audio finishes: returns to IDLE
    floor.set_state(FloorState.IDLE)
    assert floor.state == FloorState.IDLE
    assert floor.is_idle is True

    # Verifies all clean transitions were observed
    assert events == [
        (FloorState.IDLE, FloorState.LISTENING),
        (FloorState.LISTENING, FloorState.SPEAKING),
        (FloorState.SPEAKING, FloorState.IDLE),
    ]


@pytest.mark.asyncio
async def test_pipeline_ptt_mutes_playback_and_transitions_states():
    import asyncio

    from verbix.core.events import TextDelta
    from verbix.voice.base import MockAudioSink, MockTextToSpeech
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer

    floor_events = []

    class FakeRuntime:
        def __init__(self):
            self.floor = FloorArbiter(on_change=lambda p, n: floor_events.append((p, n)))
            self.is_idle = True
            self.user_active = False
            self.wake_output = None

        async def submit(self, text, output):
            return "response"

        async def cancel_turn(self):
            pass

    rt = FakeRuntime()
    sink = MockAudioSink(playback_delay=0.5)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    pipeline = VoicePipeline(runtime=rt, stt=None, player=player)

    # Start playback
    await pipeline.turn_output.emit(TextDelta(text="Some assistant speech."))
    await asyncio.sleep(0.05)
    assert pipeline.turn_output.is_playing is True

    # PTT recording starts: hard-mutes playback and enters LISTENING
    await pipeline.start_recording()
    assert sink.stopped is True  # Speaker playback hard-muted
    assert rt.floor.state == FloorState.LISTENING
    assert rt.floor.is_listening is True

    # Stop recording & execute voice turn: enters SPEAKING, then finishes at IDLE
    await pipeline.capture.stop_recording()
    await pipeline.execute_voice_turn("User question")
    assert rt.floor.state == FloorState.IDLE
    assert rt.floor.is_idle is True


@pytest.mark.asyncio
async def test_pipeline_logs_user_transcript_ptt():
    """Pipeline logs finalized PTT transcripts with duration and ignores empty ones."""
    from unittest.mock import AsyncMock, MagicMock

    from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer

    printed = []
    printer = MagicMock()
    printer.print_user = lambda text, duration=None: printed.append((text, duration))

    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.submit = AsyncMock(return_value="ok")

    cap = MagicMock()
    cap.sample_rate = 16000
    cap.is_recording = False
    cap.start_recording = AsyncMock()
    cap.stop_recording = AsyncMock(return_value=b"\x01\x00" * 16000)  # 1.0s

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=MockSpeechToText(predefined_transcriptions=["hello from ptt", ""]),
        player=AudioPlayer(tts=MockTextToSpeech(), sink=MockAudioSink()),
        capture=cap,
        printer=printer,
    )

    res1 = await pipeline.stop_recording_and_process()
    assert res1["action"] == "processing"
    if pipeline._active_voice_task:
        await pipeline._active_voice_task
    assert len(printed) == 1
    assert printed[0][0] == "hello from ptt"
    assert printed[0][1] == 1.0

    res2 = await pipeline.stop_recording_and_process()
    assert res2["action"] == "idle"
    assert len(printed) == 1

