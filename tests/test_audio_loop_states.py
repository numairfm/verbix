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
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.base import MockAudioSink, MockTextToSpeech
    from verbix.voice.player import AudioPlayer
    from verbix.core.events import TextDelta

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
    await pipeline.execute_voice_turn("User question")
    assert rt.floor.state == FloorState.IDLE
    assert rt.floor.is_idle is True
