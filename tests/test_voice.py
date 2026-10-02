from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from verbix.core.models import WorkerResult
from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime
from verbix.voice.base import MockAudioSink, MockSpeechToText, MockTextToSpeech
from verbix.voice.pipeline import VoicePipeline, split_sentences
from verbix.voice.player import AudioPlayer


def test_split_sentences():
    text = "Hello world! This is Verbix. How are you today? All good\nLet's go."
    sentences = split_sentences(text)
    assert len(sentences) == 5
    assert sentences[0] == "Hello world!"
    assert sentences[1] == "This is Verbix."
    assert sentences[2] == "How are you today?"
    assert sentences[3] == "All good"
    assert sentences[4] == "Let's go."


@pytest.mark.asyncio
async def test_audio_player_foreground_speech(tmp_path: Path):
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink)

    sentences = ["First sentence.", "Second sentence.", "Third sentence."]
    played = await player.speak_foreground_sentences(sentences)

    assert played == "First sentence. Second sentence. Third sentence."
    assert len(tts.synthesized_texts) == 3
    assert len(sink.played_chunks) == 6  # 2 chunks per sentence


@pytest.mark.asyncio
async def test_audio_player_barge_in_interruption(tmp_path: Path):
    tts = MockTextToSpeech()
    # 0.05s per chunk. 2 chunks per sentence = 0.10s per sentence
    sink = MockAudioSink(playback_delay=0.05)
    player = AudioPlayer(tts=tts, sink=sink)

    sentences = ["Sentence one.", "Sentence two.", "Sentence three."]

    async def speak_task():
        return await player.speak_foreground_sentences(sentences)

    task = asyncio.create_task(speak_task())

    # Wait for first sentence (0.10s) to finish, interrupt during sentence two (0.13s)
    await asyncio.sleep(0.13)
    played_so_far = player.interrupt()
    result = await task

    # Interruption stopped subsequent sentences
    assert "Sentence three." not in result
    assert "Sentence one." in played_so_far


@pytest.mark.asyncio
async def test_audio_player_announcement_queue(tmp_path: Path):
    provider = ProviderClient(base_url="http://mock", model="mock", api_key="k")
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    runtime.create_receptionist()

    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink, runtime=runtime)

    # Register task and finish it
    from verbix.core.models import TaskState

    task_id = "task-build-1"
    runtime.tasks[task_id] = TaskState(id=task_id, title="Background Build", status="running")

    worker_res = WorkerResult(
        success=True,
        speech_summary="Build finished successfully in 4 seconds.",
        details="Binaries in dist/.",
    )
    runtime._finish_task(task_id, worker_res, status="completed")

    task = runtime.get_task(task_id)
    assert task is not None
    assert task.announced is False

    # Drain announcement queue
    drained = await player.drain_announcements()
    assert drained == 1
    assert task.announced is True
    assert "Build finished successfully in 4 seconds." in tts.synthesized_texts


@pytest.mark.asyncio
async def test_voice_pipeline_ptt_flow(tmp_path: Path):
    provider = ProviderClient(base_url="http://mock", model="mock", api_key="k")
    runtime = TaskRuntime(workspace=tmp_path, provider=provider)
    receptionist = runtime.create_receptionist()

    # Pre-populate history so receptionist can respond
    stt = MockSpeechToText(predefined_transcriptions=["Where is the workspace?"])
    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    player = AudioPlayer(tts=tts, sink=sink, runtime=runtime)

    pipeline = VoicePipeline(runtime=runtime, stt=stt, player=player)

    # Simulate PTT down
    await pipeline.on_ptt_down()
    assert not player.is_playing

    # Simulate feeding recorded audio directly to stt test
    transcription = await stt.transcribe(b"\x00" * 3200)
    assert transcription == "Where is the workspace?"

    # Execute voice turn manually with mock receptionist
    receptionist.history.append_user("Where is the workspace?")
    receptionist.history.append_assistant("The workspace is in /tmp.")

    spoken = await player.speak_foreground_sentences(["The workspace is in /tmp."])
    assert spoken == "The workspace is in /tmp."
    assert "The workspace is in /tmp." in tts.synthesized_texts
