"""End-to-end validation and latency benchmark for the real Verbix voice loop.

Measures:
- Key release (T0) -> STT transcription (T1)
- Model time to first sentence (T2)
- PocketTTS first-audio-chunk synthesis (T3)
- Audio playback start (T4)
Total Time to First Audio (TTFA) = T4 - T0
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np

from verbix.cli import load_env
from verbix.core.events import TextDelta, TurnComplete
from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime
from verbix.voice.pipeline import split_sentences
from verbix.voice.pocket_tts import PocketTTSEngine
from verbix.voice.sinks import SubprocessAudioSink
from verbix.voice.stt import FasterWhisperSTT


async def main() -> None:
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("VERBIX_API_KEY")
    if not api_key:
        print("ERROR: No API key found in environment or .env")
        return

    model_id = os.environ.get("VERBIX_MODEL", "google/gemini-3.5-flash-lite")
    workspace = Path.cwd().resolve()

    print("=== Verbix End-to-End Voice Loop Validation ===")
    print(f"Model: {model_id}")
    print(f"Workspace: {workspace}")

    # 1. Initialize components
    print("\n[1/4] Initializing local engines...")
    stt = FasterWhisperSTT(model_size="tiny.en")
    await stt.warmup()
    print("  FasterWhisperSTT warmed up (tiny.en, int8 CPU).")

    tts = PocketTTSEngine()
    await tts.start()
    print("  PocketTTSEngine resident server ready (24kHz float32).")

    sink = SubprocessAudioSink(sample_rate=24000, audio_format="f32")
    provider = ProviderClient(
        base_url="https://openrouter.ai/api/v1",
        model=model_id,
        api_key=api_key,
        max_tokens=256,
    )
    runtime = TaskRuntime(workspace=workspace, provider=provider)
    runtime.create_receptionist()

    # 2. Prepare speech audio (2.5s speech from scratch/test_tts.wav resampled to 16kHz PCM)
    test_wav = Path("scratch/test_tts.wav")
    raw = test_wav.read_bytes()
    data_idx = raw.find(b"data") + 8
    f32_24k = np.frombuffer(raw[data_idx:], dtype=np.float32)
    n_16k = int(len(f32_24k) * 16000 / 24000)
    f32_16k = np.interp(
        np.linspace(0, len(f32_24k) - 1, n_16k),
        np.arange(len(f32_24k)),
        f32_24k,
    )
    audio_bytes = (f32_16k * 32767).astype(np.int16).tobytes()
    print(f"  Prepared test speech: {len(audio_bytes) / 32000:.2f}s audio.")

    # 3. Benchmark End-to-End Voice Loop
    print("\n[2/4] Executing PTT release -> STT -> Receptionist -> PocketTTS...")
    t0_release = time.perf_counter()

    # Step 1: STT
    transcription = await stt.transcribe(audio_bytes, sample_rate=16000)
    t1_stt = time.perf_counter()
    stt_latency = (t1_stt - t0_release) * 1000
    print(f"  [STT Result]: {transcription!r} (in {stt_latency:.1f}ms)")

    # Step 2: Stream receptionist & measure time to first sentence
    t2_first_sentence = 0.0
    first_sentence = ""
    current_buf = ""

    accumulated_response = ""
    async for event in runtime.step_receptionist(transcription):
        if isinstance(event, TextDelta):
            accumulated_response += event.text
            current_buf += event.text
            if not first_sentence:
                sentences = split_sentences(current_buf)
                if len(sentences) > 1:
                    first_sentence = sentences[0]
                    t2_first_sentence = time.perf_counter()
        elif isinstance(event, TurnComplete):
            if not first_sentence and current_buf.strip():
                first_sentence = current_buf.strip()
                t2_first_sentence = time.perf_counter()

    if t2_first_sentence == 0.0:
        t2_first_sentence = time.perf_counter()

    llm_first_sentence_latency = (t2_first_sentence - t1_stt) * 1000
    print(
        f"  [LLM First Sentence]: {first_sentence!r} (in {llm_first_sentence_latency:.1f}ms)"
    )
    print(f"  [Full Response]: {accumulated_response.strip()!r}")

    # Step 3: PocketTTS First-Chunk Synthesis
    t3_first_chunk = 0.0
    first_audio_chunk: bytes = b""
    async for chunk in tts.synthesize(first_sentence):
        if not first_audio_chunk:
            first_audio_chunk = chunk
            t3_first_chunk = time.perf_counter()
            break

    tts_first_chunk_latency = (t3_first_chunk - t2_first_sentence) * 1000
    print(
        f"  [PocketTTS First Chunk]: {len(first_audio_chunk)} bytes (in {tts_first_chunk_latency:.1f}ms)"
    )

    # Step 4: Play first audio chunk to speakers
    t4_played = time.perf_counter()
    await sink.play(first_audio_chunk)
    sink_latency = (time.perf_counter() - t4_played) * 1000

    total_ttfa = (t3_first_chunk - t0_release) * 1000
    print("\n[3/4] Latency Scoreboard (Key Release to First Audio):")
    print(f"  - Local STT Transcription:        {stt_latency:6.1f} ms")
    print(f"  - LLM Time-to-First-Sentence:     {llm_first_sentence_latency:6.1f} ms")
    print(f"  - PocketTTS First-Audio-Chunk:    {tts_first_chunk_latency:6.1f} ms")
    print(f"  - Audio Sink Playback Latency:    {sink_latency:6.1f} ms")
    print("  -----------------------------------------------")
    print(
        f"  * TOTAL TIME TO FIRST AUDIO:      {total_ttfa:6.1f} ms ({total_ttfa / 1000:.2f}s)"
    )

    # 4. Workspace Action Validation
    print(
        "\n[4/4] Validating Receptionist Workspace Action (No clarifying question)..."
    )
    ws_prompt = "Look at the files in the workspace src directory and list what modules we have."
    print(f"  User input: {ws_prompt!r}")

    spawned = False
    async for event in runtime.step_receptionist(ws_prompt):
        from verbix.core.events import ToolCallStarted

        if isinstance(event, ToolCallStarted) and event.tool_name == "task_spawn":
            spawned = True
            print(f"  SUCCESS: task_spawn invoked with args: {event.arguments}")
        elif isinstance(event, TextDelta):
            pass
        elif isinstance(event, TurnComplete):
            print(f"  Receptionist reply: {event.final_text!r}")

    assert spawned, "Expected task_spawn to be called when user refers to workspace!"
    print(
        "  Task successfully spawned for workspace query without asking for clarification."
    )

    # Cleanup
    await tts.aclose()
    print("\nAll checks passed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
