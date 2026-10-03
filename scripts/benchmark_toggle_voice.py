from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import numpy as np

from verbix.cli import load_env
from verbix.core.models import CanonicalMessage
from verbix.provider.client import ProviderClient
from verbix.voice.pocket_tts import PocketTTSEngine
from verbix.voice.stt import FasterWhisperSTT


async def main() -> None:
    load_env()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found.")
        return

    # 1. Initialize STT and PocketTTS
    print("Initializing FasterWhisperSTT and PocketTTSEngine...")
    stt = FasterWhisperSTT(model_size="tiny.en")
    await stt.warmup()

    tts = PocketTTSEngine(voice="alba")
    await tts.load()

    provider = ProviderClient(
        base_url="https://openrouter.ai/api/v1",
        model="google/gemini-3.5-flash-lite",
        api_key=api_key,
        max_tokens=256,
    )

    # 2. Prepare test speech audio (2.56s utterance: "The build finished successfully in 12 seconds.")
    wav_path = Path("scratch/test_tts.wav")
    if not wav_path.exists():
        print("ERROR: scratch/test_tts.wav not found.")
        return

    raw = wav_path.read_bytes()
    idx = raw.find(b"data") + 8
    f32_24k = np.frombuffer(raw[idx:], dtype=np.float32)
    n_16k = int(len(f32_24k) * 16000 / 24000)
    audio_16k = np.interp(
        np.linspace(0, len(f32_24k) - 1, n_16k),
        np.arange(len(f32_24k)),
        f32_24k,
    )
    test_pcm = (audio_16k * 32767).astype(np.int16).tobytes()

    print(
        f"Test audio: {len(test_pcm)} bytes ({len(audio_16k) / 16000:.2f}s of 16kHz PCM)"
    )
    print("Beginning 10 repeated benchmark runs...\n")

    stt_latencies: list[float] = []
    ttft_latencies: list[float] = []
    ttfs_latencies: list[float] = []
    tts_chunk_latencies: list[float] = []
    total_ttfa_latencies: list[float] = []

    for run_idx in range(1, 11):
        # A. Simulate speech capture with LiveSTT running in background during key hold
        live_session = stt.create_live_session(sample_rate=16000)
        live_session.start()

        chunk_size = 1600  # 50ms chunks
        for i in range(0, len(test_pcm), chunk_size):
            live_session.feed(test_pcm[i : i + chunk_size])
            await asyncio.sleep(0.045)  # slightly faster than real-time

        # 150ms natural pause before toggle-off
        live_session.feed(b"\x00" * 4800)
        await asyncio.sleep(0.15)

        # B. User toggles off!
        t_toggle_off = time.perf_counter()

        # 1. Finalize STT
        transcription = await live_session.finish()
        t_stt_done = time.perf_counter()
        stt_latency = (t_stt_done - t_toggle_off) * 1000

        # 2. LLM streaming with minimal thinking (reasoning_effort="low")
        system_prompt = (
            "You are Verbix Receptionist, an orchestration assistant. "
            "Reply concisely and directly in one short conversational sentence."
        )
        messages = [
            CanonicalMessage(role="system", content=system_prompt),
            CanonicalMessage(
                role="user", content=transcription or "Where is the workspace?"
            ),
        ]

        t_llm_start = time.perf_counter()
        t_first_token: float | None = None
        t_first_sentence: float | None = None
        accumulated_text = ""
        first_sentence = ""

        from verbix.provider.models import TextDelta as ProviderTextDelta

        async for event in provider.stream(messages, reasoning_effort="low"):
            now = time.perf_counter()
            if isinstance(event, ProviderTextDelta) and event.text:
                if t_first_token is None:
                    t_first_token = now
                accumulated_text += event.text
                if t_first_sentence is None and any(
                    p in accumulated_text for p in (". ", "! ", "? ", ".\n", "!\n")
                ):
                    t_first_sentence = now
                    first_sentence = accumulated_text.strip()
                    break

        if not first_sentence:
            first_sentence = accumulated_text.strip()
        if t_first_sentence is None:
            t_first_sentence = time.perf_counter()
        if t_first_token is None:
            t_first_token = t_first_sentence

        ttft = (t_first_token - t_llm_start) * 1000
        ttfs = (t_first_sentence - t_llm_start) * 1000

        # 3. PocketTTS synthesis to first audio chunk
        t_tts_start = time.perf_counter()
        first_pcm_chunk_time: float | None = None

        async for _chunk in tts.synthesize(first_sentence):
            if first_pcm_chunk_time is None:
                first_pcm_chunk_time = time.perf_counter()
                break  # only measure time to first chunk

        t_first_audio = first_pcm_chunk_time or time.perf_counter()
        tts_chunk_latency = (t_first_audio - t_tts_start) * 1000
        total_ttfa = (t_first_audio - t_toggle_off) * 1000

        stt_latencies.append(stt_latency)
        ttft_latencies.append(ttft)
        ttfs_latencies.append(ttfs)
        tts_chunk_latencies.append(tts_chunk_latency)
        total_ttfa_latencies.append(total_ttfa)

        print(
            f"Run {run_idx:2d}: STT={stt_latency:5.1f}ms | TTFT={ttft:5.1f}ms | "
            f"TTFS={ttfs:5.1f}ms | TTS={tts_chunk_latency:5.1f}ms | Total TTFA={total_ttfa:6.1f}ms "
            f"-> '{first_sentence[:40]}...'"
        )

    await tts.aclose()

    def stats(arr: list[float]) -> str:
        s = sorted(arr)
        p50 = s[len(s) // 2]
        p95 = s[int(len(s) * 0.95)]
        return f"p50: {p50:6.1f} ms | p95: {p95:6.1f} ms"

    print("\n" + "=" * 70)
    print("VOICE LATENCY BREAKDOWN (10 Runs):")
    print("=" * 70)
    print(f"STT Finalize Latency (Live STT during hold):  {stats(stt_latencies)}")
    print(f"LLM Time-to-First-Token (TTFT, effort=low):  {stats(ttft_latencies)}")
    print(f"LLM Time-to-First-Sentence (TTFS):          {stats(ttfs_latencies)}")
    print(f"PocketTTS Synthesis to First Audio Chunk:   {stats(tts_chunk_latencies)}")
    print("-" * 70)
    print(f"TOTAL KEY-RELEASE TO FIRST AUDIO (TTFA):    {stats(total_ttfa_latencies)}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
