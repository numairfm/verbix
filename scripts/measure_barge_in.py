from __future__ import annotations

import asyncio
import time

from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime
from verbix.voice.base import AudioSink
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer
from verbix.voice.pocket_tts import PocketTTSEngine
from verbix.voice.stt import FasterWhisperSTT


class MockSink(AudioSink):
    """Sink that simulates real-time playback and immediate cancellation."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.channels = 1
        self.audio_format = "f32"
        self._playing = False
        self._stop_requested = False
        self.chunks_received = 0

    @property
    def is_playing(self) -> bool:
        return self._playing

    async def play(self, audio_data: bytes) -> bool:
        self._playing = True
        self.chunks_received += 1
        # Each chunk is ~80ms of audio
        n_samples = len(audio_data) // 4
        chunk_dur = n_samples / self.sample_rate

        # Sleep in 5ms slices to allow rapid interruption
        slices = max(1, int(chunk_dur / 0.005))
        for _ in range(slices):
            if self._stop_requested:
                self._playing = False
                return False
            await asyncio.sleep(0.005)

        self._playing = False
        return True

    def stop(self) -> None:
        self._stop_requested = True
        self._playing = False

    def reset(self) -> None:
        self._playing = False
        self._stop_requested = False
        self.chunks_received = 0


def calc_stats(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    s = sorted(values)
    p50 = s[len(s) // 2]
    p95 = s[int(len(s) * 0.95)]
    return p50, p95


async def measure_loop_lag(duration_sec: float) -> list[float]:
    """Records event loop scheduling jitter every 5ms."""
    delays: list[float] = []
    t_end = time.perf_counter() + duration_sec
    target = 0.005
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        await asyncio.sleep(target)
        actual = time.perf_counter() - t0
        jitter = max(0.0, (actual - target) * 1000.0)
        delays.append(jitter)
    return delays


async def main() -> None:
    print("=" * 70)
    print("BARGE-IN & CONCURRENCY LATENCY BENCHMARK (10 Runs)")
    print("=" * 70)

    # Initialize components
    tts = PocketTTSEngine(voice="alba")
    await tts.load()
    sink = MockSink()
    stt = FasterWhisperSTT(model_size="tiny.en")

    # Minimal mock runtime and pipeline
    dummy_provider = ProviderClient(
        base_url="https://openrouter.ai/api/v1", model="test"
    )
    from pathlib import Path

    runtime = TaskRuntime(workspace=Path("."), provider=dummy_provider)
    player = AudioPlayer(tts=tts, sink=sink, runtime=runtime)
    pipeline = VoicePipeline(runtime=runtime, stt=stt, player=player)

    long_sentence = (
        "This is an extensive dialogue response intended to test barge-in interruption. "
        "The assistant will keep talking about various build and code aspects until interrupted."
    )

    barge_in_stop_latencies: list[float] = []
    tts_cancel_latencies: list[float] = []
    loop_jitters: list[float] = []

    for run_idx in range(1, 11):
        sink.reset()

        # 1. Start event loop lag monitor concurrently
        lag_task = asyncio.create_task(measure_loop_lag(0.6))

        # 2. Begin playback in background task
        play_task = asyncio.create_task(
            player.speak_foreground_sentences([long_sentence])
        )

        # Wait until playback actively begins and at least 2 chunks have arrived
        while sink.chunks_received < 2:
            await asyncio.sleep(0.01)

        # 3. Simulate user pressing toggle (barge-in!)
        t_toggle = time.perf_counter()
        toggle_res = await pipeline.toggle()

        # Measure time until sink is verified stopped
        while sink.is_playing:
            await asyncio.sleep(0.001)
        t_sink_stopped = time.perf_counter()

        # Measure time until play_task has completed/cancelled
        try:
            await asyncio.wait_for(play_task, timeout=0.5)
        except (TimeoutError, asyncio.CancelledError):
            pass
        t_task_done = time.perf_counter()

        barge_in_ms = (t_sink_stopped - t_toggle) * 1000.0
        tts_cancel_ms = (t_task_done - t_toggle) * 1000.0

        barge_in_stop_latencies.append(barge_in_ms)
        tts_cancel_latencies.append(tts_cancel_ms)

        jitters = await lag_task
        loop_jitters.extend(jitters)

        # Clean up capture started by toggle
        if pipeline.capture.is_recording:
            await pipeline.capture.stop_recording()

        print(
            f"Run {run_idx:2d}: Barge-in stop={barge_in_ms:4.2f}ms | "
            f"TTS Task Terminated={tts_cancel_ms:4.2f}ms | "
            f"Action='{toggle_res.get('action')}'"
        )
        await asyncio.sleep(0.05)

    await tts.aclose()

    b_p50, b_p95 = calc_stats(barge_in_stop_latencies)
    c_p50, c_p95 = calc_stats(tts_cancel_latencies)
    j_p50, j_p95 = calc_stats(loop_jitters)

    print("\n" + "=" * 70)
    print("BARGE-IN & CONCURRENCY BENCHMARK RESULTS (p50 / p95)")
    print("=" * 70)
    print(
        f"Barge-in (Toggle to Audio Sink Stopped): p50 = {b_p50:5.2f} ms | p95 = {b_p95:5.2f} ms"
    )
    print(
        f"TTS Generator Cancellation Completed:  p50 = {c_p50:5.2f} ms | p95 = {c_p95:5.2f} ms"
    )
    print(
        f"Event Loop Lag during TTS Synthesis:    p50 = {j_p50:5.2f} ms | p95 = {j_p95:5.2f} ms"
    )
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
