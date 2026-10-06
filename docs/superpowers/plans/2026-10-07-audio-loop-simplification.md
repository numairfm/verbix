# 3-State Audio Architecture & Announcer Collapse Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify the Verbix audio loop into a rock-solid, single-speaker 3-state system (`IDLE`, `LISTENING`, `SPEAKING`) by collapsing the redundant vocal announcer agent into silent task ledger completion, eliminating audio collisions and speaker bleed into the microphone.

**Architecture:** The foreground assistant becomes the sole entity connected to speakers. Background workers complete silently into the `TaskLedger`; their results are immediately visible to the foreground assistant in memory without unsolicited vocal interruptions. `SpeechOutput` provides uninterrupted FIFO queueing with clean sequential playback. When `LISTENING`, speaker playback is hard-muted/stopped so Verbix cannot speak over the user's recording. All states and callbacks are designed as clean, configurable interfaces to allow easy binding by future desktop UIs.

**Tech Stack:** Python 3.12+, `asyncio`, `sounddevice`, `pytest-asyncio`.

**Spec:** Conversational audio state model:
1. `IDLE`: No active turn. Speakers and microphone quiet.
2. `LISTENING`: User owns the floor. Microphone is active; speaker playback is hard-muted/stopped.
3. `SPEAKING`: Assistant owns the floor, covering its thinking/generation time and audio playback. Barge-in (user pressing PTT) halts playback immediately.
4. Background tasks finish silently into `TaskLedger` memory; vocalized only when the user explicitly asks.

## Global Constraints

- Zero new external dependencies.
- Strict PTT (push-to-talk); no open-mic loops.
- Single owner of speech: only the foreground assistant may synthesize and play speech.
- All existing tests (153 passed) must remain green.
- Ponytail minimalism: delete code before adding code.
- Clean interfaces: states, callbacks, and mute behavior are configurable and observable for future UI integration.

## Review Focus

1. **Hardware buffer drain on barge-in:** Calling `turn_output.interrupt()` must abort active sink streams immediately so no residual sound bleeds into the opened microphone.
2. **Back-to-back utterance FIFO ordering:** Submitting a second utterance *while* the first is actively playing must queue and play in exact FIFO order without dropping or truncating either utterance.
3. **Rapid PTT taps:** Double-tapping Enter within 100ms must cleanly reset state to `IDLE` via VAD silence gating without deadlocking the floor lock.
4. **Task ledger visibility:** When a background task completes while the user is idle, the foreground assistant's next turn prompt must contain the completed task status without needing manual refresh.
5. **Clean teardown:** Terminating the runtime or CLI (`Ctrl+C`, `exit`) must cleanly cancel all background workers and drain queues without orphaned asyncio tasks.

---

### Task 1: Collapse Redundant Announcer & Silent Task Completion

**Files:**
- Modify: `src/verbix/runtime/runtime.py`
- Modify / Update: `tests/test_runtime_e2e.py`, `tests/test_announcer_loop.py`, `tests/test_runtime_races.py`
- Delete: `src/verbix/runtime/announcer.py`
- Delete: `src/verbix/minisweagent/config/announcer.yaml`
- Test: `tests/test_silent_task_completion.py`

**Interfaces:**
- Consumes: `TaskLedger.mark_completed(task_id, result)`, `WorkerTask.on_completed(callback)`
- Produces: `Runtime._on_task_finished(task_id: str, result: str)` (updates ledger and context; zero speech emission)

**Backward Compatibility & Test Migration Note:**
- `FloorState`: keep `FOREGROUND = "SPEAKING"` and `ANNOUNCING = "SPEAKING"` as aliases so tests in `test_ledger_and_floor.py` pass without breakage.
- In `test_runtime_e2e.py` and `test_announcer_loop.py`: migrate tests that checked `self.announcer` vocal synthesis to verify that task completions record silently into `TaskLedger` and prompt context instead of invoking vocal synthesis.

- [x] **Step 1: Write the failing test for silent task completion and batch coalescing**

```python
# tests/test_silent_task_completion.py
import asyncio
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from verbix.runtime.runtime import Runtime
from verbix.runtime.floor import FloorState

@pytest.mark.asyncio
async def test_worker_completion_updates_ledger_without_speech(tmp_path):
    spoken = []
    class FakeOutput:
        async def emit(self, event):
            spoken.append(event)
        def delivered_text(self):
            return ""

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    rt.wake_output = FakeOutput()
    
    # Register and complete multiple tasks in rapid succession
    rt.ledger.register_task("task-10", "check git")
    rt.ledger.register_task("task-11", "run tests")
    
    rt._on_task_finished("task-10", "git clean")
    rt._on_task_finished("task-11", "tests passed")
    
    await asyncio.sleep(0.05)
    
    # Ledger must have both results
    assert "git clean" in rt.ledger.get_status("task-10")
    assert "tests passed" in rt.ledger.get_status("task-11")
    
    # MUST NOT have triggered unsolicited speech
    assert len(spoken) == 0
    # Floor must remain IDLE
    assert rt.floor.state == FloorState.IDLE
    await rt.aclose()
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_silent_task_completion.py -v`
Expected: FAIL (missing `_on_task_finished` or announcer loop triggered).

- [x] **Step 3: Collapse Announcer and wire silent ledger recording in `src/verbix/runtime/runtime.py`**

1. Remove `self.announcer`, `self.announcer_loop`, and `announcer.yaml` loader.
2. In `Runtime._spawn_worker`: on worker finish, route to `self._on_task_finished(task_id, result)` which records directly into `self.ledger.mark_completed(task_id, result)` and emits `task_completed` event.
3. Remove `AnnouncerLoop` instantiation, imports, and background polling task.
4. Delete `src/verbix/runtime/announcer.py` and `src/verbix/minisweagent/config/announcer.yaml`.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_silent_task_completion.py -v`
Expected: PASS

- [x] **Step 5: Run existing suite and commit**

Run: `uv run pytest -q`
```bash
git add src/verbix/runtime/runtime.py tests/test_silent_task_completion.py
git rm src/verbix/runtime/announcer.py src/verbix/minisweagent/config/announcer.yaml
git commit -m "refactor(runtime): collapse redundant announcer agent into silent ledger completion"
```

---

### Task 2: Continuous FIFO Speech Output Queueing

**Files:**
- Modify: `src/verbix/voice/output.py`
- Test: `tests/test_speech_output_queue.py`

**Interfaces:**
- Consumes: `SpeechOutput.emit(event)`, `SpeechOutput.interrupt()`
- Produces: Seamless FIFO sequential playback of multiple utterances through `self.sink.play_stream()` without mid-stream termination or dropped frames.

- [x] **Step 1: Write the failing test verifying strict FIFO ordering during active playback race**

```python
# tests/test_speech_output_queue.py
import asyncio
import pytest
from verbix.core.events import TextDelta, TurnComplete
from verbix.voice.output import SpeechOutput

@pytest.mark.asyncio
async def test_consecutive_utterances_play_in_strict_fifo_order_during_active_playback():
    played_chunks = []
    first_playing = asyncio.Event()

    class DelayedTTS:
        async def synthesize(self, text):
            if "First" in text:
                first_playing.set()
                # Simulate slow first sentence streaming
                await asyncio.sleep(0.08)
            yield f"audio({text})"

    class FakeSink:
        def __init__(self):
            self.stopped = False
        def stop(self):
            self.stopped = True
        async def play_stream(self, generator):
            async for chunk in generator:
                played_chunks.append(chunk)

    sink = FakeSink()
    out = SpeechOutput(tts=DelayedTTS(), sink=sink)

    # Dispatch first utterance
    await out.emit(TextDelta(text="First sentence."))
    await out.emit(TurnComplete(final_text="First sentence."))

    # Wait until first utterance is actively synthesizing/playing
    await asyncio.wait_for(first_playing.wait(), timeout=1.0)

    # Dispatch second utterance WHILE first is actively playing
    await out.emit(TextDelta(text="Second sentence."))
    await out.emit(TurnComplete(final_text="Second sentence."))

    # Wait for both utterances to drain completely
    for _ in range(50):
        if len(played_chunks) >= 2:
            break
        await asyncio.sleep(0.02)

    # MUST play in exact FIFO sequence: First, then Second
    assert played_chunks == ["audio(First sentence.)", "audio(Second sentence.)"]
    assert sink.stopped is False
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_speech_output_queue.py -v`
Expected: FAIL (second chunk dropped or stream closed prematurely by `TurnComplete` end frame).

- [x] **Step 3: Update `SpeechOutput` in `src/verbix/voice/output.py`**

1. Remove `("end", "")` frame emission on `TurnComplete` that closes the generator while subsequent frames are queued.
2. In `_feed`: keep generator yielding audio as long as the current `epoch` matches `self._epoch`.
3. Ensure FIFO queue processing: subsequent utterances wait for the previous utterance's generator chunks to finish naturally.
4. In `interrupt()`: bump `self._epoch += 1`, clear frame queue, call `self.sink.stop()`.

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_speech_output_queue.py -v`
Expected: PASS

- [x] **Step 5: Run existing suite and commit**

Run: `uv run pytest -q`
```bash
git add src/verbix/voice/output.py tests/test_speech_output_queue.py
git commit -m "fix(voice): enforce continuous FIFO speech output queueing across concurrent utterances"
```

---

### Task 3: 3-State Floor Arbiter & Speaker Mute while LISTENING

**Files:**
- Modify: `src/verbix/runtime/floor.py`
- Modify: `src/verbix/cli.py`
- Test: `tests/test_audio_loop_states.py`

**Interfaces:**
- Consumes: `FloorState.IDLE`, `FloorState.LISTENING`, `FloorState.SPEAKING`
- Produces: Strict mutual exclusion where `LISTENING` hard-mutes speaker playback so Verbix never speaks over user recording. Clean observable callbacks for future UI bindings.

- [x] **Step 1: Write the failing test for 3-state transitions and speaker mute semantics**

```python
# tests/test_audio_loop_states.py
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
```

- [x] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_audio_loop_states.py -v`
Expected: FAIL (missing `FloorState.SPEAKING`, `set_state`, or property helpers).

- [x] **Step 3: Update `FloorState` in `src/verbix/runtime/floor.py` and PTT mute in `src/verbix/cli.py`**

1. In `src/verbix/runtime/floor.py`:
   - Define enum `FloorState(str, Enum)`: `IDLE = "IDLE"`, `LISTENING = "LISTENING"`, `SPEAKING = "SPEAKING"`.
   - Maintain alias `FOREGROUND = "SPEAKING"` for backwards compatibility.
   - Add explicit `set_state(state: FloorState | str)` method and properties (`is_idle`, `is_listening`, `is_speaking`).
2. In `src/verbix/cli.py`:
   - When PTT key triggers recording:
     - `pipeline.turn_output.interrupt()` (immediately silences active playback)
     - `runtime.floor.set_state(FloorState.LISTENING)` (locks floor into listening)
   - When PTT key stops recording:
     - `runtime.floor.set_state(FloorState.SPEAKING)` (covers thinking + speech)
   - When turn completes and `turn_output.is_playing` is False:
     - `runtime.floor.set_state(FloorState.IDLE)`

- [x] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_audio_loop_states.py -v`
Expected: PASS

- [x] **Step 5: Run full test suite and commit**

Run: `uv run pytest -q`
```bash
git add src/verbix/runtime/floor.py src/verbix/cli.py tests/test_audio_loop_states.py
git commit -m "feat(runtime): enforce 3-state floor arbiter with speaker muting while listening"
```
