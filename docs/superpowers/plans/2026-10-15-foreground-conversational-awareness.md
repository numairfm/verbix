# Foreground Conversational & Task Awareness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the foreground agent live awareness of worker capabilities, running tasks, and newly completed results, enable proactive announcements in continuous voice mode with live barge-in and a short post-speech quiet window, eliminate double-spoken task completions, and fix pre-output stream retry on empty initial SSE chunks.

**Architecture:** Derive worker tool capabilities dynamically from `build_worker_registry()` into the foreground system prompt and `SpawnTool`. Govern task completion delivery across the announcer, foreground turn folding, and `Runtime.status()` through a single `TaskLedger.delivery` state (`"pending"` -> `"heard"` only after a turn completes OK). Have `VoicePipeline` publish `user_speaking` on `FloorArbiter` so `Runtime` can schedule proactive announcements in continuous `LISTENING` mode after a brief quiet window without reading STT internals while keeping mic/AEC barge-in live.

**Tech Stack:** Python `>=3.13,<3.15`, `asyncio`, `pydantic>=2.12`, `litellm>=1.75.5`.

**Spec:** `docs/superpowers/specs/2026-10-15-foreground-conversational-awareness-design.md`

## Global Constraints

- **Models decide, runtime coordinates, tools act:** No similarity or regex gate on `spawn` objectives. Foreground tools remain strictly `spawn`, `status`, `cancel`.
- **Dynamic worker capabilities:** Derive worker tool names and summaries from `build_worker_registry(workspace)` (including whether `bash` is enabled via `probe_sandbox().available`), never from a hardcoded string list.
- **Single delivery state:** `TaskLedger.delivery` (`"pending" | "heard" | "interrupted"`) is the single source of truth for whether a completed or failed task has been delivered to the user. A completion folded into a foreground turn or spoken by the announcer is marked `"heard"` **only after that turn completes OK (not interrupted and not error)**. `Runtime.status()` also marks inspected completed/failed tasks `"heard"`.
- **Floor-owned `user_speaking` seam:** `VoicePipeline` publishes `floor.set_user_speaking(bool)` onto `FloorArbiter`; `Runtime` never reads `_live_stt` or STT internals.
- **Concise, general prompt rules:** Limit external-facts delegation to facts that change (news, live prices, current status), prefer the most recent exchange for short follow-ups, phrase the incomplete-fragment rule generally without word blacklists, and briefly report newly completed results before answering.

## Review Focus

1. **Foreground turn interrupted mid-speech after folding a pending task completion (Task 2):** If a foreground turn folds a `"pending"` task completion into its grounding envelope and the user barges in (`cancel_turn` / `CancelledError`) before the turn completes, the task's `delivery` state in `TaskLedger` must remain `"pending"` (never prematurely marked `"heard"`).
2. **Multiple tasks completing while busy, then one inspected via `status(task_id="task-1")` while `"task-2"` remains uninspected (Task 2):** Calling `runtime.status(task_id="task-1")` must mark only `"task-1"` as `"heard"` while leaving `"task-2"` as `"pending"` so `"task-2"` is still delivered.
3. **User begins speaking during the continuous-mode post-speech quiet window before `drain_pending_announcements` acquires the floor (Task 3):** If `floor.set_user_speaking(True)` fires while the quiet-window timer is waiting, the scheduled announcement drain must abort so Verbix never speaks over the user's new utterance.
4. **Barge-in during a continuous-mode proactive announcement (Task 3):** When the announcer is speaking in continuous mode (`FloorState.SPEAKING`), `PTTCapture` and `EchoCanceller` must remain active and route mic chunks to `_live_stt.feed_barge_in()` so user speech interrupts the announcement immediately.
5. **OpenRouter stream sending multiple empty role/keepalive SSE chunks before raising `MidStreamFallbackError` (Task 4):** `LitellmModel.query_stream()` must leave `yielded_any = False` across any number of empty/keepalive chunks until a `ThoughtDelta`, `TextDelta`, or `ToolCallDelta` is actually yielded.

---

### Task 1: Dynamic Worker Tool Capability Grounding

**Files:**
- Modify: `src/verbix/runtime/workers.py:22-49`
- Modify: `src/verbix/runtime/runtime.py:149-169`
- Modify: `src/verbix/tools/spawn.py:11-36`
- Test: `tests/test_runtime_e2e.py`

**Interfaces:**
- Consumes: `build_worker_registry(workspace: Path | None = None) -> ToolRegistry` in `src/verbix/runtime/workers.py`.
- Produces: `describe_worker_tools(workspace: Path | None = None) -> tuple[list[str], str]` in `src/verbix/runtime/workers.py`, computed **once per `Runtime`** in `build_foreground_agent()` (`runtime.worker_tool_names` and `runtime.worker_capabilities`) and reused by `Runtime._build_grounded_prompt()` and `SpawnTool.description`.

- [ ] **Step 1: Write the failing test in `tests/test_runtime_e2e.py`**

```python
def test_foreground_grounding_and_spawn_tool_derive_worker_capabilities_once_per_runtime(
    tmp_path, monkeypatch
):
    from verbix.runtime import workers as workers_mod
    from verbix.tools.sandbox import SandboxStatus
    from verbix.tools.spawn import SpawnTool

    calls = 0
    orig_describe = getattr(workers_mod, "describe_worker_tools", None)

    # 1. When sandbox is available, worker capabilities include screenshot, web_search, and bash
    monkeypatch.setattr(
        "verbix.tools.sandbox.probe_sandbox",
        lambda allow_network=False: SandboxStatus(
            available=True, backend="bwrap", reason="ok", capabilities="sandboxed bash"
        ),
    )
    rt_with_bash = Runtime(workspace=tmp_path, provider=MagicMock())
    assert "screenshot" in rt_with_bash.system_prompt
    assert "web_search" in rt_with_bash.system_prompt
    assert "web_fetch" in rt_with_bash.system_prompt
    assert "bash" in rt_with_bash.system_prompt
    assert "screenshot" in SpawnTool(rt_with_bash).description
    assert hasattr(rt_with_bash, "worker_capabilities")
    assert hasattr(rt_with_bash, "worker_tool_names")

    # 2. When sandbox is unavailable, bash is automatically omitted from worker capabilities
    monkeypatch.setattr(
        "verbix.tools.sandbox.probe_sandbox",
        lambda allow_network=False: SandboxStatus(
            available=False, backend=None, reason="no bwrap", capabilities=""
        ),
    )
    rt_no_bash = Runtime(workspace=tmp_path, provider=MagicMock())
    assert "screenshot" in rt_no_bash.system_prompt
    assert "bash" not in rt_no_bash.system_prompt
    assert "bash" not in SpawnTool(rt_no_bash).description
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runtime_e2e.py::test_foreground_grounding_and_spawn_tool_derive_worker_capabilities_once_per_runtime -v`
Expected: FAIL (`AssertionError: assert 'screenshot' in rt_with_bash.system_prompt`)

- [ ] **Step 3: Implement `describe_worker_tools()` in `src/verbix/runtime/workers.py`, cache on `Runtime` in `build_foreground_agent()`, and reuse in `_build_grounded_prompt()` and `SpawnTool.description`**

1. In `src/verbix/runtime/workers.py`:
   ```python
   def describe_worker_tools(workspace: Path | None = None) -> tuple[list[str], str]:
       """Derives worker tool names and a concise summary from the live worker registry."""
       reg = build_worker_registry(workspace=workspace)
       names: list[str] = []
       parts: list[str] = []
       for d in reg.get_definitions():
           fn = d.get("function", {})
           name = fn.get("name", "")
           desc = (fn.get("description", "") or "").strip()
           short = desc.split(". ")[0].rstrip(".")
           if name:
               names.append(name)
               parts.append(f"{name} ({short})" if short else name)
       return names, "; ".join(parts)
   ```
2. In `src/verbix/runtime/foreground.py` (`build_foreground_agent`):
   - Compute `runtime.worker_tool_names, runtime.worker_capabilities = describe_worker_tools(runtime.workspace)` once before calling `runtime._build_grounded_prompt()`.
3. In `src/verbix/runtime/runtime.py:157-165` (`_build_grounded_prompt`):
   - Reuse `getattr(self, "worker_capabilities", "")` to include `\n- Background worker capabilities (via spawn): {self.worker_capabilities}.` in `grounding`.
4. In `src/verbix/tools/spawn.py`:
   - Make `description` a `@property` on `SpawnTool` that reads `getattr(self.runtime, "worker_tool_names", None)` and returns:
     `f"Starts an autonomous background worker to execute a task. Workers have access to: {', '.join(names)}."` (falling back to `"Starts an autonomous background worker to execute a task."` if empty).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_runtime_e2e.py::test_foreground_grounding_and_spawn_tool_derive_worker_capabilities_dynamically -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/workers.py src/verbix/runtime/runtime.py src/verbix/tools/spawn.py tests/test_runtime_e2e.py
git commit -m "feat(runtime,tools): derive worker capabilities dynamically from worker registry"
```

---

### Task 2: Single-State Completion Delivery (`TaskLedger.delivery`), Active/Completed Task Grounding & `status()` Heard Marking

**Files:**
- Modify: `src/verbix/runtime/ledger.py:19-73`
- Modify: `src/verbix/runtime/workers.py:51-97`
- Modify: `src/verbix/runtime/announcer.py:132-161`
- Modify: `src/verbix/runtime/runtime.py:171-277, 367-398`
- Test: `tests/test_idle_drained_announcements.py`, `tests/test_runtime_e2e.py`

**Interfaces:**
- Consumes: `TaskLedger.mark_heard(task_id: str)`, `TaskEntry.delivery`.
- Produces:
  - `TaskLedger.pending_deliveries() -> list[TaskEntry]` in `src/verbix/runtime/ledger.py` (returns completed/failed entries where `delivery == "pending"`).
  - `Runtime._request_grounding(interruption: Any = None, *, include_tasks: bool = False) -> tuple[str, list[str]]` (or helper) injecting `[Active background tasks: ...]` and `[Newly completed background tasks (report briefly before answering): ...]` into foreground user turns, marking folded tasks `heard` only after the turn completes OK.
  - `Runtime.status(task_id: str | None = None, full: bool = False) -> str` marking any returned completed/failed task `heard` in `self.ledger` and pruning it from `self._pending_announcements`.

- [ ] **Step 1: Write the failing reproduction tests in `tests/test_idle_drained_announcements.py` and `tests/test_runtime_e2e.py`**

```python
# In tests/test_idle_drained_announcements.py:
@pytest.mark.asyncio
async def test_pending_completion_folded_into_foreground_turn_is_spoken_once_not_twice(tmp_path):
    spoken_turns = []

    class FakeOutput:
        def __init__(self):
            self.buf = []
        async def emit(self, event):
            if isinstance(event, TextDelta):
                self.buf.append(event.text)
        def delivered_text(self):
            return "".join(self.buf)

    seen_user_messages = []
    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        seen_user_messages.append(messages[-1].content if messages else "")
        yield ProviderTextDelta(text="Nvidia agreed to buy Hugging Face for 12.9B.")
        yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    wake_out = FakeOutput()
    rt.wake_output = wake_out

    # Task finishes while floor is busy (so announcer does not run yet)
    rt.floor.set_state(FloorState.LISTENING)
    rt.floor.set_user_speaking(True)
    rt.ledger.register_task("task-2", "Verify if NVIDIA acquired Hugging Face")
    rt._on_task_finished("task-2", "Agreed to buy for $12.9B.")
    assert rt.ledger.get("task-2").delivery == "pending"

    # User asks "What's the status?" -> completion is folded into the foreground turn
    rt.floor.set_user_speaking(False)
    turn_out = FakeOutput()
    reply = await rt.submit("What's the status?", turn_out)
    assert "12.9B" in reply
    assert "Agreed to buy for $12.9B." in seen_user_messages[0]
    assert rt.ledger.get("task-2").delivery == "heard"

    # Wait after turn ends: announcer MUST NOT speak task-2 a second time!
    await asyncio.sleep(0.1)
    assert wake_out.delivered_text() == ""
    await rt.aclose()


@pytest.mark.asyncio
async def test_interrupted_foreground_turn_leaves_folded_completion_pending(tmp_path):
    provider = MagicMock()

    async def slow_stream(messages, tools=None, **kw):
        yield ProviderTextDelta(text="Starting to tell you...")
        await asyncio.sleep(10.0)

    provider.stream = slow_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    rt.floor.set_state(FloorState.LISTENING)
    rt.floor.set_user_speaking(True)
    rt.ledger.register_task("task-2", "Verify deal")
    rt._on_task_finished("task-2", "Deal confirmed.")

    task = asyncio.create_task(rt.submit("What happened?"))
    await asyncio.sleep(0.05)
    await rt.cancel_turn(heard="")
    with pytest.raises(asyncio.CancelledError):
        await task

    # Because the foreground turn was interrupted before completing OK, delivery stays pending!
    assert rt.ledger.get("task-2").delivery == "pending"
    await rt.aclose()


@pytest.mark.asyncio
async def test_status_tool_marks_only_inspected_completed_task_heard(tmp_path):
    from verbix.runtime.worker import WorkerTask

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    rt.floor.set_state(FloorState.SPEAKING)
    for tid, res in (("task-1", "res 1"), ("task-2", "res 2")):
        t = WorkerTask(id=tid, description=f"job {tid}", session=MagicMock(messages=[]))
        t._status = "completed"
        t._result = res
        rt.tasks[tid] = t
        rt.ledger.register_task(tid, f"job {tid}")
        rt._on_task_finished(tid, res)

    assert rt.ledger.get("task-1").delivery == "pending"
    assert rt.ledger.get("task-2").delivery == "pending"

    out = rt.status(task_id="task-1")
    assert "res 1" in out
    assert rt.ledger.get("task-1").delivery == "heard"
    assert rt.ledger.get("task-2").delivery == "pending"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_idle_drained_announcements.py::test_pending_completion_folded_into_foreground_turn_is_spoken_once_not_twice tests/test_idle_drained_announcements.py::test_status_tool_marks_only_inspected_completed_task_heard -v`
Expected: FAIL

- [ ] **Step 3: Implement single-state completion delivery, active/completed task grounding, and `status()` heard marking**

1. In `src/verbix/runtime/ledger.py`:
   - Add `def pending_deliveries(self) -> list[TaskEntry]: return [e for e in self._tasks.values() if e.status in ("completed", "failed") and e.delivery == "pending"]`.
2. In `src/verbix/runtime/workers.py:51-97`:
   - In `finish_runtime_task()` and `fail_runtime_task()`, remove the immediate `runtime.foreground.messages.append(...)` block so completions are only added to `foreground.messages` when delivered (either folded into a foreground turn or spoken by `drain_pending_announcements`).
3. In `src/verbix/runtime/runtime.py`:
   - Update `_request_grounding(self, interruption: Any = None, *, include_tasks: bool = False) -> tuple[str, list[str]]` (keeping `_request_grounding(interruption) -> str` backwards-compatible, and adding `_foreground_grounding(self, interruption: Any = None) -> tuple[str, list[str]]`):
     - Include `[Active background tasks: ...]` for all `self.list_active_tasks()`.
     - Include `[Newly completed background tasks (report briefly before answering): ...]` for all `self.ledger.pending_deliveries()`, returning their `task_id` list as `folded_ids`.
   - In `_step_inner()` / `step()` / `submit()`:
     - When running on `self.foreground` (`session is self.foreground`), call `_foreground_grounding(interruption)` to prepend active and newly completed tasks to `user_text`.
     - Only when the foreground turn finishes normally without cancellation or error (`not was_interrupted` and `exit_reason == "ok"`), mark each `tid` in `folded_ids` as `self.ledger.mark_heard(tid)` and remove `tid` from `self._pending_announcements`.
   - In `Runtime.status(self, task_id: str | None = None, full: bool = False) -> str`:
     - When `task_id` is given and `task.status in ("completed", "failed")`, mark `self.ledger.mark_heard(task.id)` and remove `task.id` from `self._pending_announcements`.
     - When `task_id is None`, for each task in `recent_done`, mark `self.ledger.mark_heard(t.id)` and remove `t.id` from `self._pending_announcements`.
4. In `src/verbix/runtime/announcer.py`:
   - In `drain_pending_announcements(runtime)`: filter `batch` so any `task_id` whose `runtime.ledger.get(task_id)` already has `delivery == "heard"` or `"interrupted"` is skipped.
   - When `final_text` is produced (and the announcer turn was not interrupted), append `{"role": "user", "content": "\n".join(f"[Task '{tid}' completed]: {res}" for tid, res in batch)}` and `{"role": "assistant", "content": final_text}` to `runtime.foreground.messages`, and mark `runtime.ledger.mark_heard(tid)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_idle_drained_announcements.py tests/test_announcer_config.py tests/test_runtime_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/ledger.py src/verbix/runtime/workers.py src/verbix/runtime/announcer.py src/verbix/runtime/runtime.py tests/test_idle_drained_announcements.py
git commit -m "fix(runtime): unify task delivery state and fold pending completions into foreground turns"
```

---

### Task 3: Proactive Continuous-Mode Announcements With Sticky `user_speaking` & Live Barge-In

**Files:**
- Modify: `src/verbix/runtime/floor.py:19-75`
- Modify: `src/verbix/voice/pipeline.py:214-257, 291-415, 513-550`
- Modify: `src/verbix/runtime/runtime.py:103-125, 328-344`
- Modify: `src/verbix/runtime/workers.py:60-72, 84-96`
- Modify: `src/verbix/runtime/announcer.py:132-161`
- Test: `tests/test_conversational_voice_loop.py`, `tests/test_ledger_and_floor.py`

**Interfaces:**
- Consumes: `FloorArbiter` in `src/verbix/runtime/floor.py`.
- Produces:
  - `FloorArbiter.user_speaking: bool`, `FloorArbiter.set_user_speaking(active: bool) -> None`, and `FloorArbiter.can_announce() -> bool` in `src/verbix/runtime/floor.py`.
  - Sticky `user_speaking` in `VoicePipeline`: latched `True` on first detected speech (in `LISTENING` or on barge-in) and kept `True` across mid-utterance pauses, endpointing, and `finalize()` until `execute_voice_turn()` starts (or `finalize()` returns `""`).
  - Live barge-in during proactive announcements: interrupting a proactive announcement leaves the task `pending`, sets `user_speaking = True` so the announcer does not restart over the user, and folds the completion into the user's next foreground turn.

- [ ] **Step 1: Write the failing reproduction tests in `tests/test_conversational_voice_loop.py`**

```python
@pytest.mark.asyncio
async def test_continuous_mode_proactively_announces_completed_task_while_listening(
    tmp_path,
):
    from verbix.provider.models import Finish as ProviderFinish
    from verbix.provider.models import TextDelta as ProviderTextDelta
    from verbix.runtime.runtime import Runtime

    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        yield ProviderTextDelta(text="Nvidia closed at 228 dollars and 52 cents.")
        yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    sink = MockAudioSink(playback_delay=0.02)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=rt,
        stt=stt,
        player=player,
        capture=capture,
        aec=MagicMock(is_available=False),
    )

    await pipeline.start_continuous_session()
    assert rt.floor.state == FloorState.LISTENING

    # Background task finishes while in continuous LISTENING mode (user is silent)
    rt.ledger.register_task("task-1", "Get NVIDIA stock price")
    rt._on_task_finished("task-1", "$228.52")

    for _ in range(40):
        if rt.ledger.get("task-1").delivery == "heard":
            break
        await asyncio.sleep(0.02)

    assert rt.ledger.get("task-1").delivery == "heard"
    assert len(sink.played_chunks) > 0
    assert rt.floor.state == FloorState.LISTENING
    await pipeline.stop_continuous_session()
    await rt.aclose()


@pytest.mark.asyncio
async def test_barge_in_during_proactive_announcement_keeps_task_pending_and_folds_into_next_turn(
    tmp_path,
):
    from verbix.provider.models import Finish as ProviderFinish
    from verbix.provider.models import TextDelta as ProviderTextDelta
    from verbix.runtime.runtime import Runtime

    seen_fg_prompts = []
    announcer_calls = 0
    provider = MagicMock()

    async def fake_stream(messages, tools=None, **kw):
        nonlocal announcer_calls
        last_msg = messages[-1].content if messages else ""
        if "[Present these completed findings" in last_msg:
            announcer_calls += 1
            yield ProviderTextDelta(text="Long announcement sentence one. Long announcement sentence two.")
            yield ProviderFinish(reason="stop")
        else:
            seen_fg_prompts.append(last_msg)
            yield ProviderTextDelta(text="Got it, folding the result into your answer.")
            yield ProviderFinish(reason="stop")

    provider.stream = fake_stream
    rt = Runtime(workspace=tmp_path, provider=provider)
    sink = MockAudioSink(playback_delay=0.25)
    player = AudioPlayer(tts=MockTextToSpeech(), sink=sink)
    stt = FakeSTT()
    stt._texts = ["initial session", "Wait what happened?"]
    capture = FakeCapture()

    pipeline = VoicePipeline(
        runtime=rt,
        stt=stt,
        player=player,
        capture=capture,
        aec=MagicMock(is_available=False),
    )

    await pipeline.start_continuous_session()
    sess1 = stt.sessions[0]
    sess1.feed_barge_in = lambda chunk: b"BARGE" in chunk

    # 1. Task finishes while listening -> proactive announcement starts playing
    rt.ledger.register_task("task-1", "Check stock")
    rt._on_task_finished("task-1", "NVDA is $228.52")

    for _ in range(30):
        if pipeline.turn_output.is_playing:
            break
        await asyncio.sleep(0.01)
    assert pipeline.turn_output.is_playing is True
    assert announcer_calls == 1

    # 2. User speaks over the proactive announcement -> barge-in interrupts it!
    capture.on_chunk(b"BARGE_SPEECH")
    await asyncio.sleep(0.05)

    assert pipeline.turn_output.is_playing is False
    assert rt.floor.user_speaking is True
    # Task MUST remain pending (not heard), and announcer MUST NOT restart while user_speaking is True!
    assert rt.ledger.get("task-1").delivery == "pending"
    await asyncio.sleep(0.08)
    assert announcer_calls == 1

    # 3. User finishes their barge-in utterance -> completion is folded into the foreground turn
    sess2 = stt.sessions[1]
    sess2.on_endpoint()
    for _ in range(40):
        if rt.ledger.get("task-1").delivery == "heard":
            break
        await asyncio.sleep(0.02)

    assert rt.ledger.get("task-1").delivery == "heard"
    assert len(seen_fg_prompts) == 1
    assert "NVDA is $228.52" in seen_fg_prompts[0]
    assert "Wait what happened?" in seen_fg_prompts[0]
    assert announcer_calls == 1  # announcer never ran a second time!

    await pipeline.stop_continuous_session()
    await rt.aclose()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_conversational_voice_loop.py::test_continuous_mode_proactively_announces_completed_task_while_listening tests/test_conversational_voice_loop.py::test_barge_in_during_proactive_announcement_keeps_task_pending_and_folds_into_next_turn -v`
Expected: FAIL (`delivery` stays `"pending"` because `floor.state == FloorState.LISTENING` and `user_speaking` does not exist on `FloorArbiter`).

- [ ] **Step 3: Implement sticky `FloorArbiter.user_speaking`, continuous-mode floor restoration, and announcement barge-in preservation**

1. In `src/verbix/runtime/floor.py`:
   - Add `self.user_speaking: bool = False` in `FloorArbiter.__init__()`.
   - Add `set_user_speaking(self, active: bool) -> None`:
     - If `self.user_speaking != active`: sets `self.user_speaking = active`, and if `not active` and `self.on_change is not None`, calls `self.on_change(self._state, self._state)`.
   - Add `can_announce(self) -> bool`:
     - Returns `self._state == FloorState.IDLE or (self._state == FloorState.LISTENING and not self.user_speaking)`.
   - In `FloorArbiter.acquire(self, state)`:
     - Record `prev = self._state` before `self.set_state(target)`.
     - In `finally:`, restore `FloorState.LISTENING` if `prev == FloorState.LISTENING` and not `self.interrupt_requested`, else `FloorState.IDLE`.
2. In `src/verbix/voice/pipeline.py`:
   - In `_feed_chunk()` (when in continuous listening mode, not speaking): after `self._live_stt.feed(chunk)`, if `getattr(self._live_stt, "is_speech_detected", False)` is `True` and `hasattr(self.runtime, "floor")`, call `self.runtime.floor.set_user_speaking(True)`. (Sticky: never set `False` on mid-utterance silence dips in `_feed_chunk`).
   - In `_interrupt_active_speech()`: set `self.runtime.floor.set_user_speaking(True)` before `set_state(FloorState.LISTENING)` so the interrupted announcement does not re-trigger `_on_floor_state_changed` while the user is speaking their barge-in turn.
   - In `_run_continuous_loop()`:
     - If `not transcribed_text` after `finalize()`: call `self.runtime.floor.set_user_speaking(False)` before `_setup_turn_listening()`.
     - In `execute_voice_turn()`: once `self.runtime.floor.set_state(FloorState.SPEAKING)` is set, clear `self.runtime.floor.set_user_speaking(False)`.
3. In `src/verbix/runtime/runtime.py`, `src/verbix/runtime/workers.py`, and `src/verbix/runtime/announcer.py`:
   - Update `finish_runtime_task()`, `fail_runtime_task()`, `_on_floor_state_changed()`, and `drain_pending_announcements()` to check `runtime.floor.can_announce()`.
   - In `drain_pending_announcements()`: do **not** pop/clear items from `runtime._pending_announcements` permanently before `runtime.submit(prompt, runtime.wake_output, agent=runtime.announcer)` completes OK! If `runtime.submit` is interrupted/cancelled (`self.floor.interrupt_requested` or `CancelledError`), leave the tasks in `runtime._pending_announcements` with `delivery == "pending"` so they fold into the user's barge-in foreground turn!

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conversational_voice_loop.py tests/test_ledger_and_floor.py tests/test_idle_drained_announcements.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/floor.py src/verbix/voice/pipeline.py src/verbix/runtime/runtime.py src/verbix/runtime/workers.py src/verbix/runtime/announcer.py tests/test_conversational_voice_loop.py
git commit -m "feat(voice,runtime): enable proactive continuous-mode announcements with floor user_speaking and quiet window"
```

---

### Task 4: Concise Foreground Prompt Rules & `LitellmModel.query_stream()` Pre-Event Retry Fix

**Files:**
- Modify: `src/verbix/agent/config/voice.yaml:16-25`
- Modify: `src/verbix/runtime/foreground.py:64-79`
- Modify: `src/verbix/agent/models/litellm_model.py:198-266`
- Test: `tests/test_prompt_invariants.py`, `tests/test_stream_retry.py`

**Interfaces:**
- Consumes: `LitellmModel.query_stream` in `src/verbix/agent/models/litellm_model.py`, `voice.yaml` in `src/verbix/agent/config/voice.yaml`.
- Produces: Updated `# TOOLS` rules in `voice.yaml` / `DEFAULT_VERBIX_PROMPT` and pre-event retry on empty initial SSE chunks in `LitellmModel.query_stream()`.

- [ ] **Step 1: Write the failing tests in `tests/test_prompt_invariants.py` and `tests/test_stream_retry.py`**

```python
# In tests/test_prompt_invariants.py:
def test_foreground_prompt_covers_changing_facts_recent_exchange_and_completed_results():
    import yaml
    from pathlib import Path
    from verbix.runtime.runtime import DEFAULT_VERBIX_PROMPT

    voice_data = yaml.safe_load(Path("src/verbix/agent/config/voice.yaml").read_text(encoding="utf-8"))
    for sys_prompt in (voice_data["agent"]["system_template"], DEFAULT_VERBIX_PROMPT):
        lower = sys_prompt.lower()
        assert "facts that change" in lower or "current news" in lower
        assert "most recent exchange" in lower
        assert "incomplete fragment" in lower
        assert "newly completed" in lower


# In tests/test_stream_retry.py:
@pytest.mark.asyncio
async def test_midstream_fallback_after_empty_role_chunk_still_retries(monkeypatch):
    empty_role_chunk = {"choices": [{"delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
    midstream_err = litellm.exceptions.MidStreamFallbackError(
        message="Connection closed mid-handshake", model="test/model", llm_provider="openrouter"
    )

    calls = 0

    async def fake_acompletion(**kwargs):
        nonlocal calls
        calls += 1

        async def gen():
            yield empty_role_chunk
            if calls == 1:
                raise midstream_err
            yield _text_chunk("recovered after empty role chunk", "stop")

        return gen()

    monkeypatch.setattr("litellm.acompletion", fake_acompletion)
    model = LitellmModel(model_name="test/model")
    events = [e async for e in model.query_stream([{"role": "user", "content": "hi"}])]
    assert calls == 2
    assert any(isinstance(e, TurnComplete) and "recovered" in e.final_text for e in events)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_prompt_invariants.py::test_foreground_prompt_covers_changing_facts_recent_exchange_and_completed_results tests/test_stream_retry.py::test_midstream_fallback_after_empty_role_chunk_still_retries -v`
Expected: FAIL

- [ ] **Step 3: Update `voice.yaml`, `DEFAULT_VERBIX_PROMPT`, and `LitellmModel.query_stream()`**

1. In `src/verbix/agent/config/voice.yaml:16-24` and `src/verbix/runtime/foreground.py:64-79`:
   - Update `# TOOLS` / `# CAPABILITY ROUTING & DELEGATION` to include:
     - *"Answer general knowledge questions directly; for facts that change — current news, live prices, or current status — delegate to spawn to verify rather than answering from memory."*
     - *"For short follow-ups ('do that', 'try anyway', 'prove me wrong'), prefer the most recent exchange as the subject."*
     - *"Check active background tasks in the request context before calling spawn so running work is not duplicated. When an utterance is an incomplete fragment with no clear request, briefly ask what the user meant instead of spawning work."*
     - *"When newly completed background results appear in the request context, briefly report them before answering the user's latest input."*
2. In `src/verbix/agent/models/litellm_model.py:198-266`:
   - Remove `yielded_any = True` at line 199 (immediately on `async for chunk in response:`) and set `yielded_any = True` only right before `yield ThoughtDelta(...)`, `yield TextDelta(...)`, and `yield ToolCallDelta(...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_prompt_invariants.py tests/test_stream_retry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/agent/config/voice.yaml src/verbix/runtime/foreground.py src/verbix/agent/models/litellm_model.py tests/test_prompt_invariants.py tests/test_stream_retry.py
git commit -m "fix(agent,prompts): refine foreground conversational rules and retry stream on empty initial chunks"
```

---

### Task 5: Scripted Text-Mode Replay of the Live Session

**Files:**
- Modify: `tests/test_runtime_e2e.py`
- Test: `tests/test_runtime_e2e.py`

**Interfaces:**
- Consumes: `Runtime` (`submit`, `spawn`, `status`, `_build_grounded_prompt`, `_foreground_grounding`), `TaskLedger`.
- Produces: `test_live_session_replay_awareness_and_single_delivery` in `tests/test_runtime_e2e.py`.

- [ ] **Step 1: Add the scripted text-mode replay test in `tests/test_runtime_e2e.py`**

```python
@pytest.mark.asyncio
async def test_live_session_replay_awareness_and_single_delivery(tmp_path):
    """Replays the live session scenarios in text mode:
    1. Pending completion folded into 'What's the status?' turn is spoken once, never twice.
    2. Foreground sees 'screenshot' in worker capabilities and resolves 'Then prove me wrong'
       against the recent screen exchange rather than re-verifying the heard Hugging Face task.
    3. Running screen task appears in [Active background tasks] on fragment 'And', preventing
       a duplicate spawn.
    """
    captured_fg_calls = []
    announcer_outputs = []

    class RecordingWakeOutput:
        async def emit(self, event):
            if isinstance(event, TextDelta):
                announcer_outputs.append(event.text)
        def delivered_text(self):
            return "".join(announcer_outputs)

    provider = MagicMock()

    async def scripted_stream(messages, tools=None, **kw):
        sys_msg = messages[0].content if messages and messages[0].role == "system" else ""
        last_user = messages[-1].content if messages else ""
        captured_fg_calls.append({"system": sys_msg, "user": last_user, "messages": list(messages)})

        if "What's the status?" in last_user:
            # Must see the newly completed task-1 result in the turn context
            assert "task-1" in last_user and "12.9B" in last_user
            yield ProviderTextDelta(text="Nvidia agreed to buy Hugging Face for 12.9B.")
            yield ProviderFinish(reason="stop")
        elif "Can you take a look at my screen" in last_user:
            # System prompt must advertise screenshot capability for workers
            assert "screenshot" in sys_msg
            yield ProviderTextDelta(text="Sure, I can have a worker capture your screen.")
            yield ProviderFinish(reason="stop")
        elif "Then prove me wrong" in last_user:
            # Most recent exchange is about the screen; task-1 is already heard and not re-injected
            assert "Newly completed background tasks" not in last_user
            prev_assistant = [m.content for m in messages if m.role == "assistant"][-1]
            assert "screen" in prev_assistant.lower()
            yield ProviderToolCallDelta(
                index=0,
                call_id="call_screen_1",
                name="spawn",
                arguments_delta='{"objective": "Take a screenshot and describe the screen"}',
            )
            yield ProviderFinish(reason="tool_calls")
        elif messages and messages[-1].role == "tool":
            yield ProviderTextDelta(text="Checking your screen now.")
            yield ProviderFinish(reason="stop")
        elif last_user.endswith("\nAnd"):
            # Must see task-2 actively running in the request grounding and NOT spawn a duplicate
            assert "Active background tasks:" in last_user
            assert "Take a screenshot and describe the screen" in last_user
            yield ProviderTextDelta(text="Still checking your screen — did you want to add something?")
            yield ProviderFinish(reason="stop")
        else:
            yield ProviderTextDelta(text="ok")
            yield ProviderFinish(reason="stop")

    provider.stream = scripted_stream

    # Worker model stays running on task-2 so we can test the active-task fragment turn
    worker_provider = MagicMock()
    worker_gate = asyncio.Event()

    async def slow_worker_stream(messages, tools=None, **kw):
        await worker_gate.wait()
        yield ProviderTextDelta(text="Screen shows terminal.")
        yield ProviderFinish(reason="stop")

    worker_provider.stream = slow_worker_stream

    rt = Runtime(workspace=tmp_path, provider=provider, worker_model=worker_provider)
    rt.wake_output = RecordingWakeOutput()

    # Scenario 1: task-1 completes while user is speaking "What's the status?"
    rt.floor.set_state(FloorState.LISTENING)
    rt.floor.set_user_speaking(True)
    rt.ledger.register_task("task-1", "Verify if NVIDIA ever acquired Hugging Face")
    rt._on_task_finished("task-1", "Nvidia agreed to buy Hugging Face for $12.9B.")
    rt._task_counter = 1

    rt.floor.set_user_speaking(False)
    reply1 = await rt.submit("What's the status?")
    assert "12.9B" in reply1
    assert rt.ledger.get("task-1").delivery == "heard"
    await asyncio.sleep(0.05)
    # No second announcer speech!
    assert announcer_outputs == []

    # Scenario 2: User asks about screen, then says "Then prove me wrong"
    reply2 = await rt.submit("Can you take a look at my screen and tell me what you see")
    assert "screen" in reply2.lower()

    reply3 = await rt.submit("Then prove me wrong")
    assert "Checking your screen now." in reply3
    assert len(rt.list_active_tasks()) == 1
    assert rt.tasks["task-2"].description == "Take a screenshot and describe the screen"

    # Scenario 3: While task-2 is running, user says "And" -> sees active task, no duplicate spawn
    reply4 = await rt.submit("And")
    assert "Still checking your screen" in reply4
    assert len(rt.tasks) == 1  # only task-2 in rt.tasks; no task-3 spawned!

    await rt.cancel("all")
    await rt.aclose()
```

- [ ] **Step 2: Run the scripted replay test and full test suite**

Run: `set -o pipefail; uv run pytest tests/test_runtime_e2e.py::test_live_session_replay_awareness_and_single_delivery -v && uv run pytest -q && uv run ruff check src tests`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_runtime_e2e.py
git commit -m "test(runtime): add scripted replay of live voice session awareness and single-delivery invariants"
```
