# Foreground Conversational & Task Awareness Spec

**Date**: 2026-10-15  
**Status**: Approved Decisions Incorporated  
**Branch**: `reaudit-unification` (`69c6e5c`)

---

## 1. Root-Cause Diagnosis of the Live Session Transcript

| Observed Live Behavior | Transcript Timestamp | Layer | Root Cause (`file:line`) |
|---|---|---|---|
| **1. Same task result spoken twice** (and completions silent until user spoke `"Hello"` / `"And"` / `"God"`) | `20:00:06–20:00:46`, `20:03:48–20:04:07`, `20:09:41–20:09:46` | **Runtime Bug** | (a) In continuous mode, `VoicePipeline` keeps `floor.state = FloorState.LISTENING` (`pipeline.py:160, 280, 560`). Because `finish_runtime_task()` (`workers.py:60`) and `drain_pending_announcements()` (`announcer.py:134`) require `floor.is_idle` (`state == IDLE`), background completions never announce while listening—they only fire when a subsequent user turn exits `FloorArbiter.acquire()` (`floor.py:75`), which momentarily flashes `IDLE` before `execute_voice_turn()` resets `LISTENING`.<br/>(b) `finish_runtime_task()` (`workers.py:58-63`) immediately appends `[Task 'task-2' completed]: ...` to `foreground.messages` AND queues `("task-2", result)` in `_pending_announcements`. When the user asked `"What's the status?"` (`20:03:48`), the foreground read `[Task 'task-2' completed]` from `foreground.messages` and spoke the answer (`20:03:49`), while `("task-2", result)` remained `pending` and was spoken a second time (`20:04:07`) when the turn ended. |
| **2. Repeatedly claimed it couldn't see the screen and workers wouldn't help** | `20:07:17–20:08:07` | **Missing Context** | Neither `voice.yaml` (`lines 16-24`), `Runtime._build_grounded_prompt()` (`runtime.py:149-169`), nor `SpawnTool.description` (`tools/spawn.py:16`) tells the foreground model what tools background workers have (`screenshot`, `read` with image viewing, `web_search`, `web_fetch`, `write`, `edit`, `grep`, `glob`, `bash`). |
| **3. Fragment `"And"` spawned a duplicate screenshot task (`task-5`) while `task-4` was running/finishing** | `20:08:55–20:09:46` | **Missing Context + Runtime Bug** | (a) `Runtime._request_grounding()` (`runtime.py:171-194`) only injects `[Current local date and time: ...]`. It never shows active/running tasks from `self.tasks` / `self.ledger`.<br/>(b) Because `task-4` completed while `floor.state == LISTENING`, `finish_runtime_task()` silently injected `[Task 'task-4' completed]: ...` into `foreground.messages` without announcing it. When the user said `"And"` (`20:09:41`), the foreground spawned `task-5`, and `task-4`'s delayed announcement drained 3 seconds later (`20:09:46`). |
| **4. `"Then prove me wrong"` re-verified Nvidia/Hugging Face (`task-3`) instead of screen viewing** | `20:08:06–20:08:19` | **Missing Context + Weak Instruction** | Because the foreground believed screen capture was impossible (Missing Context #2), and `voice.yaml` lacked a rule preferring the most recent exchange for short follow-ups (*"prove me wrong"*, *"do that"*), the 8B model resolved `"prove me wrong"` back to `task-2`. |
| **5. Asserted from memory that a news event never happened, then reversed** | `20:02:36–20:03:49` | **Weak Instruction** | `voice.yaml:17` says *"Answer quick questions directly"*, without distinguishing stable general knowledge from facts that change (current news, live prices, current status). |
| **6. `MidStreamFallbackError` crashed turn at `20:08:37`** | `20:08:37–20:08:39` | **Runtime Bug** | `LitellmModel.query_stream()` (`litellm_model.py:199`) sets `yielded_any = True` on the first raw SSE chunk (an empty role delta on OpenRouter) before yielding any `AgentEvent`, blocking `_TRANSIENT_STREAM_ERRORS` retry (`litellm_model.py:272`). |

---

## 2. Specification of Fixes (With Approved Engineer Decisions)

### Fix 1: Derive Worker Capabilities Once Per Runtime From `build_worker_registry()`
- **Files**: `src/verbix/runtime/workers.py`, `src/verbix/runtime/foreground.py`, `src/verbix/runtime/runtime.py`, `src/verbix/tools/spawn.py`.
- **Design**:
  - Add `describe_worker_tools(workspace: Path | None = None) -> tuple[list[str], str]` in `src/verbix/runtime/workers.py` that inspects `build_worker_registry(workspace=workspace).get_definitions()` once and returns `(tool_names, formatted_summary)` (automatically including or omitting `bash` based on `probe_sandbox().available`).
  - Compute `runtime.worker_tool_names, runtime.worker_capabilities = describe_worker_tools(runtime.workspace)` once in `build_foreground_agent()` (`src/verbix/runtime/foreground.py`) and reuse those cached attributes in `Runtime._build_grounded_prompt()` and `SpawnTool.description`.
- **Verification**: Failing unit test asserting `runtime.system_prompt` and `SpawnTool(runtime).description` dynamically include all registered worker tools (including `screenshot`, `web_search`, `web_fetch`) and omit `bash` when `probe_sandbox().available` is `False`, computed once at runtime init.

### Fix 2: Inject Active & Newly Completed Tasks Into `_request_grounding()` and Unify Delivery State
- **Files**: `src/verbix/runtime/runtime.py`, `src/verbix/runtime/workers.py`, `src/verbix/runtime/announcer.py`, `src/verbix/runtime/ledger.py`.
- **Design**:
  1. **Single Delivery State (`TaskLedger.delivery`: `"pending" | "heard" | "interrupted"`)**:
     - Do **not** append `[Task '<id>' completed]: ...` to `runtime.foreground.messages` inside `finish_runtime_task()` / `fail_runtime_task()`. Instead, `finish_runtime_task()` / `fail_runtime_task()` records the result in `runtime.ledger` (`delivery = "pending"`) and appends `(task_id, result)` to `runtime._pending_announcements`.
  2. **Path A — Fold Into Foreground Turn (Marked `heard` Only After Turn Completes OK)**:
     - When a foreground turn runs (`Runtime.step()` / `Runtime.submit()` with `agent is self.foreground`), `_request_grounding()` inspects:
       - Active tasks (`self.list_active_tasks()`): formats `[Active background tasks: task-4 ('...': running for 12s)]`.
       - Unannounced completed/failed tasks (`entry.delivery == "pending"` in `self.ledger`): formats `[Newly completed background tasks (report briefly before answering): task-2 ('...'): <result>]` into the turn's grounding envelope.
     - **Only after the foreground turn completes OK** (`not was_interrupted` and `TurnComplete.is_error is False`), mark those folded task IDs `self.ledger.mark_heard(task_id)` and remove them from `self._pending_announcements`. If the turn is interrupted or fails, their delivery state remains `"pending"`.
  3. **Path B — `Runtime.status()` Marks Pending Results `heard`**:
     - When `Runtime.status(task_id=..., full=...)` formats a completed or failed task whose `ledger` entry has `delivery == "pending"`, it calls `self.ledger.mark_heard(t.id)` and removes `t.id` from `self._pending_announcements` so the announcer never re-speaks a result already inspected via `status()`.
  4. **Path C — Proactive Announcer (`drain_pending_announcements()`)**:
     - Filters `self._pending_announcements` to tasks whose `self.ledger.get(task_id).delivery == "pending"`.
     - Speaks the batch via `self.announcer`, and **only when the announcer turn completes OK (not interrupted)**, marks `self.ledger.mark_heard(task_id)` and appends the `[Task '<id>' completed]: <result>` user context + spoken `assistant` reply into `self.foreground.messages`.

### Fix 3: Proactive Continuous-Mode Announcements With Sticky `user_speaking` & Live Barge-In
- **Files**: `src/verbix/runtime/floor.py`, `src/verbix/voice/pipeline.py`, `src/verbix/runtime/runtime.py`, `src/verbix/runtime/workers.py`, `src/verbix/runtime/announcer.py`.
- **Design**:
  1. **Floor Publishes Sticky `user_speaking` (`src/verbix/runtime/floor.py`)**:
     - Add `user_speaking: bool = False` to `FloorArbiter`, updated via `FloorArbiter.set_user_speaking(active: bool) -> None` (notifying `on_change` when `user_speaking` transitions `True -> False`).
     - Add `FloorArbiter.can_announce() -> bool`: returns `True` when `self._state == FloorState.IDLE` **or** (`self._state == FloorState.LISTENING` and `not self.user_speaking`). Because `user_speaking` stays latched `True` from the first detected speech until the user's turn is submitted (or `finalize()` returns empty), mid-sentence pauses and STT finalization are automatically protected without a separate quiet-window timer.
     - In `FloorArbiter.acquire(state)`: capture `prev = self._state` before setting `target`; in `finally:`, if `prev == FloorState.LISTENING` and not `self.interrupt_requested`, restore `FloorState.LISTENING` instead of dropping to `FloorState.IDLE`.
  2. **Pipeline Latches `user_speaking` Until Turn Submission (`src/verbix/voice/pipeline.py`)**:
     - In `VoicePipeline._setup_turn_listening._feed_chunk()`, when `getattr(self._live_stt, "is_speech_detected", False)` becomes `True` during `LISTENING`, call `self.runtime.floor.set_user_speaking(True)`. Never set `user_speaking` back to `False` on mid-utterance silence dips!
     - In `_interrupt_active_speech()` (barge-in during playback or during a proactive announcement), immediately call `self.runtime.floor.set_user_speaking(True)` so the announcer cannot restart over the user while they are speaking their barge-in utterance.
     - In `_run_continuous_loop()`, clear `self.runtime.floor.set_user_speaking(False)` only when `finalize()` returns `""` (empty non-speech) or when `execute_voice_turn()` enters `FloorState.SPEAKING`.
  3. **Barge-In on Proactive Announcements**:
     - During a proactive announcement in continuous mode, `Runtime.submit(..., agent=self.announcer)` holds the floor (`FloorState.SPEAKING`) and sets `self._turn_handle` to the announcer turn task.
     - When user speech triggers `_interrupt_active_speech()`, `cancel_turn()` cancels the announcer turn task before `submit()` completes, so `drain_pending_announcements()` does **not** mark the task `"heard"` (leaving it `"pending"` in `TaskLedger` and `_pending_announcements`), `floor.user_speaking` stays `True` so the announcer does not restart over the user, and the still-`"pending"` completion is folded into the user's next foreground turn!

### Fix 4: Concise, General Foreground Prompt Rules (`src/verbix/agent/config/voice.yaml` & `src/verbix/runtime/foreground.py`)
- **Files**: `src/verbix/agent/config/voice.yaml`, `src/verbix/runtime/foreground.py`.
- **Design**:
  - Update `# TOOLS` in `voice.yaml` (and `DEFAULT_VERBIX_PROMPT`) with four concise, general rules:
    1. **Facts That Change**: *"Answer general knowledge questions directly; for facts that change — current news, live prices, or current status — delegate to spawn to verify rather than answering from memory."*
    2. **Short Follow-Ups**: *"For short follow-ups ('do that', 'try anyway', 'prove me wrong'), prefer the most recent exchange as the subject."*
    3. **Fragments & Active Tasks**: *"Check active background tasks in the request context before calling spawn so running work is not duplicated. When an utterance is an incomplete fragment with no clear request, briefly ask what the user meant instead of spawning work."*
    4. **Newly Completed Results**: *"When newly completed background results appear in the request context, briefly report them before answering the user's latest input."*

### Fix 5: `LitellmModel.query_stream()` Pre-Event Retry Fix (`src/verbix/agent/models/litellm_model.py`)
- **Files**: `src/verbix/agent/models/litellm_model.py`, `tests/test_stream_retry.py`.
- **Design**:
  - Set `yielded_any = True` in `LitellmModel.query_stream()` only when an actual `ThoughtDelta`, `TextDelta`, or `ToolCallDelta` is yielded to the caller, so empty initial SSE role chunks from OpenRouter do not block `_TRANSIENT_STREAM_ERRORS` retry.

### Fix 6: Scripted Text-Mode Replay of the Live Session (`tests/test_runtime_e2e.py`)
- **Files**: `tests/test_runtime_e2e.py`.
- **Design**:
  - Add an end-to-end scripted multi-turn test `test_live_session_replay_awareness_and_single_delivery` replaying the session's failure modes in text mode:
    1. Spawn `task-2` (`Verify if NVIDIA ever acquired Hugging Face`), let it complete while floor is busy, then submit `"What's the status?"` -> verify the completion is folded into the foreground turn, spoken **once**, marked `heard`, and **not** spoken a second time when the turn ends.
    2. Submit `"Can you take a look at my screen and tell me what you see"` followed by `"Then prove me wrong"` -> verify the foreground prompt/context exposes `screenshot` from `build_worker_registry()` and the follow-up turn sees the screen exchange as the most recent context rather than re-verifying the `heard` Hugging Face task.
    3. Spawn `task-4` (kept running) and submit `"And"` -> verify `_request_grounding()` includes `[Active background tasks: ...]` for `task-4` and no duplicate task is spawned.
