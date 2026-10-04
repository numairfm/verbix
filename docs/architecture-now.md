# Verbix — how it works now

Single source of truth for the running system. Written from the code in this tree
(`main`, base `2d1bd62` + the uncommitted changes described in §1). Everything else in
`docs/` is historical and lives in `docs/archive/` (§16).

**Evidence tags**

- **[R]** read directly from the cited `file:line`.
- **[X]** confirmed by running something — the test suite, or a scripted fake model
  driving the **real** `TaskRuntime` / `VoicePipeline` / `AgentSession`.
- **[I]** inferred; follows from the code but not observed.

Line numbers are from this tree and will drift. File names and function names will not.

**What I ran**

- Full suite: `.venv/bin/python -m pytest -q` → **90 passed**, 0 skipped, ~58 s. **[X]**
  Not hermetic: it needs `bwrap` (`/usr/bin/bwrap`, present), cached `faster-whisper`
  weights, and `scratch/test_tts.wav`. **[X]**
- `ruff check src tests` → clean. **[X]**
- `pyright src/verbix` → only unresolved-import errors for `pydantic`, `httpx2`,
  `numpy`, `faster_whisper`; no errors introduced by these changes. **[X]**
- Coverage percentages were **not** measured (no coverage tool installed); §12 reasons
  from the test sources. **[R]**

---

## 1. What changed in this pass, and why

The brief was: *ask and it answers; ask for work and it acknowledges, spawns, you keep
talking, and the result comes out of the same voice as if one assistant did everything.
Results are voiced one at a time, never over me, and a result is never silently lost.
Fix with structure, not prompt pleading.*

Every change below is a **structural** change — the guarantee lives in code, not in
instructions the model is asked to remember.

### 1.1 `announced` now means "the user heard it"

Before: any foreground turn that finished normally set `announced=True` on every pending
completion (`tasks.py`, old `_run_turn`). A user turn that never mentioned the task
silently swallowed the result, and the dedicated wake turn was often the very turn that
got cancelled.

Now `announced` is set in exactly one place — after the runtime has actually emitted the
announcement to the output sink (`_maybe_announce`, `runtime/tasks.py:401-436`). A user
turn that does not mention a finished task leaves it `announced=False`, and it is spoken
as soon as the runtime is idle. `test_user_turn_does_not_clear_pending_announcement` and
`test_scenario_unrelated_question_while_task_finishes` pin this. **[X]**

### 1.2 Completions enter history only at safe boundaries

Before, `_finish_task` injected a `role="user"` runtime notice into the main history the
moment a worker finished. If the foreground was between an assistant `tool_calls`
message and its tool results, that produced **invalid** history. I reproduced it in
phase 1 with a suspending tool. **[X]**

Now `_record_completion` (`runtime/tasks.py:372-387`) injects only when no turn is in
flight; otherwise the task goes on `_pending_completions` and is drained by
`_drain_pending_completions` (`runtime/tasks.py:389-398`) at the top of the next turn,
before the model call — the only point where history provably has no dangling tool call.
`test_completion_never_injected_mid_tool_call` reproduces the old race with a 120 ms
suspending tool and asserts the history stays valid. **[X]**

### 1.3 One voice: the runtime speaks all task framing

New file `runtime/voice.py` (54 lines) is the **only** place that authors text about
tasks:

| Function | Output |
|---|---|
| `spawn_acknowledgement(title, deduped=False)` | `"On it — I'll take care of <title>."` / `"I'm already on <title>."` |
| `completion_announcement(task)` (success) | `"I'm done — <worker's spoken summary>."` |
| `completion_announcement(task)` (failure) | `"I couldn't finish <title>."` |
| `completion_announcement(task)` (cancel) | `"I stopped working on that."` |
| `completion_announcement(task)` (no result) | `"That finished, but I don't have a result to report."` |

Titles are capped at 45 characters for speech (`voice.py:15-22`). The words "task" and
"worker" never appear in these strings.

Consequences:

- **Spawn ack** comes from `TaskSpawnTool.execute` (`runtime/orchestration.py:134`),
  still `is_terminal=True`, so a spawn is **one model call** — no second call to phrase
  the acknowledgement.
- **Completion announcement** is a bare `TurnComplete` on the `wake_output` sink
  (`runtime/tasks.py:424-427`) with **no model call at all**. The model is not asked to
  narrate; it cannot leak "worker 3 failed", mention the finish protocol, or forget to
  speak.
- **Worker failure summaries are first-person** (`runtime/tasks.py:863-881`):
  `"I stopped before finishing X."`, `"I hit an internal error while working on X."`.
  A cancel records `"I stopped working on X."` (`runtime/tasks.py:599`), which is what the
  model would say if asked — cancels are not voiced (§4.7).
- **A completion can still be silent**, but only when the *worker* deliberately said
  there is nothing to say: `silent` is ANDed with `success` at finish time
  (`runtime/orchestration.py:67`) and `_finish_task` re-forces `announced=False` for
  anything that is not a successful completion (`runtime/tasks.py:632-640`). Failures,
  cancels and blocks are therefore **always voiced**. **[X]**

### 1.4 Announcements are serialized and never overlap a turn

`_maybe_announce` (`runtime/tasks.py:401-436`) requires `is_idle` and a
`wake_output`, and the `_announcing` set guarantees one announcement in flight
(`runtime/tasks.py:412, 421`). It re-schedules itself with `loop.call_soon` in a
`finally`, so N completions queue up and each is voiced once, in turn, never over the
user. `test_multiple_completions_each_announced_once` asserts each is announced exactly
once and that **no model call** happens. **[X]**

### 1.5 Whole-file writes cannot silently clobber another worker

`ToolContext` gained `file_versions: dict[str, str]` (`tools/base.py:17-29`) — the
sha256 this session last observed for each absolute path. `file_read` records it
(`tools/filesystem.py:61`), `file_write`/`file_edit` record their own writes
(`filesystem.py:125, 199`), and both **refuse** to write over an existing file that this
session never read (`filesystem.py:110-116`) or that changed since
(`filesystem.py:117-122`, `170-181`). The guard is forwarded through
`_WorkerWriteGuard` into the inner `ToolContext` (`runtime/tasks.py:100`).
`test_file_write_refuses_blind_overwrite` and `test_file_write_refuses_stale_overwrite`
cover both refusals. **[X]**

### 1.6 STT finalize latency regression: root-caused and reverted

`git log -p` showed commit `3b0c7e9` changed `LiveSTTSession.finish` from transcribing
only the **untranscribed tail** to transcribing the **whole buffer** ("If speech was
detected in the tail, never drop it: transcribe full buffer"). The tail was already
covered by `last_transcribed_len` tracking, so the change made every finalize re-run
whisper over the entire recording — pure added latency, no correctness gain. **[R]**

Reverted: `finish` now transcribes `buf[last_transcribed_len:]` and appends it to
`latest_text` (`voice/stt.py:190-247`). The full-buffer pass survives only as the
fallback when nothing has been transcribed yet (`voice/stt.py:230`).
`test_stt_finalize_transcribes_tail_not_whole_buffer` monkeypatches
`faster_whisper.vad.get_speech_timestamps` and asserts only the 32 000 tail bytes are
handed to whisper. **[X]**

### 1.7 CLI surface

- Prompt is `you> ` in both text REPL (`cli.py:410`) and voice mode (`cli.py:338`).
- Assistant prefix is `verbix: ` (`cli.py:188`); headers are `verbix (model: …, workspace:
  …)` (`cli.py:403`) and `verbix voice mode (…)` (`cli.py:326`). "Verbix Receptionist"
  is gone.
- **Thoughts and tool diagnostics go to stderr** (`cli.py:203-210`), so stdout is a clean
  transcript of what was said. **[X]**

Scripted preview of the transcript (scripted fake provider, real runtime): **[X]**

```
you> hi
verbix: Hi there. How can I help?
you> what is the date?
verbix: On it — I'll take care of check the date.
verbix: I'm done — It is Sunday, October 4, 2026.
```

(stderr carried `[tool: task_spawn(...)]` and `[c1 -> ok: On it — …]`.)

---

## 2. Module map

~4 700 lines of source across 33 files, ~3 300 lines of tests across 16 files. **[R]**

| File | LOC | Owns |
|---|---|---|
| `core/models.py` | 72 | `CanonicalMessage`, `CanonicalToolCall`, `WorkerResult`, `TaskState` |
| `core/events.py` | 38 | 5 `AgentEvent` dataclasses: `TextDelta`, `ThoughtDelta`, `ToolCallStarted/Finished`, `TurnComplete` |
| `core/history.py` | 157 | append; runtime completion injection; aborted-turn repair; transient running-task render |
| `core/agent.py` | 299 | `AgentSession.step()` — the one inner loop; tool-call assembly; concatenated-JSON split; `CycleDetector` |
| `provider/models.py` | 52 | provider stream events, `ToolDefinition` |
| `provider/errors.py` | 26 | provider error hierarchy |
| `provider/client.py` | 259 | streaming OpenAI-compatible SSE over `httpx2`; stale-connection retry |
| `runtime/voice.py` | **54** | **all runtime-authored first-person task framing** |
| `runtime/orchestration.py` | 210 | `task_spawn`, `task_cancel`, `task_status`, `worker_finish` |
| `runtime/tasks.py` | 887 | `TaskRuntime`: registry, turn serialization, announcement, write locks, workers |
| `runtime/worktree.py` | 102 | git worktree per isolated task |
| `tools/base.py` | 85 | `Tool`, `ToolResult`, `ToolContext` (with `file_versions`), lenient JSON |
| `tools/filesystem.py` | 208 | `file_read`/`file_write`/`file_edit` + containment + stale-write guards |
| `tools/bash.py` / `tools/sandbox.py` | 79 / 118 | sandboxed `bash`; bubblewrap executor, fails closed |
| `voice/pipeline.py` | 164 | toggle state machine, barge-in, sentence streaming, `wake_output` ownership |
| `voice/player.py` | 79 | `AudioPlayer`: serialized playback, instant stop, earcon |
| `voice/{capture,stt,pocket_tts,sinks,base}.py` | 121/247/123/145/105 | `pw-record`, faster-whisper + live session, PocketTTS, sinks, protocols |
| `ipc/{protocol,server,client}.py` | 33/304/120 | JSONL over a Unix socket; 5 actions; event fan-out |
| `cli.py` | 445 | argparse, `.env` loader, 4 run modes, `CliPrinter` |

**Dependency direction.** Imports are acyclic: `cli → ipc/voice → runtime → core →
provider/tools`. `core` never imports `runtime` or an interface. **[R]**

The direction holds for imports but **not** for state ownership: the runtime stores a
`wake_output: TurnOutput | None` that the voice pipeline or IPC server assigns into it
(`runtime/tasks.py:158`; `voice/pipeline.py:90`; `ipc/server.py:98-99`), and the runtime
reaches into `main_session.history` directly. The "receptionist profile" also lives in
the runtime rather than a config layer. **[R]**

---

## 3. The two loops

### 3.1 Inner loop — one foreground turn

`AgentSession.step()` is one whole turn and may contain many model calls
(`core/agent.py:89-299`). **[R]**

```
while turn_count < max_turns (30):                     agent.py:117
    render [system] + history, with a transient         agent.py:122
      "[Running: …]" line prefixed to the LAST user msg
    stream one model completion                         agent.py:131-157
    if finish_reason == "length":                       agent.py:159-167
        with tool calls -> stop with "cut off"
        without        -> append " ... (reply cut off …)"
    if the model requested tool calls:                  agent.py:169-219
        assemble calls; split concatenated JSON
        append assistant{tool_calls}                    agent.py:215
        3 identical consecutive calls -> stop           agent.py:222-239
        for each call: execute, append tool result      agent.py:244-274
        if any result.is_terminal -> append assistant(observation), end turn   agent.py:264-281
        else loop again (model now sees tool results)   agent.py:284
    else:                                              agent.py:288-291
        append assistant(text), end turn
max_turns exhausted -> synthetic stop message           agent.py:296-299
```

Facts, all **[R]** unless noted:

- The loop is **model-driven**. There is no router and no deterministic fast path.
- `ToolResult.is_terminal` (`tools/base.py:14`) ends the turn **without** a second model
  call. Only `task_spawn` and `worker_finish` set it. This is what makes a spawn cost
  exactly one model call. **[X]** `test_scenario_ask_spawn_result_voiced` asserts
  `len(provider.calls) == 1`.
- `thought` is stored on the history message but never serialized back to the provider
  (`provider/client.py`). **[R]**
- The `[Running: …]` annotation is **ephemeral**: added to a copy at render time, never
  written to history (`core/history.py:128-157`). **[X]**
- Cycle detection catches **3 identical consecutive** `(name, args)` calls
  (`core/agent.py:44`, `224`). An A,B,A,B alternation is not caught. **[R]**
- `Usage` is parsed and discarded; truncation is handled but never surfaced as a metric.
  **[R]**

### 3.2 Outer loop — turn serialization, announcement, workers

- **One user turn at a time.** `start_turn` cancels the in-flight user turn and starts a
  new one (`runtime/tasks.py:254-274`). `is_idle` = no live turn **and** the user is not
  recording (`runtime/tasks.py:173-177`). **[X]**
- **There is no wake turn any more.** The old `kind="wake"` path, its chain counter and
  its "any turn that completes flags completions announced" side effect were removed
  (`runtime/tasks.py:299-367`). Announcements are runtime-authored emissions, not model
  turns. **[R]**
- **Workers start immediately** with `asyncio.create_task` (`runtime/tasks.py:584-589`).
  No queue, no cap beyond writer slots.
- **A worker finishes by calling `worker_finish`** (`runtime/orchestration.py:14-80`).
  `on_finish` calls `_finish_task` **synchronously** (`orchestration.py:76`), so the task
  flips to completed before the worker's `step()` returns. **[X]**
- **A worker that stops without `worker_finish` is re-prompted once** with the session
  capped at 3 turns; if it still does not finish, the task is marked **failed**. Success
  is never fabricated (`runtime/tasks.py:843-870`). **[X]**
- **Completion is recorded at a safe boundary and announced when idle**
  (`runtime/tasks.py:610-666`, `372-436`). **[X]**

### 3.3 Who decides what

| Decision | Decided by | Where |
|---|---|---|
| Answer vs. act (which tool) | **Model** | `core/agent.py:169` |
| Spawn a task or not | **Model** | `runtime/tasks.py:216-231`, `runtime/orchestration.py:83` |
| Task title / prompt | **Model** | `runtime/orchestration.py:99-113` |
| Task id, workspace, worktree isolation, network | **Runtime** construction flags; **not** model-selectable | `runtime/tasks.py:113-123`, `780`; schema exposes no such fields |
| Whether a spawn dedups | **Runtime**, key `turn:<scope>:<title>\|<prompt>` | `runtime/tasks.py:551, 571` |
| When a worker is "done" | **Model** calls `worker_finish`; runtime fallback = re-prompt then fail | `runtime/tasks.py:843-870` |
| Whether a completion is silent | **Model** sets `silent`; runtime ANDs it with `success` | `runtime/orchestration.py:64-67` |
| **The words used for ack and for announcements** | **Runtime** | `runtime/voice.py` |
| **When an announcement happens** | **Runtime** — idle + serialized, never model-scheduled | `runtime/tasks.py:401-436` |
| Whether a completion may enter history yet | **Runtime** — safe boundary check | `runtime/tasks.py:372-398` |
| History repair on abort | **Runtime** | `runtime/tasks.py:337-340`, `core/history.py:70-126` |

---

## 4. The announcement contract

This is the heart of the change. Written as invariants, all enforced in code.

1. **`announced == True` implies the words were emitted to the output sink.** The only
   assignment is inside `_maybe_announce` after `await output.emit(...)` returns
   (`runtime/tasks.py:426-427`). If the emit is cancelled, `announced` stays `False` and
   the announcement is retried on the next `_maybe_announce` tick. **[X]**
2. **A user turn never clears `announced`.** `_run_turn` drains pending *history notices*
   (`tasks.py:326`) but never touches `announced`. **[X]**
3. **Announcements never overlap anything.** Preconditions: `wake_output` set,
   `is_idle`, `_announcing` empty (`tasks.py:410-413`).
4. **Announcements happen one at a time.** `_announcing` holds the id while in flight and
   is cleared in a `finally` that re-schedules `_maybe_announce` (`tasks.py:421-436`).
5. **Announcements make no model call.** The model cannot forget, embellish, or leak
   internals. **[X]**
6. **Failures, cancels and blocks are never silent.** `_finish_task` sets
   `silent=True` only when `result.silent and status == "completed" and result.success`;
   everything else goes to `announced=False` (`tasks.py:632-640`). **[X]**
7. **Cancel is deliberately not announced.** `cancel_task` finishes the task *and* marks
   it `recorded=True, announced=True` (`tasks.py:605-606`), so the runtime never says a
   word about a cancellation the user just asked for. It is still recorded in history
   (so the model knows) and still broadcast as `task_cancelled` (`tasks.py:607`).
   **[X]** `test_scenario_cancel_is_deliberately_silent`. The `cancelled` branch of
   `completion_announcement` (`voice.py:47`) is therefore defensive, not the normal path.
8. **A completion that cannot be injected safely is never lost** — it waits in
   `_pending_completions` (`tasks.py:161, 382`) and is drained at the next turn's safe
   boundary (`tasks.py:389-398`). **[X]**
9. **The `wake_output` sink is idempotent about delivery bookkeeping**: the runtime calls
   `output.reset_delivered()` before emitting (`tasks.py:425`) so an announcement cannot
   inherit the previous turn's delivered-text list. **[R]**

```
user turn A ──┐                        ┌── announce task1 ──┐
              ├─ _drain_pending ─┐     │                    ├─ call_soon
user turn B ──┘   (safe boundary) │     └── announce task2 ──┘   (_maybe_announce)
                                    │
worker finishes ──► _record_completion ─┬─ turn in flight? ─► _pending_completions
                                        └─ idle? ─────────► inject now
                                                                  │
                                                          _maybe_announce (call_soon)
```

---

## 5. Capabilities, and exactly what each model sees

### 5.1 Tools

- **Foreground** — three orchestration tools only: `task_spawn`, `task_cancel`,
  `task_status`. No file, no bash (`runtime/tasks.py:199-206`). **[R]**
- **Worker** — `file_read`, `file_write`, `file_edit`, `bash`, `worker_finish`
  (`runtime/tasks.py:792-803`). The three write-capable tools are wrapped in the write
  guard; `file_read` is not (`runtime/tasks.py:63-105`). **[R]**

### 5.2 What the foreground model receives on every call

1. The system prompt — the caller's, or the built-in short receptionist prompt
   (`runtime/tasks.py:220-243`).
2. Schemas for the three orchestration tools; `task_spawn`/`task_status` embed the
   resolved workspace path (`runtime/orchestration.py:91-98, 186-191`).
3. **The entire history, untrimmed** — user, assistant, tool, and runtime notices. There
   is no cap or summarization anywhere (`core/agent.py:122`). **[R]**
4. One **transient** prefix on the last user message:
   `[Running: <id> '<title>' (<N> tool calls)]` (`core/history.py:128-157`). **[X]**
5. `reasoning: {effort}` — `"none"` for gemma models, else `"minimal"`, unless overridden
   (`runtime/tasks.py:236-239`). Workers are created without it and send no reasoning
   field (`runtime/tasks.py:817-824`). **[R]**

The runtime notice the model sees for a finished background job is `role="user"` with an
explicit attribution banner, and details capped at 600 chars:

```
[Verbix runtime notice — not said by the user]
[Background work finished: '<title>' (<id>) -> Succeeded: <summary>. Details: <≤600 chars>… (call task_status for full details)]
```

(`core/history.py:45-68`.) It exists so the model can *reason* about a result if the user
asks; it is **not** what gets spoken. **[R]**

### 5.3 What the worker model receives

- A worker system prompt naming the task and its directory, and spelling out the
  `worker_finish` contract (`runtime/tasks.py:805-815`).
- Exactly **one** user message: the spawn `prompt` (`runtime/tasks.py:836`). The worker
  gets a fresh `AgentSession` with its own history and **cannot see the chat**. **[X]**
  probe: worker roles were `['system','user']`.
- The five worker tool schemas.

`step_count` counts `ToolCallFinished` events, i.e. tool calls, not "steps"
(`runtime/tasks.py:842-844`). **[R]**

---

## 6. Life of a task

**Record.** `TaskState`: `id, title, status, step_count, recorded, announced, silent,
waiting_on, result, workspace` (`core/models.py:58-72`). Statuses
`pending|running|completed|failed|cancelled` (`core/models.py:8`). **[R]**

**Transitions** (plain assignments, no validator): **[R]**

```
spawn:            (new) -> running              runtime/tasks.py:577-581
worker_finish:    running -> completed           orchestration.py:76 -> tasks.py:610
no-finish:        running -> failed              runtime/tasks.py:863-870
exception:        running -> failed              runtime/tasks.py:872-881
cancel:           running -> cancelled           runtime/tasks.py:589-608
```

**Record vs. announce.** Two different things, deliberately decoupled:

- `recorded` = the runtime notice is in the foreground history. Written at a safe
  boundary (`tasks.py:372-398`). It is context for the model, not speech.
- `announced` = the first-person sentence was emitted to `wake_output`. Written in
  exactly one place (`tasks.py:427`).

**Silent.** Only a successful completion whose worker passed `silent=True`
(`orchestration.py:67`). Such a task is `recorded=True, announced=True` — deliberately
surfaced, never spoken (`tasks.py:634-638`). **[X]**

**Cancel.** `cancel_task` cancels the worker handle, writes a first-person result
(`"I stopped working on X."`), finishes the task, marks it **recorded and announced** —
so nothing is spoken, because the user already knows (`tasks.py:605-606`) — and broadcasts
`task_cancelled` (`tasks.py:607`). **[X]**

**Failure.** A no-finish worker and an exceptioning worker both become `failed` with a
first-person spoken summary; `details` keeps the technical text, reachable through
`task_status` (`tasks.py:863-881`). A worker's `CancelledError` is swallowed because the
record was already written by `cancel_task` (`tasks.py:872`). **[R]**

**Write locks.** A sticky per-resolved-workspace `asyncio.Lock` plus a global writer
semaphore (`runtime/tasks.py:668-724`). A worker takes it lazily at its **first
write-capable** call and holds it until the task ends; `file_read` never takes it. Waiters
consume a writer slot while waiting and are visible via `waiting_on`. Default
`max_concurrent_writers = 4` (`tasks.py:123`). With `isolate_worktrees`, each worker locks
its own worktree path so writers run in parallel. `_run_worker`'s `finally` releases
everything (`tasks.py:884-886`), so a crashed holder cannot wedge later writers. **[X]**
covered by `tests/test_step3_write_lock*.py`.

---

## 7. State: what exists, who changes it, what survives restart

| State | Lives in | Changed by | Persisted? |
|---|---|---|---|
| Foreground conversation | `ConversationHistory._messages` | `AgentSession.step`; `_record_completion` / `_drain_pending_completions`; `close_aborted_turn` on abort | **No** |
| Worker conversations | one history per worker, local to `_run_worker` | that worker's session | No |
| Task registry | `TaskRuntime.tasks` | spawn, `_finish_task`, `_run_worker`, `cancel_task`, `_maybe_announce` | No |
| Worker handles | `_worker_tasks` | spawn; popped by done-callback and by `_finish_task` (`tasks.py:642-643`) | No |
| Submission dedup index | `_submission_index` | spawn; cleared on finish (`tasks.py:571, 646-649`) | No |
| Write locks, holders, waiters, slots | `_write_locks`, `_write_lock_holders`, `_lock_waiters`, `_writer_slots` | `_write_guard`, `_release_write_lock_sync` | No |
| Turn handle + idle flags | `_turn_task`, `user_active`, `wake_output` | `start_turn`, pipeline, IPC server | No |
| **Queued completions** | `_pending_completions` (`tasks.py:161`) | `_record_completion`, `_drain_pending_completions` | No |
| **Announcement serialization** | `_announcing` (`tasks.py:164`) | `_maybe_announce` | No |
| **Stale-write detection** | `ToolContext.file_versions` (`tools/base.py:26`) | `file_read`, `file_write`, `file_edit` | No |
| Subscribers | `subscribers` | IPC server | No |
| Worktrees | `<workspace>/.verbix/worktrees/<task-id>` | `WorktreeManager.create_worktree` | **On disk, never cleaned** — `remove_worktree` has no src caller. **[R]** |
| Socket | `~/.local/share/verbix/verbix.sock` | server | transient |

**Nothing is persisted.** There is no pickle/json-dump/sqlite writer in `src/`. A restart
loses history, tasks and in-flight workers; no resume, no durable inbox. **[R]**

---

## 8. Sandbox, network, files, and what the model can influence

- **`bash` runs under bubblewrap** (`tools/sandbox.py:29-118`). If `bwrap` is missing it
  **raises** — no silent unsandboxed fallback (`sandbox.py:32-36`). It ro-binds `/usr`,
  `/lib`, `/lib64`, `/bin`, `/sbin` and selected `/etc`, provides `/proc`, `/dev`, a tmpfs
  `/tmp`, and **rw-binds the workspace**; `--unshare-net` is added unless `allow_network`
  (`sandbox.py:39-90`). The process group is killed on timeout or cancellation
  (`sandbox.py:114`). **[X]** `tests/test_tools.py` verifies workspace write allowed,
  `/root` write denied, network denied.
- **`allow_network` and `isolate_worktrees` are runtime construction flags**, not
  per-call tool args (`runtime/tasks.py:120-121, 776-780`); the spawn schema has no such
  fields (`orchestration.py:99-113`). The model cannot lift the network deny. **[R]**
- **File tools run in-process** and are not sandboxed; containment is
  `resolve()` then `relative_to(workspace)` (`tools/filesystem.py:9-19`). **[X]**
- **Stale-write guards** are described in §1.5. They are per-`AgentSession` (so per
  worker), which is exactly the scope where clobbering happens.
- **Worktree isolation** is opt-in; the main `.git` is rw-bound into the sandbox when
  present (`runtime/tasks.py:777-784`). Constant across tasks, not per-task. **[R]**
- **Model can influence**: tool choice, bash command text, file paths and content, task
  title and prompt, `success`/`silent`/`speech_summary`/`details`, and its own spoken
  answer text. **Model cannot influence**: workspace, network, worktree isolation, task
  id, dedup key, whether/when an announcement happens, and the announcement wording.

---

## 9. Interfaces

| Interface | Entry | Attaches to | Event consumer |
|---|---|---|---|
| CLI one-shot / REPL | `cli.py:398-445` | `runtime.submit(prompt, CliPrinter)` in-process | `CliPrinter` (`cli.py:187-236`) |
| `--voice` | `cli.py:311-360` | `TaskRuntime` + `VoicePipeline` in-process | `SpeechTurnOutput` |
| `--serve [--voice]` | `cli.py:277-309` | `TaskRuntime` (+ pipeline) behind `VerbixServer` | `IPCTurnOutput` / fan-out |
| `--toggle` | `cli.py:253-272` | `VerbixClient` → socket → pipeline | none |

`VerbixServer` exposes 5 actions: `ping`, `prompt`, `status`, `cancel`, `toggle`
(`ipc/server.py:208-304`). Runtime events (`task_spawned`, `task_finished`,
`worker_tool_call`, cancellations) fan out to all clients as `runtime_event`
(`ipc/server.py:151-169`). If no other sink claimed it, the runtime's `wake_output` is
`BroadcastWakeOutput` (`ipc/server.py:98-99`) — **this now carries completion
announcements**, so a daemon setup voices results through the socket to every connected
client. **[R]**

### Duplication

- **Runtime construction** is once (`build_runtime`, `cli.py:144-177`), but the **voice
  stack** is built twice (`cli.py:279-287`, `cli.py:311-320`). **[R]**
- **Event → output dispatch** is implemented three times as `isinstance` chains:
  `CliPrinter.emit` (`cli.py:203`), `IPCTurnOutput.emit` (`ipc/server.py:234`),
  `SpeechTurnOutput.emit` (`voice/pipeline.py:33`), plus `BroadcastWakeOutput.emit`
  (`ipc/server.py:56`). **[R]**
- **Status** has two paths: the `task_status` tool (`orchestration.py:177`) and the IPC
  `status` action (`ipc/server.py:270`), the latter bypassing model and history. **[R]**

### Gaps

- **No automated coverage of the CLI run-mode bodies** — `tests/test_cli.py` only covers
  `build_parser` and `load_env`. **[R]**
- **The IPC `prompt` path is untested** — `tests/test_ipc.py` covers ping/status/cancel/
  toggle only. **[R]**
- `VerbixClient.prompt` consumes lines until it sees a response with its own id
  (`ipc/client.py`), so fan-out events for other turns can interleave into its stream,
  and there is no reconnect/backoff. **[I]**

---

## 10. The voice path, and what interruption does

`cli.py:311-320` builds `FasterWhisperSTT`, `PocketTTSEngine`, `SubprocessAudioSink`,
`AudioPlayer`, `PTTCapture`, `VoicePipeline`. The pipeline assigns itself as
`runtime.wake_output` (`voice/pipeline.py:90`), which is what makes announcements
speakable. **[R]**

1. **Idle → recording.** `toggle()` sets `user_active=True`, starts capture and a
   `LiveSTTSession` that re-transcribes the growing buffer in the background, and warms
   the provider connection (`pipeline.py:119-135`, `voice/stt.py:148-174`). **[R]**
2. **Recording → processing.** Stops capture, plays an earcon, finalizes STT
   (tail-only, §1.6), and launches the turn in the background
   (`pipeline.py:136-160`). If the transcription is empty, it calls
   `runtime._maybe_announce()` so a pending result is not stranded by a silent utterance
   (`pipeline.py:153-155`). **[R]**
3. **Speak.** `SpeechTurnOutput` buffers `TextDelta`s and flushes complete sentences to
   `AudioPlayer.speak`, which streams PocketTTS chunks into a **single** sink process
   (`pipeline.py:33-61`, `voice/player.py:38-57`). At `TurnComplete` it flushes the
   remainder, or speaks `final_text` if nothing streamed (`pipeline.py:47-61`). **[X]**

**Barge-in.** Toggling during playback or a running turn calls `player.stop()`,
`runtime.cancel_turn()`, sets `user_active=True`, and starts recording
(`pipeline.py:106-117`). `_run_turn` catches `CancelledError`, calls
`close_aborted_turn(output.delivered_text())`, and re-raises (`tasks.py:337-340`);
`close_aborted_turn` fills in synthetic tool results for dangling calls or trims the
assistant message to what was actually spoken (`core/history.py:70-126`). So **history
equals what was heard**. **[X]**

Because announcements are separate `TurnComplete` emissions and `_maybe_announce`
requires `is_idle`, **barge-in during an announcement cancels the announce emit and the
task stays `announced=False`**, so it is re-voiced when idle. That is the intended
"never silently lost" behavior. **[R]**

---

## 11. Three flows, end to end

### A. Plain answer

1. CLI reads a line → `runtime.submit(prompt, printer)` (`cli.py:420`, `tasks.py:277`).
2. `start_turn` cancels any prior user turn; `_run_turn` bumps `_turn_scope`,
   drains pending completion notices, then calls
   `main_session.step(user_input=prompt, running_tasks=list_active_tasks)`
   (`tasks.py:254-334`).
3. `step` renders and streams; with no tool calls it appends the assistant message and
   yields `TurnComplete` (`core/agent.py:288-291`).
4. `CliPrinter` prints deltas with the `verbix: ` prefix, or `final_text` if nothing
   streamed (`cli.py:203-236`).
5. `finally` schedules `_maybe_announce` (`tasks.py:363-367`). **[R]**

### B. Ask for work, keep talking, hear the result

1. Model emits `task_spawn` → `TaskSpawnTool.execute` → `runtime.spawn_task`
   (`orchestration.py:113-138`).
2. `spawn_task` computes the turn-scoped dedup key, allocates `task-<hex6>`, records
   the task, broadcasts, and starts `_run_worker` (`tasks.py:538-589`).
3. The tool returns the runtime's ack string with `is_terminal=True`, so the foreground
   turn **ends on that one model call** and the ack is spoken (`orchestration.py:134-138`,
   `core/agent.py:264-281`). **[X]**
4. The worker runs; `worker_finish` calls `_finish_task` synchronously, which records the
   notice at a safe boundary (or queues it), broadcasts `task_finished`, and schedules
   `_maybe_announce` (`orchestration.py:76`, `tasks.py:650-666`). **[X]**
5. When the foreground is idle, `_maybe_announce` emits
   `TurnComplete(final_text=completion_announcement(task))` and sets `announced=True`
   (`tasks.py:424-427`). If there are two results, each is voiced in turn via the
   `call_soon` re-schedule. **[X]**

### C. Voice turn with interruption

See §10; the barge-in path cancels the turn and trims history to the delivered sentences.
**[X]**

---

## 12. Test coverage

### Covered, including the eight required scenarios

`tests/test_scenarios_now.py` drives the **real** `TaskRuntime` with a scripted fake
provider and asserts on the real outputs. All 12 pass. **[X]**

| Scenario | Test | Asserts |
|---|---|---|
| plain answer | `test_scenario_plain_answer:175` | one model call, text delivered, no spurious announcement |
| ask → spawn → result voiced | `test_scenario_ask_spawn_result_voiced:186` | exact ack string, **1 model call**, result spoken, no "worker"/"[Task" leakage |
| two stacked requests | `test_scenario_two_stacked_requests:218` | two acks, two announcements, each once, in order |
| quick failure | `test_scenario_quick_failure_is_voiced:247` | `"I couldn't finish …"`, and the worker's internal text is **not** spoken |
| barge-in | `test_scenario_barge_in_trims_history_to_spoken:267` | history == sentences actually spoken |
| cancel | `test_scenario_cancel_is_deliberately_silent:307` | cancelled is recorded and announced (i.e. **not** voiced) and never spoken |
| unrelated question while a task finishes | `test_scenario_unrelated_question_while_task_finishes:325` | the answer is spoken **and** the result is still announced afterwards |
| "thank you" after a result | `test_scenario_thank_you_after_result:357` | no duplicate announcement, no re-announcement of a finished task |
| mid-tool-call injection | `test_completion_never_injected_mid_tool_call:396` | history stays valid with a 120 ms suspending tool |
| blind overwrite refused | `test_file_write_refuses_blind_overwrite:440` | `file_write` over an unread existing file errors |
| stale overwrite refused | `test_file_write_refuses_stale_overwrite:451` | error after another session changes the file |
| STT tail-only finalize | `test_stt_finalize_transcribes_tail_not_whole_buffer:469` | only the tail bytes reach whisper |

Fakes used: `ScriptedProvider` (`tests/test_scenarios_now.py:43`), `DelayedProvider`
(`:60`), `HangingWorker` (`:81`); the harness `build` (`:149`) monkeypatches
`runtime._get_provider_for_model` so each worker gets its own scripted provider.

### Also covered

- Inner loop: conversational turn, tool-calling turn, cycle stop, history shape
  (`tests/test_agent.py`); aborted-turn repair and completion injection
  (`tests/test_history.py`).
- Turn serialization, error surfacing, spawn cancellation, ephemeral running annotation
  (`tests/test_runtime.py`), including
  `test_user_turn_does_not_clear_pending_announcement` and
  `test_multiple_completions_each_announced_once`.
- Write locks incl. reader bypass, cap, waiters, cancel-while-waiting, crash release,
  worktree keying (`tests/test_step3_write_lock*.py`).
- Worker handle reaping, no-steering absence (`tests/test_step1_loop_hardening.py`,
  `tests/test_regressions.py`).
- Sandbox filesystem/network denial (`tests/test_tools.py`). **[X]**
- Full suite: **90 passed, 0 skipped**. **[X]**

### Thin or absent

- **CLI run modes** — no automated coverage of `cli.py:253-445` bodies.
- **IPC `prompt`** — untested.
- **Real TTS / sinks / capture** — mocked (`voice/sinks.py`, `capture.py`,
  `pocket_tts.py`).
- **Live STT tests** depend on cached whisper weights and `scratch/test_tts.wav`; they
  would fail rather than skip elsewhere.
- `scripts/run_robustness_scenarios.py` is a live-model smoke test using substring
  matching, not a quality gate.
- No coverage tool is installed, so no percentages are claimed. **[R]**

---

## 13. Comparison with the intended experience

| Intended | As built | Evidence |
|---|---|---|
| Ask and it answers | Works, text and voice | §§3, 11 |
| Ask for work → acknowledges, spawns, you keep talking | Works; ack is runtime-authored, spawn is one model call | §1.3, `orchestration.py:134` **[X]** |
| Result comes out of the same voice, first person | **Structural now** — the runtime authors every task sentence; the model is not asked to narrate | §4, `runtime/voice.py` **[X]** |
| Never talk of tasks or workers unless asked | Enforced in code: those words are absent from `voice.py` | §1.3 **[X]** |
| Results voiced one at a time, never over me | `_announcing` + `is_idle` precondition | §4.3-4 **[X]** |
| A result is never silently lost | `announced` only set after emission; pending queue; failures never silent | §4 **[X]** |
| Completions never corrupt history | Safe-boundary injection only | §1.2 **[X]** |
| Workers never silently clobber each other | `file_versions` guards | §1.5 **[X]** |
| Barge-in / hands busy | Works; history trimmed to what was heard | §10 **[X]** |
| Multi-step projects referring back to the chat | **Partial.** The worker gets one self-contained prompt and cannot see the chat | `tasks.py:836` **[R]** |
| Instant host actions ("skip this song") | **Missing.** No host/MCP/MPRIS tool | `tasks.py:792-803` **[R]** |
| Screen / deictic ("this") | **Missing.** No capture path | **[R]** |
| Persistence across restart | **Missing** | §7 **[R]** |
| Per-task capability profiles | **Missing.** One network/worktree pair for the whole runtime | `tasks.py:120-121` **[R]** |
| Worktree lifecycle | Created, never removed or merged | `remove_worktree` has no src caller **[R]** |

---

## 14. Diagram

```
 keyboard / hotkey                      terminal / IPC client
        │ toggle                               │
        ▼                                      ▼
 ┌───────────────────┐              ┌───────────────────────────┐
 │ VoicePipeline     │              │ cli.py REPL   VerbixServer │
 │ capture→LiveSTT   │              │ (CliPrinter)  (JSONL/UDS)  │
 │ TextDelta→speech  │              └─────────────┬─────────────┘
 └────────┬─────▲────┘                            │
          │     │ speak / stop                     │ events
          │  ┌──┴──────────┐                       │
          │  │ AudioPlayer │◄── wake_output ───────┘
          │  │ TTS → 1 sink│        (announcements ride here)
          │  └───────▲──────┘
          ▼          │ pipeline.py:90
 ┌────────────────────────────────────────────────────────────────┐
 │ TaskRuntime                                                      │
 │  _turn_task · user_active · is_idle          (turn serialization)│
 │  tasks{} · _submission_index · _write_locks · _writer_slots      │
 │  _pending_completions  ── safe-boundary injection queue          │
 │  _announcing           ── one announcement in flight             │
 │  main_session AgentSession — tools: spawn/cancel/status          │
 │  _run_worker  AgentSession(own history, own file_versions)       │
 │      tools: file_read, file_write*, file_edit*, bash*, finish    │
 │  voice.py: every spoken task sentence is authored HERE           │
 │  _finish_task → _record_completion → _maybe_announce             │
 └──────────────┬───────────────────────────────┬─────────────────┘
                ▼                               ▼
        ProviderClient (SSE/httpx2)     bwrap sandbox / worktrees
```

---

## 15. Top risks and open questions

### Risks

1. **Context grows without bound.** Full history every call; 600-char details inlined
   (`core/agent.py:122`, `core/history.py:57-62`). No cap, no summarization. **[R]**
2. **Nothing survives a restart.** **[R]** §7.
3. **Worker finish is trust-based.** A worker calling `worker_finish(success=True, …)`
   decides its own spoken summary, which is used verbatim on success
   (`voice.py:53-56`). The re-prompt-then-fail fallback exists but nothing verifies the
   claimed outcome. **[R]**
4. **Unbounded workers / no queue.** Every accepted spawn is a live task; the only cap
   is writer slots. **[R]**
5. **The spawn dedup key is turn-scoped**, so the same request twice in one turn dedups
   but in two turns runs twice (`tasks.py:551`). Deliberate, but it means a
   re-phrased loop can spawn a lot. **[R]**
6. **Zero token/cost telemetry.** `Usage` is parsed and discarded. **[R]**
7. **Worktrees leak.** Never removed or merged. **[R]**
8. **The CLI and IPC prompt paths are untested** — the most user-visible surfaces. **[R]**
9. **Announcement wording is a fixed string set.** It is consistent and cannot leak, but
   it cannot convey nuance ("two things changed and one surprised me") without going back
   to a model turn, which is what caused the original bug. **[I]**

### Open questions

1. Should a **silent success** ever be forced loud — e.g. if the user asked about that
   task directly a turn later?
2. If a user turn and an announcement want the floor at the same instant, is
   "user first, announcement immediately after" right, or should the announcement wait a
   beat?
3. Should a **failure** announcement carry a short reason out loud ("the build was
   missing a dependency") rather than only "I couldn't finish X"? That needs
   runtime-authored selection from `details`, or a bounded model turn.
4. **Foreground failure handling**: retry/backoff and a persisted error record?
5. **Persistence and resume**: what is the contract for dangling tool calls and running
   workers across restart?
6. **History policy**: cap, summarize, or persist-and-trim? Should `details` be
   referenceable rather than inlined?
7. **Worktree lifecycle**: clean up on finish, or offer merge/review?
8. **Capability profiles**: should network / worktree isolation become per-task,
   user-configured profiles?

---

## 16. Archived docs

`docs/` contains this file and `docs/archive/` only.

| Archived file | Why it is no longer authoritative |
|---|---|
| `docs/archive/proposal.md` | Design intent (steering, confirmation gates, `TaskGroup`, worktree merge). Mechanisms that were never built or were deliberately removed. |
| `docs/archive/architecture-current.md` | A snapshot at `3b0c7e9`. Stale on nearly every mechanism: it documents `task_steer`/`_steer_queues` (removed), `SafeCommandChecker` (removed), `TaskState.progress` and three task events (removed), "no turn lock" (there is one), `role="system"` completion messages (now `role="user"` with a banner), no-finish workers recorded as success (now failed), barge-in as a no-op (now cancels and trims), `--direct` mode (removed), per-task `allow_network` (runtime flag). |
| `docs/archive/qwen-loop-adaptation.md` | Closest of the three to reality, and accurate on record-vs-announce and fail-closed workers — but it describes the **wake-turn** mechanism that has now been replaced by `_maybe_announce`, and its line references predate the current tree. |

When code and any archived doc disagree, the code wins and this file is the description.
