# Verbix — Current Architecture (as built)

Snapshot: commit `3b0c7e9`, 2026-10-03. Written from the code, not from `docs/proposal.md`.
No source code was changed to produce this document.

**Evidence tags** used throughout:

- **[R]** verified by reading the code at the cited `file:line`.
- **[X]** verified by executing something (pytest, coverage, or a throwaway in-memory script that touched no repo files).
- **[I]** inferred. It follows from the code but I did not observe it happen.

Line numbers refer to the files at this commit.

---

## 1. Module map

Size is lines of Python (`wc -l`). Source is 4,441 lines. Tests are 1,817. Scripts are 1,355, which is about 30% of the size of `src/`.

| Package / file | LOC | Owns | Depends on |
|---|---|---|---|
| `core/agent.py` | 270 | `AgentSession.step()`, the agent loop. Also `CycleDetector`, tool-call assembly, the Gemma concatenated-JSON splitter. | core, provider.models, tools.base |
| `core/history.py` | 143 | `ConversationHistory`, an in-memory message list. Also render with running-task annotation, completion injection, barge-in prune. | core.models |
| `core/models.py`, `events.py` | 62 + 59 | `CanonicalMessage`, `TaskState`, `WorkerResult`, and `AgentEvent` dataclasses. | none |
| `provider/client.py` | 269 | `ProviderClient`, a streaming OpenAI-compatible SSE client on `httpx2`. Includes warm, RTT probe, stale-connection retry. | core.models |
| `provider/{models,errors}.py` | 78 | Stream events and an error hierarchy. | none |
| `runtime/tasks.py` | 508 | `TaskRuntime`, the largest and most central file. Task registry, worker spawn/cancel/steer, receptionist construction, completion injection, announcement queue, subscriber broadcast. | core, provider, runtime.orchestration, tools |
| `runtime/orchestration.py` | 250 | The 5 tools: `task_spawn`, `task_cancel`, `task_steer`, `task_status`, `worker_finish`. | core.models, tools.base, runtime.tasks (type-only) |
| `runtime/worktree.py` | 102 | `WorktreeManager`, git worktrees under `.verbix/worktrees`. | none |
| `tools/{base,bash,filesystem,sandbox}.py` | 74 + 133 + 165 + 105 | `Tool` protocol, lenient JSON parsing, bash via bubblewrap, 3 file tools, `SafeCommandChecker`. | none |
| `voice/pipeline.py` | 205 | `VoicePipeline`: toggle state machine, live STT wiring, turn execution, confirmation futures. | runtime, voice.* |
| `voice/player.py` | 180 | `AudioPlayer`: serialized playback, interrupt, announcements. | runtime (mutates it) |
| `voice/{stt,capture,pocket_tts,sinks,base}.py` | 232 + 121 + 123 + 123 + 100 | faster-whisper plus Silero VAD, `pw-record` capture, PocketTTS thread, `pw-play` subprocess sink, Protocols and mocks. | none |
| `ipc/{server,client,protocol}.py` | 285 + 132 + 34 | JSONL over a Unix socket. 10 actions. | core.events; runtime and voice (type-only) |
| `cli.py` | 516 | argparse, `load_env`, 5 run modes, renderers. | everything |

Class count in `src/` is 66: core 16, provider 14, voice 14, tools 10, runtime 7, ipc 5 **[X]** (`grep '^class '`).

### Dependency direction

```
cli ──► ipc.server ──► runtime.tasks ──► core.agent ──► provider.client
 │            │              │  ▲            │
 │            ▼              ▼  │            └──► tools.base
 └──► voice.pipeline ──► voice.player ──┘ (player mutates runtime; runtime holds player: Any)
```

- The intended layering (core ← provider ← runtime ← interfaces) holds for imports. Runtime is not imported by core or provider.
- It does **not** hold for state. `runtime.player: Any` (`tasks.py:78`) is set by `AudioPlayer.__init__` (`player.py:31-32`) and again by `VoicePipeline.__init__` (`pipeline.py:42-43`). `AudioPlayer` writes `TaskState.announced` (`player.py:140`) and drains `runtime.announcement_q` (`player.py:152`). The `Any` type is what makes the Protocol-style decoupling work in typing but not in design. **[R]**

---

## 2. The core loop in plain language

`AgentSession.step(user_input, running_tasks)` at `core/agent.py:85-270`. One `step()` is one whole turn, possibly many LLM calls. It is an async generator that yields events and also mutates history as it goes.

```
append user message                               agent.py:95-96
loop up to max_turns (30):                        agent.py:113
    prompt = [system] + history, with the         agent.py:117-122
             running-task line prefixed to the
             LAST user message (transient)
    stream one provider completion                agent.py:129
        text/thought deltas  → yielded + buffered
        tool-call deltas     → buffered by index   agent.py:140-149
    if any tool calls:
        build calls (split concatenated JSON)      agent.py:152-192
        append assistant{tool_calls} to history    agent.py:198-202
        3 identical consecutive calls → stop       agent.py:204-214
        for each call: parse args leniently, execute, append tool msg   agent.py:216-249
        if any tool result was is_terminal:
            append assistant(observation); end turn (no 2nd LLM call)   agent.py:251-254
        else loop again
    else:
        append assistant(text); end turn          agent.py:259-264
max_turns exhausted → synthetic TurnComplete      agent.py:266-270
```

### Who decides what

| Decision | Decided by | Where | Notes |
|---|---|---|---|
| Act (tool) or reply (text) | **Model** | `agent.py:152` | The runtime has no router and no deterministic fast-path. |
| When to spawn a task | **Model** | prompt `tasks.py:133-134`, tool `orchestration.py:77-136` | Only prompt prose and a tool description. |
| Which workspace the task uses | **Runtime** | `spawn_task` `tasks.py:236`; the tool does not expose `workspace` (`orchestration.py:126-130`) | Good. Covered by a test (`test_runtime.py:359-361`). |
| Whether a worker is isolated in a worktree | **Model** | `isolate_worktree` arg `orchestration.py:106-118` | Default off, never enforced. |
| Whether bash gets network | **Model** | `allow_network` arg `bash.py:80-84,103` | The model can lift the sandbox's network deny with no gate. |
| What ends a receptionist turn | **Mixed** | model stops calling tools (`agent.py:259`), or runtime via `is_terminal` (`237-254`), the cycle detector (`204-214`), or `max_turns` (`113,266`) | Only `task_spawn` is terminal on the receptionist side. |
| What ends a worker | **Model, with a runtime fallback** | the model should call `worker_finish` (prompt `tasks.py:406-410`); if it just stops, the runtime fabricates a result with **`success=True`** (`tasks.py:470-488`) | A failed worker that explains the failure in prose is recorded as a success. **[R]** |
| Spoken completion text | **Model-authored, runtime-delivered** | `speech_summary` is sliced to 180 chars (`orchestration.py:55-58`), spoken verbatim (`player.py:134`) | No markdown strip. A slice can cut mid-word. |
| What the receptionist is told about a finished task | **Runtime** | `history.py:53-65` formats `[Task completed: …]`. Zero LLM calls. | Includes the full `details` string with no cap. |
| Whether the model sees task state | **Runtime** | the `[Running: …]` annotation `history.py:116-143` | Running tasks only. Finished tasks enter history once. |

---

## 3. Three flows, end to end

### A. Plain text turn (default CLI REPL)

1. `cli.py:478-493`: read a line, then `run_stream(runtime.step_receptionist(prompt))`.
2. `TaskRuntime.step_receptionist` (`tasks.py:156-176`) calls `on_main_turn_start` (drains buffered completions, sets `is_main_turn_active=True`). It then iterates `main_session.step(..., running_tasks=self.list_active_tasks)`.
3. `AgentSession.step` appends the user message (`agent.py:95`), renders `[system] + history` (`117-122`), and calls `provider.stream`.
4. `ProviderClient.stream` (`client.py:110-134`) wraps `_stream_once` (`136-269`). That POSTs `/chat/completions` with `stream:true`, `reasoning:{effort}` and `tools`. It parses SSE into `TextDelta`/`ThoughtDelta`/`ToolCallDelta`/`Finish`/`Usage`.
5. No tool calls, so the assistant message is appended and `TurnComplete` is yielded (`agent.py:259-263`).
6. `finally: on_main_turn_end()` (`tasks.py:175-176`) clears the active flag and drains pending completions.
7. `cli.py:153-197` renders deltas to stdout. `print_announcements` (`cli.py:453-458`) prints any queued task completions.

### B. "Check the repo": spawn → worker → completion → history

1. Turn as in A, but the model emits a `task_spawn` tool call (`title`, `prompt`, optional `isolate_worktree`).
2. `agent.py:152-192` builds the canonical call. `198-202` appends the assistant message with `tool_calls`. `221` runs `parse_lenient_json`. `234` calls `TaskSpawnTool.execute` (`orchestration.py:115-136`).
3. `TaskRuntime.spawn_task` (`tasks.py:226-270`):
   - allocates `task-N` (`_task_counter`, resets every process)
   - inserts `TaskState(status="running")` and a steer queue
   - broadcasts `task_spawned`
   - if a player is attached, fires `asyncio.create_task(player.speak_foreground_sentences(["Starting task: …"]))` (`251-257`). That task is fire-and-forget: no reference is kept and exceptions are never retrieved. **[R]**
   - launches `asyncio.create_task(_run_worker(...))` (`267`). This is bare `create_task` stored in a dict, not a `TaskGroup`.
4. The tool returns `is_terminal=True` with the ack text. `agent.py:247-254` appends the tool message and an assistant message with the ack. It yields `TurnComplete`, with no second LLM call and **no `TextDelta`**. **[R]**
   - CLI: the ack appears only as the `[call -> ok: …]` preview, because `run_stream` prints `TurnComplete.final_text` nowhere (`cli.py:185-189`).
   - Voice: spoken via the `spawn_task` side channel, because `execute_voice_turn` only speaks `TextDelta` text (`pipeline.py:173-195`).
5. Worker (`tasks.py:367-508`): optional worktree (`380-389`). Tools are file read/write/edit, `bash`, `worker_finish` (`394-400`). It runs a fresh `AgentSession` with its own `ConversationHistory` and only the model-written `prompt` as the user message. It has no access to the chat. **[R]**
6. The worker calls `worker_finish`. `WorkerFinishTool.execute` (`orchestration.py:51-74`) records the result and calls `on_finish`, which calls `_finish_task` (`tasks.py:326-365`):
   - sets status/result
   - `announcement_q.put_nowait`
   - if the main session is idle, injects into main history **now**. Otherwise it appends to `_pending_completions` (`347-354`).
   - broadcasts `task_finished`
7. Next user turn: `on_main_turn_start` drains (`tasks.py:309`). The model sees a `role="system"` message `[Task completed: 'x' (task-1) -> Succeeded: … Details: …]` (`history.py:61-65`) and answers from it without `task_status`.

Failure branches **[R]**:

- the worker stops without finishing → fabricated `success=True` (`tasks.py:470-488`)
- max turns → `failed` (`490-496`)
- any exception → `failed` (`501-508`)
- cancel → `_finish_task(status="cancelled")` immediately (`272-289`)

### C. Voice turn with barge-in

1. First toggle, idle. `VoicePipeline.toggle` (`pipeline.py:61-87`) → `start_recording` (`89-104`). It creates a `LiveSTTSession`, wires `capture.on_chunk`, fires `provider.warm()` in the background, and starts `pw-record` (`capture.py:38-92`).
2. While recording, `LiveSTTSession` re-transcribes the **entire** buffer every time ≥0.3 s of new audio exists (`stt.py:149-174`). Cost is quadratic in utterance length. **[R]**
3. Second toggle. `stop_recording_and_process` (`pipeline.py:106-126`):
   - stop capture
   - fire the cue task
   - `live.finish` (Silero VAD check on the un-transcribed tail, otherwise re-transcribe the full buffer, `stt.py:176-232`)
   - `asyncio.create_task(execute_voice_turn(text))` and return `"processing"` immediately
4. `execute_voice_turn` (`pipeline.py:164-205`) streams `step_receptionist`. On a sentence boundary it `await`s `player.speak_foreground_sentences(complete)` **inline**. This pauses consuming the LLM stream during playback. The remainder is spoken at `TurnComplete`. Then `drain_announcements`.
5. `AudioPlayer._synthesize_and_play` (`player.py:163-180`) iterates PocketTTS chunks and calls `sink.play(chunk)` per chunk. `SubprocessAudioSink.play` spawns a **new `pw-play` process per call** (`sinks.py:78-123`). **[R]** [I] audible gaps are likely but unmeasured.
6. Barge-in, toggle during playback (`pipeline.py:68-79`): `player.interrupt()` sets `_is_interrupted`, kills the sink subprocess, and tries to cancel `_active_play_task`. That attribute is **never assigned** (`player.py:34,46,81-82`), so the cancel is a no-op. **[R]** The pipeline then tries to prune the last history message and starts recording.

**What barge-in actually does [X]** (throwaway script, mock LLM emitting 7 sentences, toggle during sentence 1):

```
toggle -> interrupted_and_recording
spoken before barge-in: ['One.']
spoken after  barge-in: ['Two.Three.', 'Four.Five.', 'Six.Seven.']
history last: assistant 'One. Two. Three. Four. Five. Six. Seven.'
```

- Barge-in stops only the current sentence. The turn task is never cancelled. `toggle` ignores `_is_active_turn`, which is write-only (`pipeline.py:45,166,203`).
- The next `speak_foreground_sentences` resets `_is_interrupted = False` (`player.py:98`), so the rest of the reply is spoken while the mic is recording.
- History keeps the full unplayed text. The prune did nothing because the last message was still the user message, so `ValueError` was swallowed (`pipeline.py:73-77`). The parity rule ("history = what was spoken") is violated.
- Sentences glue (`'Two.Three.'`). `current_buffer = sentences[-1]` drops the trailing space (`pipeline.py:185`), so a next delta without a leading space merges. **[X]** with synthetic deltas. [I] real providers will hit it intermittently.

The 1.2 ms / 1.5 ms barge-in figure from earlier is the software time from `toggle()` to `sink.stop()`. It is not acoustic latency, and as shown above it does not cover cancelling the turn.

---

## 4. State, ownership, concurrency

### What exists

| State | Lives in | Mutated by | Persisted? |
|---|---|---|---|
| Main conversation | `ConversationHistory._messages` (`history.py:18`) | **Three writers in three modules**: `AgentSession.step` (`agent.py:95,198,247,252,262`), `TaskRuntime._finish_task`/`drain` (`tasks.py:324,350`), `VoicePipeline.toggle` via `prune` (`pipeline.py:73`) | No |
| Worker conversations | one `ConversationHistory` per worker, local to `_run_worker` | that worker's `AgentSession` | No |
| Task registry | `TaskRuntime.tasks` dict (`tasks.py:59`) | spawn, `_finish_task`, `_run_worker` (`step_count`), **`AudioPlayer`** (`announced`, `player.py:140`) | No |
| Worker handles, steer queues | `_worker_tasks`, `_steer_queues` (`tasks.py:63-64`) | spawn only. Never removed. **[R]** | No |
| Pending completions, `is_main_turn_active` | `tasks.py:74-75` | `on_main_turn_*`, `_finish_task` | No |
| Announcement queue | `announcement_q` (`tasks.py:67`) | produced by `_finish_task` | No |
| Subscribers | `subscribers: list[Queue]` (`tasks.py:70`) | IPC server only | No |
| Player flags | `_is_playing`, `_is_interrupted`, `_played_text` (`player.py:34-41`) | player | No |
| Live STT buffer, confirmation future | pipeline | pipeline | No |
| Git worktrees | `<workspace>/.verbix/worktrees/` | `WorktreeManager` | **Yes, on disk, never cleaned** |
| Unix socket | `~/.local/share/verbix/verbix.sock` | server | Transient file |

Nothing is persisted. A restart loses history, tasks, and in-flight workers. The proposal deferred this (`proposal.md` §7.1), so this is by design, but note that `task-N` ids restart too (see risk 8).

### Concurrency model

- Single process, single asyncio loop. Threads: PocketTTS generation (`pocket_tts.py:95-111`), `asyncio.to_thread` for Whisper (`stt.py:111`), stdin reads in the executor (`cli.py:342,352,369`).
- Locks: `AudioPlayer._playback_lock` (`player.py:40`) is the only one that matters. `PocketTTSEngine._lock` guards loading only. `FasterWhisperSTT._lock` is declared and never used (`stt.py:31`). **[R]**
- **There is no turn lock.** Nothing serializes `step_receptionist`. Voice toggles (`pipeline.py:125`), IPC `prompt` (`server.py:175`), and the REPL can all start turns concurrently. `is_main_turn_active` is a bool, not a counter (`tasks.py:74`). **[R]** [I] two overlapping turns interleave messages in one history and the first to finish clears the flag while the other is mid-tool-call, which allows a completion to be injected between an assistant `tool_calls` message and its tool results.
- Completion vs turn ordering is race-free **within one turn**, because `_finish_task` is synchronous (no `await` between the check and the append) and the loop is single-threaded. The earlier bug was fixed by draining at `on_main_turn_start` (`tasks.py:309`).
- Cancellation: `cancel_task` cancels the worker task and finishes the task record (`tasks.py:272-289`). `execute_sandboxed` kills the whole process group on `CancelledError`/timeout (`sandbox.py:97-105`).

### Exactly what the receptionist model sees each LLM call

1. **System prompt** (`tasks.py:129-138`): about 8 sentences. It lists the four tools, says "spawn immediately", and says completions appear in history so do not call `task_status`.
2. **Tool schemas** (4). Descriptions embed the workspace path (`orchestration.py:88,228`). That is how "check the repo" is meant to resolve without a clarifying question.
3. **All history, in order**: user, assistant, tool, and runtime `system` completion messages. `thought` is stored but never sent (`client.py:86-107`). **There is no truncation, summarization, or cap anywhere.** **[R]**
4. **Transient**: `[Running: task-1 'title' (step N)]` lines prefixed to the *last user message* only (`history.py:116-143`). Never persisted.
5. `reasoning:{effort}` and `max_tokens=2048` (default).

Measured earlier: about 480 prompt tokens on an empty conversation with Flash-Lite. This is from my previous pass, not re-run here.

Not in the prompt: any conversation context in workers, any worker progress (the `TaskState.progress` list is never written), any workspace file listing.

`step_count` increments once per outer worker iteration (`tasks.py:442`), which is once per `session.step()`. A worker normally runs its whole tool loop inside a single `step()`. So the "(step N)" in the annotation is nearly always 1, regardless of how much work was done. (The scenario run showed models replying "still on step 1".) **[R]**

---

## 5. How CLI, IPC and voice attach

| Interface | Entry | Attaches to | Event consumer |
|---|---|---|---|
| CLI REPL / one-shot | `cli.py:444-501` | `runtime.step_receptionist` in-process | `run_stream` (`cli.py:153-197`) |
| `--direct` | `cli.py:389-442` | a separate `AgentSession` with file/bash tools, no runtime, no receptionist | `run_stream` |
| `--voice` | `cli.py:303-387` | `TaskRuntime` + `VoicePipeline` in-process | `execute_voice_turn` |
| `--serve [--voice]` | `cli.py:261-301` | `TaskRuntime` (+ pipeline) behind `VerbixServer` | `server._dispatch_request` |
| `--toggle/--confirm/--deny` | `cli.py:205-235` | `VerbixClient` → socket → pipeline | none |

### Duplicated

- **Runtime construction** (`TaskRuntime(...)` + `create_receptionist()`) appears 3 times (`cli.py:265-271`, `304-310`, `445-451`). The voice stack (STT + TTS + sink + player + capture + pipeline) is built twice (`273-283`, `311-318`).
- **`AgentEvent` → output dispatch** is written three times as `isinstance` chains: CLI (`cli.py:159-189`), IPC server (`server.py:181-203`), voice pipeline (`pipeline.py:173-195`).
- **Task event vocabulary**: typed `TaskSpawnedEvent`/`TaskProgressEvent`/`TaskFinishedEvent` exist in `core/events.py` and are exported but **not used anywhere in `src/`**. The runtime instead broadcasts untyped `dict[str, Any]` (`tasks.py:91-98`). **[R]**
- **Status**: three paths. The `task_status` tool, the IPC `status` action (`server.py:220-224`, which bypasses the model and history), and the running annotation.
- **Spoken-title cap**: copied in `tasks.py:252` and `orchestration.py:131`.
- **Announcement consumers**: CLI prints (`cli.py:453`), `AudioPlayer` speaks (`player.py:146`), IPC server never drains. In `--serve` without voice the queue grows unbounded. [I]

### Interface gaps **[R]**

- **Completions are never spoken proactively.** `drain_announcements` has exactly one caller: the end of `execute_voice_turn` (`pipeline.py:197`). If a task finishes while the user is idle, nothing speaks until the next completed voice turn. The proposal's "Budget B: ~130 ms finish→speech" is not implemented.
- If a drain is interrupted, the popped item is dropped (`player.py:155-160`). It is never re-queued. `announced` stays false and nothing retries.
- IPC `prompt` blocks that connection's read loop for the entire turn (`server.py:142,175`). A client must open a second connection to cancel or toggle during a turn. [I]
- IPC catches `(RuntimeError, OSError, ValueError)` (`server.py:215`). `ProviderError` derives from `Exception` (`provider/errors.py:4`). A 400/429 during an IPC prompt escapes the handler and the client sees a dropped socket.
- The CLI REPL and the voice loop do not catch `ProviderError` either (`cli.py:495-499`, `383-384`). One 429 or 400 ends the REPL process. In voice, the exception is swallowed inside a fire-and-forget task (`pipeline.py:125`) and the user hears silence.

---

## 6. Per-model quirks the provider layer works around

| Quirk | Handling | Where | Layering note |
|---|---|---|---|
| `reasoning:{effort}` is OpenRouter's schema, sent on every request | always sent when set | `client.py:156-157` | Not valid for a generic OpenAI endpoint. |
| Gemma rejects `minimal`; Flash-Lite and 2.5 Pro reject `none` | `"none" if "gemma" in model.lower() else "minimal"`, with an override | `tasks.py:141-143` | Substring match in the runtime. Any other model that rejects `minimal` will 400. |
| Reasoning arrives as `reasoning_content` **or** `reasoning` | both read | `client.py:227` | |
| `reasoning_tokens` location varies | two shapes read | `client.py:258-263` | |
| Gemma emits two tool calls as concatenated JSON in one call | split into N calls `{id}_{i}` | `agent.py:156-192` | A provider quirk fixed inside the core loop. |
| Weak models wrap args in fences or prose | `parse_lenient_json`; failure becomes `{"_raw": …}` and the tool errors | `tools/base.py:37-74` | |
| Stale keep-alive connection dies mid-stream | retry once if nothing was yielded | `client.py:118-134` | Added after a live failure. |
| Vertex-only access for Gemma | `default_extra_body={"provider":{"order":["Google"]}}` | scripts only | **The CLI cannot set it** (`cli.py:254-259`), so Gemma via the CLI is not pinned. |

Things the layer does **not** handle:

- `Finish` and `Usage` events are produced (`client.py:251-269`) and **ignored by the agent loop** (`agent.py:133-149` handles only text, thought, tool deltas). A `finish_reason="length"` truncation is treated as a normal reply, or as a complete tool call. Token usage is never logged outside the benchmark scripts. **[R]**
- [I] The Gemma "concatenated JSON" fix probably papers over an **index collision** (two parallel calls streamed with the same `index`, which `tool_call_chunks.setdefault(event.index, …)` at `agent.py:141` merges). It only works when both calls have the same tool name. A mixed `task_spawn` + `task_status` pair at the same index would merge into invalid JSON under one name. I did not confirm the wire shape.
- `role="system"` messages mid-conversation (`history.py:65`) are sent as-is. They work on the tested Gemini and Gemma routes, but some backends reject or reorder them. [I]
- `max_tokens=2048` is shared with reasoning on some models. [I]

---

## 7. Where the design trusts the model (likely bug sources)

Structure enforces **none** of the following. Items are ordered by how much damage a bad outcome does.

1. **`allow_network` on bash** (`bash.py:80-84,103`). The model decides whether the sandbox network deny applies. No gate. **[R]**
2. **The sandbox existing at all.** If `bwrap` is missing, `execute_sandboxed` silently runs plain `bash -c` (`sandbox.py:77-78`). `bwrap` is present on this machine **[X]** (`/usr/bin/bwrap`). `SafeCommandChecker` is instantiated (`bash.py:95`) but `is_safe` is never called. The confirmation machinery has no caller (see §9). File tools run in-process outside the sandbox, containment by path check only (`filesystem.py:9-19`).
3. **Calling `worker_finish`.** Prompt-only (`tasks.py:406-410`). If skipped, the fallback declares `success=True` (`tasks.py:470-488`). In the robustness run, the two background tasks both failed in a non-git scratch directory (`bwrap` has no `~/.cache`, no git repo), and the models reported that correctly only because `worker_finish` was called.
4. **Honest `success`, `speech_summary`, `details`.** All model-authored. `details` goes into the receptionist's permanent history unbounded (`history.py:63-64`), and it is a prompt-injection channel from repo content, through the worker, into a model that holds `task_spawn`/`task_cancel`. [I]
5. **Choosing to spawn at all, and clarify-vs-act.** Prompt prose plus workspace in the tool description. Gemma and Flash-Lite passed the scripted scenarios, but those checks are substring-loose (§10).
6. **`isolate_worktree`** for mutating tasks (`orchestration.py:106-118`). Default off. Two mutating workers can edit the same tree.
7. **Disambiguating "cancel it" / "steer that"** across several running tasks. The annotation lists only id + title + step (`history.py:134`).
8. **Reporting progress truthfully.** The model has no worker progress to report (`TaskState.progress` unused). Its only signal is the step number, which is nearly constant (§4).
9. **Not looping.** Only 3 *identical consecutive* calls are caught (`agent.py:41-56`). An A,B,A,B alternation runs to the 30-turn cap.
10. **Well-formed tool args.** Failure degrades to a tool error and a retry, not a crash, except non-`dict` values. `bash.py:98` does `args.get("command","").strip()`, so an integer `command` raises `AttributeError`, which `agent.py:240` does not catch (only `RuntimeError`/`OSError`/`ValueError`). That propagates through `step()`. [I]
11. **Parallel `task_spawn` calls.** History records only the last ack (`agent.py:237-239` overwrites). Voice speaks every ack from `spawn_task`. Text and speech diverge. [R]

Structure that **does** enforce things (credit where due):

- The receptionist has no bash or file tools (`tasks.py:100-107`).
- The workspace is not model-selectable.
- A task spawn ends the turn without a second model call.
- The running annotation is never persisted and never shows finished tasks.
- Workers cannot see the chat.
- File paths are contained (`filesystem.py:9-19`).

---

## 8. Drift from `docs/proposal.md`

| Proposal | As built |
|---|---|
| Structured concurrency via `TaskGroup` (§5.1, Slice 6) | Bare `asyncio.create_task` + dict (`tasks.py:267`). No cleanup of `_worker_tasks`/`_steer_queues`. |
| `AgentSession` loop "under 200 lines" (§3.1) | 270 lines, and the Gemma splitter added about 30. |
| Deterministic runtime answers for status (§2.1.3, §5.2) | Only the running annotation and IPC `status`. The parity helper `append_runtime_turn` exists (`history.py:45-51`) and is **unused**. Terminal-spawn parity is hand-written at `agent.py:252`. |
| Steering: queue, **SIGINT to subshell**, inject at next tool boundary as a `user` message (§6.3) | Queue only. No SIGINT. Delivery effectively never happens: see below. |
| Confirmation gate for mutating commands, physical key, deny on timeout (§5.4) | The gesture machinery is built in the pipeline, IPC and CLI, but `request_confirmation` has **no production caller**. |
| Network "explicitly declared in task configuration or granted on request" (§5.4) | Model-controlled tool arg. |
| Barge-in prunes history to what was physically played (§5.5.5) | Prune function exists; it no-ops in the real sequence **[X]**. |
| Announcement `announced` flag, set after playback (§5.5.4) | Set (`player.py:140`) and **never read** by logic. Interrupted announcements are dropped, not retried. |
| Finish→speech ~130–150 ms (Budget B) | No trigger. Spoken only after the next voice turn. |
| Receptionist TTFS 200–350 ms (Budget A, "Estimated") | Measured in my previous pass: LLM TTFT p50 about 970 ms, p95 about 1160 ms for Flash-Lite `minimal`. The estimate was about 3× optimistic. |
| Worktree: offered merge/review at finish (§6.2) | No merge, no cleanup. Branch `worker/task-N`, never removed. |
| Workspace selection via alias / spoken override / focused window (§6.1) | Daemon default only. The model is deliberately prevented from choosing. |
| Pydantic config parsing; Rich CLI (`verbix chat`) | argparse and plain stdout. No config file. |
| Persistence deferred | Same. No persistence. |
| Hold-to-talk PTT | Toggle-only by later decision. Reasonable, but the proposal says hold. |

### Steering is a no-op mid-run **[X]**

Throwaway script: mock LLM, worker blocked in `bash sleep 1`, `steer_task` called mid-run.

```
llm calls: 3 task status: completed
0 steer visible to model: False
1 steer visible to model: False
2 steer visible to model: False
steer queue still holds: 1
```

The worker loop only drains the steer queue at the top of its outer iteration (`tasks.py:429-440`). Each `session.step()` runs the entire tool loop (up to 30 LLM turns) and returns when the model finishes. The outer loop then returns (`462-467`, `469-488`). So the instruction sits in the queue until the task is done. The unit test only asserts the item is in the queue (`test_runtime.py:203-213`). The robustness scenario "Steer: PASS" only checks that the model called `task_steer`.

### Abstractions added that the proposal did not have

- `is_terminal` on `ToolResult` (`tools/base.py:14`) controls the loop. This is a core-loop concept the proposal did not have. It is worth keeping.
- Separate receptionist/worker models and a reasoning-effort policy.
- `LiveSTTSession`, Silero VAD tail logic, the toggle state machine, the audible cue.
- A 10-action IPC protocol, a client class, and `--direct` mode, a second agent product with no receptionist.
- 8 benchmark/validation scripts (1,355 LOC).
- Typed `Task*Event` classes (unused).

---

## 9. Complexity audit

**Counts:** 66 classes in about 4.4k lines. 4 Protocols (`Tool`, `SpeechToText`, `TextToSpeech`, `AudioSink`). 10 IPC actions. 5 CLI run modes. 3 event dispatchers. 3 status paths.

**Resemblance to the old Go design.** Mostly it escaped. The language boundary is gone (one Python process). History no longer waits for audio receipts. There is no safety LLM. But three things rhyme with the Go failure modes:

1. `[Running: …]` re-projects task state into the prompt every turn. It is the same mechanism as Go's `taskSystemContext`, restricted to running tasks and kept out of history. That restriction is what fixed the Go bug, and it holds in code. But the tool-loop re-renders it each LLM call (`agent.py:117`), so a task that finishes mid-turn silently vanishes from the prompt while its completion is buffered until turn end (`tasks.py:349-354`). Within a multi-step turn the model sees it as neither running nor done. [R] Low impact today because receptionist turns are 1–2 calls.
2. The `announced` flag plus `announcement_q` is Go's announcement/receipt system in miniature, but the flag is never consulted. [R]
3. `SafeCommandChecker` (regex allowlist and deny patterns) is the same kind of tripwire the proposal said to avoid, and it is dead code.

**Dead or vestigial** (no production reader or caller, grep-verified **[R]**):

- `SafeCommandChecker.is_safe`, and its test.
- `VoicePipeline.request_confirmation`/`confirm`/`deny`, the IPC `confirm`/`deny` actions, and the CLI confirm loop.
- `ConversationHistory.is_turn_boundary`, `append_runtime_turn`.
- `TaskState.announced` (read-side), `TaskState.progress`.
- `VoicePipeline._is_active_turn` (write-only), `AudioPlayer._active_play_task` (never assigned).
- `FasterWhisperSTT._lock`.
- `TaskSpawnedEvent`/`TaskProgressEvent`/`TaskFinishedEvent`, `AudioChunk`.
- `ProviderClient.measure_network_rtt` (script-only).
- `WorktreeManager.remove_worktree` outside the stale-cleanup branch.

**What I would simplify or cut:**

1. Pick one: wire the confirmation gate to `allow_network` and non-safe bash, or delete the gate, `SafeCommandChecker` and the IPC/CLI plumbing (about 150 lines). Keeping the machinery with no caller costs tests and reader attention.
2. One `build_runtime(args)` in the CLI. Delete `--direct`, or declare it a supported product.
3. One event dispatcher. Let the pipeline and server consume a small adapter.
4. One client per endpoint with `model` passed per call. `_get_provider_for_model` creates a new `ProviderClient` and `httpx2.AsyncClient` per worker spawn (`tasks.py:109-120,413`) and `ProviderClient.aclose` is never called by the CLI. [R]
5. Make steering real (see §11) or remove `task_steer` from the model's tool list. Right now the model is told it has a capability that doesn't work.
6. Redefine `step_count` as LLM turns or tool calls, and surface a last-tool-call line instead. Without that, status answers have nothing to say.
7. Cap `details` in the injected completion and put the full text behind `task_status`.
8. Move the Gemma-specific JSON splitter and effort policy into the provider layer behind a per-model profile.

---

## 10. Test coverage

59 tests pass in about 30 s without coverage and about 76 s with it. **[X]** Coverage is **71%** (2,268 statements). Several tests need real model weights and `scratch/test_tts.wav` (Whisper `tiny.en`, PocketTTS synthesis). The suite is not hermetic.

| Area | Cover | What the gaps mean |
|---|---|---|
| `cli.py` | 21% | Every run mode body (`206-501`) is untested. Only the argparse defaults are. |
| `voice/sinks.py` | 27% | The real playback path is never exercised. Only config is tested. |
| `ipc/server.py` | 61% | The `prompt` path (`166-218`), `steer`, `confirm` and `deny` are untested. |
| `voice/pipeline.py` | 78% | `execute_voice_turn` (`173-197`) is **entirely uncovered**. |
| `runtime/worktree.py` | 58% | The git worktree path is untested. Only the non-git fallback runs. |
| `runtime/tasks.py` | 80% | Worker steering drain (`431-440`), the finish-capture branch (`463-467`), the no-finish fallback (`477-488`) and the exception path (`501-508`) are untested. **No test runs a real `_run_worker` loop to completion.** |
| `core/agent.py` | 91% | The Gemma multi-JSON split branch (`176-178`) is untested. |
| `provider/client.py` | 81% | The retry wrapper (`129-134`) and the 400/5xx branches are untested. |
| `tools/sandbox.py` | 76% | The timeout/cancel kill path (`97-105`) is untested. |

**Loose or misleading checks** **[R]**:

- `test_agent_cycle_detection` (`test_agent.py:213-238`) passes on `"loop detected"` **or** `"stopped"`, and never checks that history is left valid. Production code returns at `agent.py:214` with an assistant `tool_calls` message and **no tool results** in history. The next request to a strict provider will fail with 400. [I] on the 400, [R] on the dangling message. No test sends a second turn.
- `test_task_completion_sync_before_foreground_turn` (`test_runtime.py:252-308`) hand-pokes `_pending_completions` and `task.status`. It does not drive the real finish → turn race.
- Barge-in tests set `player._is_playing = True` by hand (`test_voice.py:128,276`) and never run a live turn. They could not catch the behavior shown in §3C.
- `test_task_runtime_steering` only checks that the queue holds the item. This is why the steering no-op shipped.
- Cancellation test accepts `"Failed" in … or "cancelled" in …` (`test_runtime.py:196-199`).
- `test_safe_command_checker` tests code that production never calls.
- The scenario suite is a script, not pytest. It calls live models and uses substring matching: ambiguous request passes on any of `what/clarif/how/better/mean`. Steer passes if `task_steer` was *called*, not if it had any effect. In the final run, 18 of 18 scenario cells passed, so the suite is a smoke test and not a quality measure.

---

## 11. Top risks and hardest-to-change things

### Top risks (my ranking)

1. **Barge-in does not stop the turn [X].** The model's remaining text is still spoken while the mic records, and history diverges from speech. This is the core voice interaction.
2. **Steering never reaches the worker [X].** The tool is advertised to the model and has no effect while the task runs.
3. **Background completions are not spoken when they happen [R].** A user who says "run the tests" and goes idle hears nothing until their next voice turn finishes.
4. **No turn serialization [R].** Two overlapping turns (double toggle, voice plus IPC) corrupt history ordering. `is_main_turn_active` is a bool.
5. **History can be left invalid [R].** The cycle-detect exit, a cancel during tool execution, and a provider exception mid-turn all leave an assistant `tool_calls` message or a lone user message. No recovery. Combined with unhandled `ProviderError` in the CLI and voice paths (§5), a single bad turn can end the session or brick it.
6. **Sandbox is advisory [R].** The model can enable network. Without `bwrap` it silently runs unsandboxed. File tools bypass the sandbox. Confirmation is unwired.
7. **`success=True` fabrication on a worker that just stops [R].** Receptionist then tells the user it succeeded.
8. **Worktree destruction across sessions [R].** Task ids restart at `task-1` every process. `create_worktree` deletes any existing `.verbix/worktrees/task-1` and force-deletes branch `worker/task-1` (`worktree.py:45-47,69-98`). A second session's isolated task wipes the first session's unmerged work. Also [I]: a git worktree's `.git` file points outside the sandbox bind, so git inside a worktree task probably fails under `bwrap`.
9. **Unbounded context [R].** No history cap, and worker `details` are injected in full.
10. **`Finish`/`Usage` ignored [R].** Truncated replies look complete, and there is no token telemetry in the product.

### Hardest to change later

- **`AgentSession.step` as a monolithic generator that writes history between yields.** Cancellation safety, terminal tools, and every client's event contract depend on it. Making history crash-consistent (append the assistant message and tool results atomically, or repair on resume) touches the loop and every writer.
- **`role="system"` completion messages in the main history.** Switching provider routes or adding persistence turns this shape into a compatibility problem.
- **The runtime ↔ player ↔ pipeline mutual references.** `runtime.player: Any`, `AudioPlayer` writing task state, the pipeline reaching into `runtime.main_session.history`. The planned Rust shell is meant to be a thin client, but audio ownership and announcement policy currently live half in `runtime/` and half in `voice/`.
- **No single-writer discipline.** Adding a session actor or turn lock later means threading it through the CLI, IPC, and voice entry points, which currently each drive `step_receptionist` themselves.
- **In-memory-only state.** Adding persistence means serializing history, tasks, and in-flight workers, and defining resume semantics for dangling tool calls and running workers.
- **The task-event contract.** Untyped dicts today, with typed unused classes beside them. Whichever the shell binds to first becomes frozen.

---

## 12. Diagram

```
        keyboard / hotkey (toggle)            terminal / IPC client
                 │                                     │
                 ▼                                     ▼
   ┌───────────────────────────┐         ┌───────────────────────────┐
   │ VoicePipeline             │         │ cli.py REPL    VerbixServer│
   │  capture → LiveSTT (VAD)  │         │ (render)       (JSONL/UDS) │
   │  TextDelta → sentences    │         └─────────────┬─────────────┘
   └───────┬──────────▲────────┘                       │
           │          │ speak / interrupt              │
           │   ┌──────┴────────┐                       │
           │   │ AudioPlayer   │◄── announcement_q ────┤ (drained ONLY after a voice turn)
           │   │ TTS → pw-play │                       │
           │   └──────▲────────┘                       │
           ▼          │ sets .announced, .player       ▼
   ┌────────────────────────────────────────────────────────────┐
   │ TaskRuntime  (no turn lock)                                │
   │   tasks{}  _steer_queues  _pending_completions  subscribers│
   │   step_receptionist ─► AgentSession(receptionist)          │
   │       tools: task_spawn* task_cancel task_steer task_status│
   │   _run_worker ───────► AgentSession(worker, own history)   │
   │       tools: bash file_* worker_finish*  (*terminal)       │
   │   _finish_task ──► history.inject_task_completion (system) │
   └───────────────┬───────────────────────────┬────────────────┘
                   ▼                           ▼
        ProviderClient (httpx2, SSE)     bwrap sandbox / worktrees
        OpenRouter, reasoning effort     (advisory; model sets network)
```

---

## 13. Open questions for you

1. **Barge-in semantics.** When the user toggles during speech, should the whole receptionist turn be cancelled (LLM stream, queued sentences, and un-spoken history pruned), or should the text keep streaming silently into history? The parity rule implies the former.
2. **Steering.** Should a steer interrupt the worker's current LLM/tool step, or be injected at the next tool boundary? The proposal says both (SIGINT and next-step). Or should steering be dropped for now?
3. **Proactive completion speech.** Should a finished task speak immediately even if the user is mid-conversation, and when it collides with a playing reply, queue or interrupt?
4. **Concurrency contract.** Is one foreground turn at a time the rule, with a second input cancelling the first, queueing behind it, or rejected?
5. **Trust boundary.** Should `allow_network` and `isolate_worktree` leave the model's hands? And is running without `bwrap` an error or a warning?
6. **Confirmation gate.** Keep it (then what triggers it?) or delete it?
7. **`--direct`.** A supported second product, or remove?
8. **Worker failure semantics.** Should a worker that stops without `worker_finish` be `failed`, or be re-prompted once to call it?
9. **Context growth.** What is the history policy: cap, summarize, or persist-and-trim? And should completion `details` be injected in full or by reference?
10. **Persistence.** Is losing running workers and history on restart acceptable for the Rust-shell milestone?
11. **Receptionist model.** The proposal's TTFS budget was about 3× optimistic. Is the target still about 1 s key-release-to-first-audio, which needs a model change or a canned instant ack, or is the target revised?
12. **Task ids and worktrees.** Should ids be globally unique, for example timestamp-based, and should worktrees be cleaned on finish?
