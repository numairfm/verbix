# Verbix

Voice-operated desktop assistant. Talk, keep talking while work happens in the background, hear each result out of the same voice. Text REPL, voice mode, and a headless IPC daemon.

```bash
uv sync
# then provide a key via .env or environment (see Configuration)
uv run verbix
```

```
you> what time is it in tokyo
verbix: It is 6:42 PM in Tokyo right now.
you> summarize this repo in the background
verbix: task-1 started in background.
verbix: Task 'task-1' is done — the repo is a voice agent with a foreground
receptionist and background workers. [spoken when idle]
```

## Run modes

```bash
uv run verbix                          # interactive text REPL
uv run verbix "do X in background"      # one-shot (waits for workers, then exits)
uv run verbix --voice                   # push-to-talk voice loop (Enter to record)
uv run verbix --serve                   # daemon on ~/.local/share/verbix/verbix.sock
uv run verbix --serve --voice           # daemon with voice pipeline attached
uv run verbix --toggle                  # send mic toggle to a running daemon
```

Useful flags: `--model`, `--receptionist-model`, `--worker-model`, `--workspace`,
`--max-tokens`, `--show-thoughts`, `--system-prompt`, `--socket-path`.

## Configuration

Env file lookup: `--env` path → `./.env` → `~/.config/verbix/.env`. Existing
environment wins over file values (`cli.py:29`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` / `VERBIX_API_KEY` | — | provider key (LiteLLM auto-detects per provider) |
| `VERBIX_MODEL` (`OPENROUTER_MODEL` fallback) | `openrouter/google/gemini-3.5-flash-lite` | foreground + default worker model |
| `VERBIX_BASE_URL` | provider default | custom endpoint |
| `VERBIX_MAX_TOKENS` | `2048` | per-turn completion cap |

Agent tuning lives in `src/verbix/agent/config/`: `voice.yaml`
(foreground), `worker.yaml` / `swe_worker.yaml` (background workers).

### Global config and persona (`~/.verbix/`)

All Verbix configuration, auth, persona, models, screenshots, task scratch directories, and tool spill logs live under `~/.verbix/` (never inside project workspaces):
- Config precedence: builtin defaults < `~/.verbix/config.yaml` < env < CLI flags. `config.yaml` is optional and accepts `provider` and `stt_model`:

```yaml
# stt_model: /path/to/model.gguf
```

First run scaffolds `~/.verbix/` with a commented `config.yaml` and an empty `PERSONA.md`.

`~/.verbix/PERSONA.md` is an overlay appended to the system prompt under `# USER PERSONA` — it never replaces core prompts. A leading `<!-- ... -->` guidance block is stripped. `--system-prompt` replaces file persona entirely when set.

Provider selection: `--provider` flag > `VERBIX_PROVIDER` env > config `provider` name. The name selects an entry in `~/.verbix/auth.json`:

```json
{
  "openrouter": {
    "type": "api",
    "key": "sk-...",
    "model": "openrouter/google/gemini-3.5-flash-lite",
    "base_url": "https://openrouter.ai/api/v1"
  }
}
```

`--model` beats the entry model; `--api-key`/`--base-url` beat the entry key/base URL.

Audio (`sounddevice`) and on-device voice inference are included in the default environment (`uv sync`). Voice mode uses Supertonic 3 INT8 for TTS (via sherpa-onnx) and Nemotron 3.5 ASR 0.6B GGUF for STT (via transcribe.cpp). Run `uv run scripts/fetch_voice_models.py` to download and verify models into `./voice-runtime` (`VERBIX_VOICE_DIR` overrides, `~/.verbix/voice` is the installed fallback). Attribution: `THIRD-PARTY-NOTICES.md`.

## Architecture

Two layers. The agent layer does the work; the voice layer owns the floor.

### Agent layers

```
              user
               │
               ▼
   ┌───────────────────────┐
   │ main agent            │  tools: spawn(objective)
   │ (foreground)          │         status(task) · cancel(task)
   └───────────┬───────────┘  no filesystem, no shell
               │ spawn
   ┌───────────┼───────────┐
   ▼           ▼           ▼
 worker 1   worker 2   worker 3   tools: read/write/edit/grep/glob
                                          web_search/web_fetch/screenshot
                                          clipboard/media/app/open + bash (bwrap)
   └───────────┼───────────┘  (workers never see the chat —
               │ responses   only the spawn objective)
               ▼
   ┌───────────────────────┐
   │ TaskLedger +          │  completions fold into next turn
   │ _pending_announcements│  or drain proactively when silent
   └───────────┬───────────┘
               ▼
   ┌───────────────────────┐
   │ announcer companion   │  toolless VoiceAgent speaks each
   │ (same voice)          │  result through wake_output
   └───────────┬───────────┘
               ▼
              user (one voice)
```

The foreground (`Runtime.foreground`, a `VoiceAgent`) answers directly or delegates via `spawn` / `status` / `cancel` (`tools/spawn.py`, `tools/status.py`, `tools/cancel.py`). Each worker (`runtime/worker.py:WorkerTask`) runs its own `WorkerAgent` (`agent/worker.py`) in the workspace, started with `asyncio.create_task`. On finish, the result is recorded in `TaskLedger` (`runtime/ledger.py`, `delivery="pending"`) and either voiced proactively by `Runtime.announcer` when the floor is available (`floor.can_announce()`) or folded into the user's next foreground turn and marked `"heard"` once delivered.

Foreground sees the full chat plus active/newly-completed task grounding; a worker sees only its spawn objective and concludes with a final text answer when done. Cancelled workers are marked cancelled and stay silent beyond the cancel acknowledgement.

### Voice layer

One floor, three states (`runtime/floor.py`):

```
IDLE ──toggle──▶ LISTENING ──toggle──▶ SPEAKING ──drained──▶ IDLE
                       ▲                    │
                       └──── barge-in ───────┘  (stop audio, cancel turn,
                                                 repair history, record)
```

`VoicePipeline` (`voice/pipeline.py`): `PTTCapture` → live `TranscribeSTT`
session → `runtime.submit(text, SpeechOutput)` → sentence-streamed in-process
`SherpaTTS` → `AudioPlayer` → `CallbackAudioSink`. Toggle during playback
or a running turn interrupts first (`interrupted_and_recording`). Empty
transcriptions return to `IDLE` without a model call. `SpeechOutput`
(`voice/output.py`) owns sentence buffering and `delivered_text()`, which is
what `cancel_turn` uses to trim history to what was actually heard
(`runtime/runtime.py:571-580`).

Example walk (matches the whiteboard):

```
STATE: IDLE
*user presses Enter → STATE: LISTENING (mic + live STT, speaker muted)
*user presses Enter → transcription → STATE: SPEAKING
  user: Hello!
  agent: Hello, how are you?
*after playback drains → STATE: IDLE
```

## Module map

| Path | Owns |
|---|---|
| `cli.py` | argparse, `.env` loader, 4 run modes, `CliPrinter` (stdout transcript, stderr diagnostics) |
| `runtime/runtime.py` | `Runtime`: foreground/announcer agents, `submit`/`spawn`/`status`/`cancel`, pending-announcement drain |
| `runtime/workers.py` | `build_worker_registry`, `describe_worker_tools`, `spawn_worker`, task completion/failure handlers |
| `runtime/worker.py` | `WorkerTask`: background `asyncio.Task`, result/error capture, `get_status` |
| `runtime/ledger.py` | `TaskLedger`: id, objective, status, delivery (`pending`/`heard`/`interrupted`) |
| `runtime/floor.py` | `FloorArbiter`: `IDLE`/`LISTENING`/`SPEAKING` + sticky `user_speaking` + interrupt flag |
| `core/` | canonical messages/events, history repair |
| `provider/` | streaming OpenAI-compatible SSE client, errors, event types |
| `tools/` | `Tool`/`ToolContext`, `read`/`write`/`edit`/`grep`/`glob`, `web_search`/`web_fetch`, `screenshot`, `clipboard`/`media`/`app`/`open`, `spawn`/`status`/`cancel`, and `bash` via mandatory bubblewrap (`bwrap`) |
| `voice/` | `pipeline`, `capture`, `transcribe_stt` (Nemotron ASR via transcribe.cpp), `sherpa_tts` (Supertonic TTS via sherpa-onnx), `player`, `output` (sentence streaming), `callback_sink`, `base` protocols + mocks |
| `ipc/` | `server`/`client`/`protocol`: JSONL over Unix socket, `ping/prompt/status/cancel/toggle`, runtime event fan-out |
| `agent/` | agent core: `voice.py` (`VoiceAgent`), `worker.py` (`WorkerAgent`), `models/litellm_model.py`, `tools/registry.py`, `config/*.yaml` |

## IPC

`VerbixServer` fronts the same `Runtime`. Actions: `ping`, `prompt`, `status`,
`cancel`, `toggle`. Runtime events (`task_registered`, `task_completed`,
`task_cancelled`, `wake_event`) fan out to subscriber queues.

## Develop

```bash
uv run pytest -q            # full suite
uv run ruff check src tests
uv run pyright src/verbix   # third-party import noise only (pydantic/httpx2/numpy)
```

Conventions: `ruff` clean, `pyright` no new errors. Voice/audio paths use the
`Mock*` protocols in `voice/base.py` for headless tests. Live voice tests need
provisioned models via `uv run scripts/fetch_voice_models.py`.

## Platform support

- **Linux (`x86_64`, `aarch64`)**: Supported tier. Requires system packages `bubblewrap` (`bwrap`, for mandatory worker `bash` sandboxing) and `libportaudio2` (`portaudio`, for `sounddevice` audio capture and playback).
- **Windows (`amd64`) & macOS (`arm64`, `x86_64`)**: Experimental and untested by the maintainer. No sandbox backend is enabled on Windows or macOS, so the `bash` tool is automatically omitted from workers (with a startup notice on `stderr`), and workers operate using the 12 cross-platform workspace/web/desktop tools (`read`, `write`, `edit`, `grep`, `glob`, `web_search`, `web_fetch`, `screenshot`, `clipboard`, `media`, `app`, `open`). IPC (`--serve` / `--toggle`) uses POSIX Unix domain sockets.

## Limits (as built)

- Nothing survives restart: history, ledger, workers are in-memory only.
- `buffer_window` (0.3 s) is accepted by `Runtime` but not wired up —
  completions drain immediately when the floor goes idle, under a lock, not in
  timed batches.
- Workers get the objective only, never the chat; no steering a running worker
  except cancel.
- Rolling compaction keeps foreground context near ~16k tokens
  (`max_context_tokens`); no token/cost telemetry.
- One network/sandbox posture per `Runtime`, not per task.
