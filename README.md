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
uv run verbix --voice                   # push-to-talk voice mode (Enter to start/stop)
uv run verbix --serve                   # daemon on ~/.local/share/verbix/verbix.sock
uv run verbix --serve --voice           # daemon with voice pipeline attached
uv run verbix --toggle                  # send mic toggle to a running daemon
```

Voice mode (`--voice`) uses push-to-talk: press Enter (or run `verbix --toggle`) to start recording, speak, and press Enter again to transcribe and submit. Pressing Enter while Verbix is speaking interrupts playback immediately and starts recording; interrupted announcements are never re-spoken automatically and fold into your next turn's context.

Useful flags: `--model`, `--receptionist-model`, `--worker-model`, `--workspace`,
`--max-tokens`, `--retention-days`, `--show-thoughts`, `--system-prompt`, `--socket-path`.

## Configuration

Env file lookup: `--env` path → `./.env` → `~/.config/verbix/.env`. Existing
environment wins over file values (`cli.py:29`).

| Variable | Default | Purpose |
|---|---|---|
| `OPENROUTER_API_KEY` / `VERBIX_API_KEY` | — | provider key (LiteLLM auto-detects per provider) |
| `VERBIX_MODEL` (`OPENROUTER_MODEL` fallback) | `openrouter/google/gemini-3.5-flash-lite` | foreground + default worker model |
| `VERBIX_BASE_URL` | provider default | custom endpoint |
| `VERBIX_MAX_TOKENS` | `2048` | per-turn completion cap |
| `VERBIX_RETENTION_DAYS` | `14` | days to retain tool logs and screenshots |
| `VERBIX_TTS_VOICE` | `M1` | Supertonic TTS voice (`M1`–`M5`, `F1`–`F5`) |
| `VERBIX_TTS_SPEED` | `1.0` | Supertonic TTS speaking speed (`0.25`–`3.0`) |

Agent tuning lives in `src/verbix/agent/config/`: `voice.yaml`
(foreground), `worker.yaml` / `swe_worker.yaml` (background workers).

### Global state (`~/.verbix/`) vs. workspaces

Verbix separates its machine-global state under `~/.verbix/` from project workspaces:

- **Workspace (`--workspace`, default `~/.verbix/workspace/`)**: The working directory exposed read-write to background workers (`read`, `write`, `edit`, `grep`, `glob`, and sandboxed `bash`). When `--workspace` is omitted, Verbix uses a dedicated `~/.verbix/workspace/` directory rather than the current working directory, and refuses `$HOME` or `/` unless explicitly overridden with `--workspace`. Workers run directly in the configured workspace (with `HOME=/tmp` on `bwrap`'s ephemeral `tmpfs` so CLI tool dotfile noise vanishes on exit) and never create separate task or sandbox home folders.
- **Global state (`~/.verbix/`)**: Holds user configuration, persona, persistent memory, credentials, runtime logs/screenshots, and user-managed STT models:
  - `config.yaml`, `PERSONA.md`, `MEMORY.md`, `auth.json`: global configuration, persona overlay, persistent cross-session user memory, and provider credentials (never touched by retention cleanup).
  - `logs/`: spilled full outputs from truncated tool observations (`tool_<hex>`).
  - `screenshots/`: rolling PNG archive of captured desktop screenshots (`screenshot_<timestamp>.png`).
  - `voice/stt/models/`: user-managed speech-to-text (STT) GGUF models, organized by model ID (default `~/.verbix/voice/stt/models/nemotron-gguf/`), created lazily when provisioned or migrated.
- **Retention policy**: At startup, Verbix deletes entries under `~/.verbix/logs/` and `~/.verbix/screenshots/` older than `retention_days` (default `14` days; configurable via `retention_days` in `~/.verbix/config.yaml`, `VERBIX_RETENTION_DAYS`, or `--retention-days`). Retention cleanup never touches `config.yaml`, `PERSONA.md`, `MEMORY.md`, `auth.json`, workspace contents, or voice models.

Config precedence: builtin defaults < `~/.verbix/config.yaml` < env < CLI flags. `config.yaml` is optional and scaffolds with all available options commented out (`provider`, `model`, `receptionist_model`, `worker_model`, `max_tokens`, `worker_max_tokens`, `workspace`, `stt_model`, `tts_voice`, `tts_speed`, `retention_days`, `show_thoughts`, `show_worker_turns`):

```yaml
# provider: openrouter
# model: openrouter/google/gemini-3.5-flash-lite
# receptionist_model: openrouter/google/gemini-3.5-flash-lite
# worker_model: openrouter/google/gemini-3.5-flash-lite
# max_tokens: 2048
# worker_max_tokens: 8192
# workspace: ~/.verbix/workspace
# retention_days: 14
# stt_model: /path/to/model.gguf
# tts_voice: M1
# tts_speed: 1.0
# show_thoughts: false
# show_worker_turns: false
```

First run scaffolds `~/.verbix/` with a commented `config.yaml`, an empty `PERSONA.md`, and an empty `MEMORY.md`.

`~/.verbix/PERSONA.md` is an overlay appended to the system prompt under `# USER PERSONA` — it never replaces core prompts. A leading `<!-- ... -->` guidance block is stripped. `--system-prompt` replaces file persona entirely when set.

`~/.verbix/MEMORY.md` stores durable facts and preferences across sessions (managed via the `memory` tool or edited directly on disk) and is dynamically injected into `# ENVIRONMENT & GROUNDING` on every turn.

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

### Voice models: user-managed STT vs. bundled TTS

Audio (`sounddevice`) and on-device voice inference are included in the default environment (`uv sync`):
- **STT (user-managed)**: Uses Nemotron 3.5 ASR 0.6B GGUF (via `transcribe.cpp`) or another user-selected GGUF model (`--stt-model`, `VERBIX_STT_MODEL`, or `stt_model` in `config.yaml`). Models live under `~/.verbix/voice/stt/models/<model-id>/` (`VERBIX_VOICE_DIR` or `./voice-runtime` overrides the voice root). Run `uv run scripts/fetch_voice_models.py` to download and verify the default Nemotron STT model.
- **TTS (bundled application resource)**: Uses Supertonic 3 INT8 (via `sherpa-onnx`), packaged directly with the application under `verbix/voice/tts/supertonic-int8/` rather than stored as a user-managed model in `~/.verbix`. No first-run TTS download into `~/.verbix` is required.
- **Upgrades and legacy files**: Existing Nemotron STT files at `~/.verbix/voice/models/nemotron-gguf/` are automatically migrated to `~/.verbix/voice/stt/models/nemotron-gguf/` (without overwriting any existing destination file or redownloading). Legacy TTS files under `~/.verbix/voice/models/supertonic-int8/` and `provenance.json` under `~/.verbix/voice/models/` are intentionally left untouched during upgrades; once you verify voice mode works with the new setup, you can safely delete `~/.verbix/voice/models/`. Attribution: `THIRD-PARTY-NOTICES.md`.

## Architecture

Two layers. The agent layer does the work; the voice layer owns the floor.

### Agent layers

```
              user
               │
               ▼
   ┌───────────────────────┐
   │ main agent            │  tools: spawn(objective) · status(task)
   │ (foreground)          │         cancel(task) · memory(...)
   └───────────┬───────────┘  no filesystem, no shell
               │ spawn
   ┌───────────┼───────────┐
   ▼           ▼           ▼
 worker 1   worker 2   worker 3   tools: read/write/edit/grep/glob
                                          web_search/web_fetch/screenshot
                                          clipboard/media/app/open/system
                                          memory/notify + bash (bwrap)
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
| `runtime/floor.py` | `FloorArbiter`: `IDLE`/`LISTENING`/`SPEAKING` + interrupt flag |
| `core/` | canonical messages/events, history repair |
| `provider/` | streaming OpenAI-compatible SSE client, errors, event types |
| `tools/` | `Tool`/`ToolContext`, `read`/`write`/`edit`/`grep`/`glob`, `web_search`/`web_fetch`, `screenshot`, `clipboard`/`media`/`app`/`open`/`system`/`memory`/`notify`, `spawn`/`status`/`cancel`, and `bash` via mandatory bubblewrap (`bwrap`) |
| `voice/` | `pipeline`, `capture`, `transcribe_stt` (Nemotron ASR via transcribe.cpp), `sherpa_tts` (Supertonic TTS via sherpa-onnx), `player`, `output` (sentence streaming + two-tone 660 Hz / 880 Hz PTT earcons), `callback_sink`, `base` protocols + mocks |
| `ipc/` | `server`/`client`/`protocol`: JSONL over Unix socket (with loopback TCP fallback on Windows), `ping/prompt/status/cancel/toggle`, runtime event fan-out |
| `agent/` | agent core: `voice.py` (`VoiceAgent`), `worker.py` (`WorkerAgent`), `models/litellm_model.py`, `tools/registry.py`, `config/*.yaml` |

## IPC & Desktop Shell (Tauri)

`VerbixServer` (`verbix --serve --voice`) fronts the same `Runtime` and `VoicePipeline` over `~/.local/share/verbix/verbix.sock` (falling back to `127.0.0.1` loopback TCP on Windows).
- **Actions**: `ping`, `prompt`, `status`, `cancel`, `toggle`.
- **Streamed JSONL events**: `floor_state` (`idle` / `listening` / `speaking`), `user_transcript`, `task_registered`, `task_completed`, `task_failed`, `task_cancelled`, `wake_event`, `text_delta`, `turn_complete`.

The v1 desktop shell is a Tauri 2 process that owns the tray, global shortcut, autostart, single instance, panel, and voice service supervision. It runs the Python voice service unchanged as a child over this IPC protocol. The surface is tray-first; there is no overlay in v1. See [ADR-0001](docs/adr/0001-tauri-desktop-shell.md) for the shell decision, [ADR-0002](docs/adr/0002-rust-shell-owns-desktop.md) for process ownership, [ADR-0003](docs/adr/0003-tray-first-no-overlay.md) for the tray-first surface, and the [desktop shell v1 spec](docs/superpowers/specs/2026-10-20-desktop-shell-v1-design.md). GPUI is deferred, not rejected: the IPC seam lets the panel be replaced later without touching Python.

- **First-run setup**: the panel opens the setup when the provider key, the speech model, or the setup marker is missing. It writes the OpenRouter key to `~/.verbix/auth.json` (mode `0600`), downloads the speech model with the bundled downloader, and offers the autostart entry. The voice service starts only after the setup is complete. Tray **Setup…** reopens the same surface for repair.
- **Global shortcut**: `Ctrl+Alt+Space` toggles listening. X11 and XWayland register it natively; a native Wayland session (or a refused grab) shows the desktop-level fallback command, `"<path to the AppImage>" --toggle`, in the panel with a Copy button.
- **Single instance**: one shell per user. A second launch forwards its arguments (`--show`, `--hidden`, `--toggle`) to the running shell and exits; `--version` and `--help` print before the GUI, and an unknown argument exits 2.

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
- **Windows (`amd64`) & macOS (`arm64`, `x86_64`)**: Experimental and untested by the maintainer. No sandbox backend is enabled on Windows or macOS, so the `bash` tool is automatically omitted from workers (with a startup notice on `stderr`), and workers operate using the 15 cross-platform workspace/web/desktop tools (`read`, `write`, `edit`, `grep`, `glob`, `web_search`, `web_fetch`, `screenshot`, `clipboard`, `media`, `app`, `open`, `system`, `memory`, `notify`). IPC (`--serve` / `--toggle`) uses Unix domain sockets on POSIX and loopback TCP on Windows.

## Limits (as built)

- Chat history, task ledger, and workers are in-memory only (`~/.verbix/MEMORY.md` persists saved user facts across restarts).
- Workers get the objective only, never the chat; no steering a running worker
  except cancel.
- Rolling compaction keeps foreground context near ~16k tokens
  (`max_context_tokens`); no token/cost telemetry.
- One network/sandbox posture per `Runtime`, not per task.
