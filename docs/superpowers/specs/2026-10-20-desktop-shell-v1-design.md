# Verbix Desktop Shell v1 Spec

**Date**: 2026-10-20  
**Status**: Approved. Slice 1 (walking skeleton) may begin; later slices wait for their tickets.  
**Target**: HEAD `bae5915`, plus the untracked `docs/v1.md`, `GLOSSARY.md`, and ADR-0001 through ADR-0006.  
**Normative references**: `docs/v1.md`, `GLOSSARY.md`, `docs/adr/0001-*` through `0006-*`. If this spec conflicts with them, they win.

## 1. Scope

Verbix v1 is a Linux x86_64 desktop app. The shell is the long-lived Rust/Tauri 2 process. It owns the tray, global shortcut, autostart, single instance, panel, and the voice service child process. The Python voice service is the existing `verbix --serve --voice` process, unchanged, reached over the existing JSONL IPC.

Frozen decisions, not reopened here:

| Decision | ADR |
|---|---|
| Tauri 2 for the shell and panel | `docs/adr/0001-tauri-desktop-shell.md` |
| Rust owns OS surfaces; Python stays the voice service | `docs/adr/0002-rust-shell-owns-desktop.md` |
| Tray-first; no overlay in v1 | `docs/adr/0003-tray-first-no-overlay.md` |
| Linux ships; macOS and Windows are compiled seams | `docs/adr/0004-linux-first-universal-seams.md` |
| AppImage, user chooses whether and where to install | `docs/adr/0005-appimage-user-chosen-install.md` |
| Native shortcut grab first, desktop shortcut fallback | `docs/adr/0006-global-shortcut-native-with-fallback.md` |

Out of v1: settings UI, overlay, auto-update, signing, store distribution, runtime redesign, panel text input, task steering from the panel, history across sessions, shortcut customization, provider selection, model selection, log viewer. The first-run setup is the only settings surface.

## 2. Grounding

| Fact | Evidence |
|---|---|
| The voice service runs with `--serve --voice` and the default socket | `src/verbix/cli.py:554-585`, `src/verbix/cli.py:533-535` |
| The service prints a startup line on stdout and waits | `src/verbix/cli.py:570-579` |
| Shutdown on the clean path closes server, runtime, TTS, and sink | `src/verbix/cli.py:576-585` |
| IPC is newline-delimited JSON with `ping`, `prompt`, `status`, `cancel`, `toggle` | `src/verbix/ipc/server.py:286-377`, `src/verbix/ipc/protocol.py:9-33` |
| Runtime events are wrapped as `runtime_event` | `src/verbix/ipc/server.py:229-242` |
| Assistant text is broadcast as direct `text_delta` and `turn_complete` | `src/verbix/ipc/server.py:53-73`, `src/verbix/ipc/server.py:100-115` |
| Floor changes emit `floor_state` | `src/verbix/runtime/runtime.py:456-466` |
| User speech emits `user_transcript` | `src/verbix/voice/pipeline.py:117-139` |
| Task transitions emit `task_registered`, `task_completed`, `task_failed`, `task_cancelled` | `src/verbix/runtime/workers.py:88`, `:109`, `:195`, `:209-211` |
| Voice startup fails when the STT model is missing | `src/verbix/voice/transcribe_stt.py:163-172` |
| STT downloader already verifies a pinned model by SHA-256 | `scripts/fetch_voice_models.py:20-33`, `:143-173` |
| Bundled TTS is a verified application resource | `src/verbix/voice/sherpa_tts.py:45-49`, `src/verbix/_build_backend.py:22-36` |
| The `verbix` script points at `verbix:main`; `python -m verbix.cli` also works | `pyproject.toml:31-32`, `src/verbix/__init__.py:1-7`, `src/verbix/cli.py:740-741` |
| Global state is `~/.verbix`; config precedence is file < env < CLI | `src/verbix/config.py:119-190`, `src/verbix/config.py:236-259` |
| Provider credentials live in `~/.verbix/auth.json`; provider selection comes from config/env/CLI | `src/verbix/auth.py:18-20`, `src/verbix/auth.py:22-62`, `src/verbix/config.py:119-129` |

## 3. Architecture

```
autostart / AppImage launch / second launch
                  │
                  ▼
        ┌──────────────────────┐
        │ Shell (Rust, Tauri)  │
        │ tray · panel · state │
        │ shortcut · autostart │
        │ single instance      │
        └───────┬──────────────┘
                │ JSONL IPC, unchanged
                │ UDS on Linux/macOS
                │ loopback TCP on Windows
                ▼
        ┌──────────────────────┐
        │ Voice service        │
        │ (Python, unchanged)  │
        │ runtime · voice      │
        │ tools · provider     │
        └──────────────────────┘
```

Ownership rules:

- The shell never runs the agent, captures audio, speaks, or calls a provider. It sends `toggle` and renders events.
- The voice service never draws UI, registers shortcuts, or writes autostart.
- The shell never shells out to `verbix --toggle`; it speaks IPC directly.
- Provider wire behavior stays in `provider/` inside Python.
- At runtime, the shell may start the voice service, stop it, restart it, and write its log. Nothing more.

The sidecar is the same voice service, not a new mode. No IPC action is added, removed, or renamed.

## 4. Voice service contract

### 4.1 Launch

The AppImage carries a relocatable Python runtime and the installed wheel:

```
<resources>/voice/python/bin/python3 -m verbix.cli --serve --voice
```

No shell, no string interpolation, fixed arguments. The shell must not pass `--socket-path`; the service keeps its default socket so `uv run verbix --toggle` from a terminal still reaches it.

Environment changes:

| Variable | Value | Reason |
|---|---|---|
| `PYTHONPATH` | `<resources>/voice/site-packages` | installed wheel and dependencies |
| `PYTHONHOME` | unset | never inherit a host prefix |
| `PYTHONNOUSERSITE` | `1` | never read the host user's site-packages |
| `PYTHONDONTWRITEBYTECODE` | `1` | AppImage is read-only |
| `VERBIX_VOICE_DIR` | `<home>/.verbix/voice` | deterministic STT root |

Working directory: `~/.config/verbix`, created as `0700` if absent. This is the existing env loader's own directory, so `./.env` and `~/.config/verbix/.env` are the same file for users who already keep a key there. The shell never writes a key to that directory. Everything else is inherited, including display, audio, DBus, and `PATH` for `bwrap`.

Provider selection: when `~/.verbix/auth.json` has an `openrouter` entry with a non-empty `key` and neither `VERBIX_PROVIDER` (process environment or `.env`) nor a top-level `provider:` line in `~/.verbix/config.yaml` selects one, the shell adds `VERBIX_PROVIDER=openrouter` for the child. Empty, `null`, and `~` do not count as a selection. This makes the first-run key usable without editing `config.yaml`; an explicit config or environment choice always wins. The shell reads only the top-level `provider:` and `stt_model:` scalars from YAML; it leaves the rest to Python.

Setup reads the child's `.env` the same way `load_env` does, resolved against the child's working directory: `<child-cwd>/.env`, then `~/.config/verbix/.env`. Both are `~/.config/verbix/.env`. Process environment wins over `.env`. The shell reads only `VERBIX_PROVIDER`, `VERBIX_STT_MODEL`, `OPENROUTER_API_KEY`, and `VERBIX_API_KEY` from it.

Standard input is null. Standard output and standard error go to `<home>/.verbix/logs/voice-service.log`, truncated to 1 MB at each start, mode `0600`. Existing retention cleanup prunes it with the rest of `logs/` (`src/verbix/config.py:325-352`). The log writer redacts key-shaped tokens (`sk-...`, `api_key=...`, `Authorization: ...`, bearer tokens) before writing; a line it cannot redact is dropped.

Shell diagnostics go to `<home>/.verbix/logs/shell.log` with the same mode, truncation, redaction, and retention. State transitions and error text are logged; transcripts, captions, task results, and keys are not.

### 4.2 Readiness

The shell connects only after startup and remains connected:

1. Wait for the socket path or Windows endpoint file to exist.
2. Retry opening the connection and sending `{"id":"ping-<n>","action":"ping"}` every 250 ms. A stale file, or a socket that is bound but not yet accepting, is not a failed attempt.
3. Accept only a response with the matching id and `data.pong == true`.
4. Subscribe to the same connection for events.

Timeout: 30 seconds. Loading the GGUF model happens before the server starts (`src/verbix/cli.py:561-570`), so a slow disk must not be treated as a crash. If the child exits first, the supervisor handles it. If the timeout expires, stop the child, mark the attempt failed, and follow the restart policy.

The shell does not wait for the startup line on stdout. It is diagnostic text, not a readiness signal.

### 4.3 IPC usage

Requests:

| Action | Used by | Notes |
|---|---|---|
| `ping` | readiness and health | 2 s timeout |
| `toggle` | tray menu, global shortcut, fallback command | 60 s timeout; the response may wait for transcription |
| `status` | optional resync after a connection drop | text summary only; no new protocol |
| `cancel` | not used in v1 | reserved for existing CLI use |

`prompt` is not used. The panel is not a text client in v1.

Events consumed:

| IPC shape | Shell use |
|---|---|
| `runtime_event` / `floor_state` | tray state and panel floor label |
| `runtime_event` / `user_transcript` | final user transcript in the activity area |
| `text_delta` | append to the current assistant caption |
| `turn_complete` | finalize the caption, using `final_text` if no delta arrived |
| `runtime_event` / `task_registered` | add a task row in `running` |
| `runtime_event` / `task_completed` | row `completed`, result summary |
| `runtime_event` / `task_failed` | row `failed`, result summary |
| `runtime_event` / `task_cancelled` | row `cancelled` |
| `wake_event` | ignored; it duplicates `text_delta`/`turn_complete` |
| `thought_delta`, `tool_call_started`, `tool_call_finished` | ignored; not user-facing |

`runtime_event` carries `{"type": "<event>", "payload": {...}}` inside its `payload`; direct events (`text_delta`, `turn_complete`) carry their fields directly.

Malformed lines are logged without message contents and dropped. A request timeout, three consecutive malformed lines, EOF, or a write error disconnect the client. The supervisor decides whether to restart; either way, the panel clears its session view and the floor resets to `idle`.

### 4.4 Shutdown

Unix:

1. Send `SIGINT` to the child process. This is the CLI's existing clean-shutdown path (`src/verbix/cli.py:576-585`, `:728-737`).
2. Wait 5 seconds.
3. Send `SIGTERM` to the child's process group.
4. Wait 1 second.
5. Send `SIGKILL` to the process group.

The child is spawned in its own process group from the process-lifetime main thread. On Linux it is also marked with `PR_SET_PDEATHSIG(SIGTERM)` in `pre_exec`, so a shell crash cannot leave an orphan; do not spawn the service from a short-lived thread. macOS and Windows use their platform seam when implemented; Windows uses `Child::kill` in v1 because Windows is a compiled seam, not a shipped target.

Quitting also stops an in-flight STT download child. The shell does not leave either process behind.

## 5. Rust shape

Tauri types stay at the edges so process supervision, IPC framing, setup, and tray-state rules are testable without a display.

| Path | Owns |
|---|---|
| `src-tauri/src/main.rs` | process entry; calls `verbix_shell::run()` |
| `src-tauri/src/lib.rs` | Tauri builder, plugin order, command registration |
| `src-tauri/src/command.rs` | `--hidden`, `--show`, `--toggle`, `--version`, `--help` |
| `src-tauri/src/service.rs` | spawn, readiness, restart backoff, stop, child log |
| `src-tauri/src/ipc.rs` | JSONL framing, request ids, pending responses, event routing |
| `src-tauri/src/state.rs` | `ShellSnapshot`, tray-state precedence, task rows, caption buffer |
| `src-tauri/src/tray.rs` | icon, tooltip, menu, menu events |
| `src-tauri/src/panel.rs` | panel show/hide, close-to-hide, snapshot events |
| `src-tauri/src/setup.rs` | setup detection, key file, STT download, `shell.json` |
| `src-tauri/src/autostart.rs` | XDG desktop entry on Linux; unsupported stubs elsewhere |
| `src-tauri/src/platform.rs` | process-group/signal seam, Wayland detection, executable path |
| `src-tauri/panel/` | static `index.html`, `panel.css`, `panel.js` |
| `src-tauri/icons/` | app bundle icons and 32x32 tray PNGs |
| `src-tauri/tests/fake_voice_service.py` | headless IPC fixture for integration tests |

No `tauri-plugin-shell`; the child is a fixed `std::process::Command`. No `tauri-plugin-autostart`; its Linux path would point at the mounted AppImage binary instead of `$APPIMAGE`. No frontend framework and no npm: the panel uses `withGlobalTauri` and external static files.

## 6. Lifecycle and supervision

States:

| State | Meaning | Service |
|---|---|---|
| `SetupRequired` | first-run setup incomplete | not started |
| `Starting` | child spawned, waiting for IPC | running |
| `Connected` | ping succeeded | running |
| `Restarting` | unexpected exit, waiting to retry | not running |
| `Failed` | retry budget exhausted | not running |
| `Stopping` | quit requested | stopping |

Rules:

1. Spawn only when `shell.json` marks setup complete and the key and model checks pass. Slice 1 assumes setup complete; Slice 3 adds the gate.
2. Any exit while not `Stopping` is unexpected, including exit code 0.
3. Backoff: 1 s, 2 s, 5 s, 10 s, then 30 s.
4. After 5 failed attempts without a 60-second connected period, enter `Failed`. Do not retry automatically. Tray menu exposes **Restart voice service**.
5. A 60-second connected period resets the failure count.
6. On restart, clear panel tasks and captions and reset the floor to `idle`. Python keeps no history across sessions, so there is nothing to replay.
7. The voice service unlinks a stale socket before binding (`src/verbix/ipc/server.py:135-181`). The shell does not delete sockets.
8. Quit sets `Stopping` before signaling, so a normal child exit cannot trigger a restart.

A service that fails because the STT file was deleted is a setup failure, not a crash loop: before spawn, re-check setup and open the panel in `SetupRequired` if key or model is missing.

## 7. Tray

Tray state is computed, never set from multiple places:

1. `error` if the service is not connected and the state is not `Starting`; also for setup required.
2. `listening` or `speaking` from the last floor event.
3. `working` if any task is `registered` or `running` and the floor is `idle`.
4. `idle` otherwise.

`Starting` uses the idle icon with tooltip `Verbix — starting`. This avoids inventing a sixth tray state; `GLOSSARY.md` defines five.

Icon assets: `idle`, `listening`, `speaking`, `working`, `error`, 32x32 PNG with transparency, original artwork. Tooltip: `Verbix — <state>`.

Menu:

| Item | Behavior |
|---|---|
| Open Verbix | show and focus the panel |
| Toggle listening | send IPC `toggle`; disabled while not connected |
| Restart voice service | enabled in `Failed` or disconnected states except `SetupRequired` |
| Launch at login | check item; writes or removes the autostart entry |
| Setup… | opens the first-run setup panel for repair |
| Quit | stop the service and exit |

Left-click on the tray icon opens the panel. If a desktop does not deliver tray activation, the menu and second launch remain.

GNOME does not show StatusNotifierItem icons without the AppIndicator extension. The shell still runs. The panel opens by launching the AppImage again, and the global shortcut still toggles. First-run setup states this dependency once; v1 adds no second always-visible surface.

## 8. Panel

One panel, one window, hidden by default:

- 420x560 logical pixels, resizable, minimum 360x420.
- Normal decorated window. Not always-on-top, not click-through, not an overlay.
- Hidden from the taskbar (`skipTaskbar`) where the compositor honors it.
- Closing the window hides it; it does not quit. Escape hides it.
- First run opens it for setup. Normal launch stays hidden.
- A second launch, tray **Open Verbix**, or `--show` shows it.
- Static HTML, CSS, and one JS file. No network requests, no remote assets, no inline script.

Content:

1. Header: product name, tray state dot, Hide button.
2. Activity: floor label, latest final user transcript, current assistant caption.
   - `user_transcript` clears the caption and starts a new activity.
   - `text_delta` appends. If the previous caption was finalized, the first delta starts a new caption and replaces the old text.
   - `turn_complete` finalizes the caption and uses `final_text` when no delta arrived. A worker announcement therefore starts a new caption instead of running into the previous answer.
3. Tasks: newest 20 rows from this session. Each row shows id, short objective, task state, and a result summary capped at 280 characters. Completed, failed, and cancelled rows remain.
4. Setup: replaces the activity area while setup is incomplete.
5. Footer: shortcut hint or fallback command, and a service error with the log path and a Retry button.

No text input, no prompt box, no task cancel button, no history list. Use semantic buttons and labels, visible focus, readable body text, and a text label beside every state color; tray state is not carried by color alone.

### 8.1 Snapshot contract

Rust owns all derived state. The panel renders a complete snapshot; it does not keep a reducer or infer tray state.

```json
{
  "phase": "connected",
  "tray_state": "idle",
  "floor": "idle",
  "shortcut": {
    "accelerator": "Ctrl+Alt+Space",
    "fallback": null
  },
  "activity": {
    "transcript": "what time is it in tokyo",
    "caption": "It is 6:42 PM in Tokyo."
  },
  "tasks": [
    {
      "id": "task-1",
      "objective": "summarize this repo",
      "state": "completed",
      "result": "The repo is a voice agent with a foreground and workers."
    }
  ],
  "setup": {
    "required": false,
    "key_ready": true,
    "key_editable": true,
    "model_ready": true,
    "model_path": "/home/user/.verbix/voice/stt/models/nemotron-gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf",
    "model_downloadable": true,
    "autostart": true,
    "autostart_supported": true,
    "download": null,
    "error": null
  },
  "error": null
}
```

Snapshot rules:

- One event name: `shell://state`. Emit on change, at most once per 50 ms; also return the current snapshot from the `get_state` command on panel load.
- `phase` is the lifecycle state from Section 6 in `snake_case`: `setup_required`, `starting`, `connected`, `restarting`, `failed`, `stopping`.
- `task_registered` adds a row directly as `running`; the worker starts immediately and there is no separate `task_running` event. Displayed task states are `running`, `completed`, `failed`, `cancelled`.
- `autostart_supported` is false on macOS and Windows; the panel hides or disables the choice.
- `key_editable` is true when the key field can write the selected provider (`openrouter` or no selection); it is false when setup must show the `auth.json` path instead.
- `model_path` is the resolved authoritative or default path, or null. `model_downloadable` is false when a config or environment path is authoritative, true otherwise.
- `shortcut.accelerator` is null when the native grab is unavailable; `shortcut.fallback` is the command or null.
- `setup.download` is null or `{"running": true, "last_line": "<text>"}`.
- Cap transcript and caption at 16,000 characters; cap task rows at 20, task objectives at 200 characters, and task results at 280 characters.
- The snapshot never contains the provider key, file contents, or voice-service child output. Setup errors may include the downloader's last stderr line.
- Rendering uses `textContent`, never `innerHTML`. Transcripts and task results are untrusted text.
- CSP: `default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src ipc: http://ipc.localhost`. No remote origins.

Panel commands:

| Command | Purpose |
|---|---|
| `get_state` | full snapshot |
| `hide_panel` | hide the panel |
| `toggle` | send IPC toggle |
| `restart_service` | retry after `Failed` |
| `save_provider_key` | first-run/repair only |
| `start_model_download` | first-run/repair only |
| `set_autostart` | first-run/tray choice |
| `finish_setup` | write setup marker and start the service |

## 9. Toggle, shortcut, fallback

The toggle is one action: send IPC `toggle` and let Python decide start, stop, barge-in, or empty transcript. The shell does not track recording locally. Floor events drive tray state; the response action is checked only for errors.

Default accelerator: `Ctrl+Alt+Space`, fixed in v1.

- X11 and XWayland: register through `tauri-plugin-global-shortcut`. If registration fails, use the fallback.
- Wayland: the plugin cannot grab natively. Use the fallback regardless of registration result.
- Shortcut press while the service is starting or down: show the panel with the service state or error. Do not queue toggles.
- Shortcut press while connected: send `toggle`.

Fallback: a desktop-level shortcut bound to the shell's toggle command, managed by the user in desktop settings. The command shown is:

```
"<path to AppImage>" --toggle
```

The path is the resolved `$APPIMAGE` when running from an AppImage, otherwise `current_exe()`. When the fallback is active, first-run setup and the panel footer show this command with a Copy button. Copy uses `navigator.clipboard.writeText`; the command is also selectable text if the webview denies clipboard access. The shell does not write GNOME or KDE shortcut settings and does not use the XDG GlobalShortcuts portal (ADR-0006). If the AppImage moves, the fallback must be rebound; setup shows the current command again.

## 10. Single instance

`tauri-plugin-single-instance` is registered before any other plugin. One shell per user. A second launch forwards its arguments to the running instance and exits 0.

| Arguments | First launch | Already running |
|---|---|---|
| none | start hidden | show panel |
| `--hidden` | start hidden | do nothing |
| `--show` | show panel | show panel |
| `--toggle` | start hidden, then toggle after ready | send toggle |
| `--version` | print and exit before GUI | print and exit |
| `--help` | print usage and exit | print usage and exit |
| unknown | usage on stderr, exit 2 before GUI | usage on stderr, exit 2 before forwarding |

`--version` and `--help` are handled before Tauri starts so CI can smoke-test an AppImage without a display.

The forwarded `--toggle` goes through the same shell state and IPC client as a tray click. It never starts a second service. If setup is incomplete, any toggle request opens the setup panel instead of sending IPC.

## 11. Autostart

Linux entry: `~/.config/autostart/verbix.desktop`, mode `0644`.

```ini
[Desktop Entry]
Type=Application
Name=Verbix
Comment=Voice-operated desktop assistant
Exec="<path>" --hidden
Terminal=false
X-GNOME-Autostart-enabled=true
```

- `<path>` is `$APPIMAGE` when set, otherwise `current_exe()`. Never the mounted `/tmp/.mount_*` path.
- Quote the executable path; escape backslashes and quotes; double `%` to `%%` per the Desktop Entry spec.
- Enable replaces the file atomically. Disable removes only `verbix.desktop`.
- A moved AppImage leaves a stale entry. The tray check item and setup show the entry state; toggling off and on writes the current path (ADR-0005).
- No systemd unit, no package install, no stable copy.
- macOS and Windows: the autostart interface compiles and returns `Unsupported` in v1. The UI hides or disables the choice. No artifacts ship.

## 12. First-run setup

Shown when `~/.verbix/shell.json` lacks `setup_complete`, or when the key or STT model is missing. The panel opens automatically; this overrides `--hidden`, so an autostart launch with incomplete setup still shows the panel. The voice service does not spawn until the key and model are ready. Autostart is offered before Finish.

### 12.1 Provider key

- Password field, never echoed back. Accept non-empty, at most 4096 bytes, no newline.
- Write or update the `openrouter` entry in `~/.verbix/auth.json`:

  ```json
  {
    "openrouter": {
      "type": "api",
      "key": "<key>",
      "model": "openrouter/google/gemini-3.5-flash-lite",
      "base_url": "https://openrouter.ai/api/v1"
    }
  }
  ```

- Replace only `key`; preserve every other entry and field. For a new entry, use the values shown. Create `~/.verbix` as `0700` if absent. Write through a same-directory temporary file, mode `0600`, then rename. Refuse to follow a symlink at the target path.
- Do not edit `config.yaml`. At spawn, the shell adds `VERBIX_PROVIDER=openrouter` only when this entry has a non-empty key and no config or environment provider is set (Section 4.1).
- The step is ready when any of these holds:
  - `OPENROUTER_API_KEY` or `VERBIX_API_KEY` is in the process environment or the existing `~/.config/verbix/.env`.
  - A provider is selected (`VERBIX_PROVIDER` from the process environment or `.env` wins over `config.yaml`), and `auth.json` has a valid `type: api` entry with a non-empty `key` for it.
  - No provider is selected, and `auth.json` has a valid `openrouter` entry with a non-empty `key`.
- Do not rewrite a ready key.
- If the selected provider is not `openrouter` and has no key, first-run cannot write it. Show the `auth.json` path and stop; this surface only writes `openrouter`. If the selected provider is `openrouter`, the key field writes that entry.
- No network validation. The shell never speaks to a provider. A bad key surfaces as a service failure in the panel after start, with the child log path.

Credentials stay in the global state folder (`GLOSSARY.md`, `src/verbix/auth.py:18-20`).

### 12.2 Speech model

Resolution follows Python's precedence (`src/verbix/voice/transcribe_stt.py:100-145`):

1. A top-level `stt_model:` in `~/.verbix/config.yaml`, if present and non-empty. It is authoritative.
2. Otherwise, `VERBIX_STT_MODEL` from the process environment or `.env`, if present and non-empty. It is authoritative.
3. Otherwise, the first `*.gguf` file under, in order: `~/.verbix/models/**`, `~/.verbix/voice/stt/models/**`, `~/.verbix/voice/models/**`.

The shell scans only the top-level `stt_model:` scalar, as it does for `provider:`, and does not parse the rest of the YAML. Relative config or environment paths resolve against `~/.config/verbix`, the child's working directory. The voice root is pinned to `~/.verbix/voice` for the child (Section 4.1), so detection ignores `VERBIX_VOICE_DIR`. If the authoritative config or environment path is set but is not a file, the model step is not ready: show that path, do not run the downloader, and do not mark the step ready. Otherwise the step is ready when a `.gguf` file exists under one of the default roots.

Otherwise run the bundled copy of the existing downloader:

```
<resources>/voice/python/bin/python3 <resources>/voice/fetch_voice_models.py --root <home>/.verbix/voice
```

Same environment, process group, and parent-death setup as the service. Capture stdout and stderr; show the last non-empty line and a spinner. The download is about 496 MB and is SHA-256 verified by the script. Hide panel does not cancel. Quit kills it. On exit 0, re-check the model path. On any other exit, show the last stderr line and Retry. Do not mark the step ready.

Setup exposes **Re-download** when the ready model is under `~/.verbix/voice`; the script re-hashes that root and replaces only missing or mismatched files. For a model under `~/.verbix/models`, or an explicit config or environment path, show the path and hide Re-download; the script only provisions `~/.verbix/voice`.

### 12.3 Autostart

Checkbox reflects the live desktop file; absent or `X-GNOME-Autostart-enabled=false` means off. Changing it writes or removes the entry immediately. Finish accepts either choice. Autostart is never enabled without an explicit check.

### 12.4 Finish and repair

- Finish requires key and model ready. It writes `~/.verbix/shell.json` atomically, mode `0600`:

  ```json
  {"setup_complete": true}
  ```

- Then it starts the voice service and switches the panel to the main view.
- If `shell.json` is missing or invalid, treat setup as incomplete and overwrite on Finish. Do not delete user files.
- Tray **Setup…** reopens the same panel at any time. Existing steps are pre-marked; the key can be replaced; the model check is idempotent; autostart re-points. This is the first-run surface, not a general settings UI.
- Existing CLI users see setup once; their key and model are detected and pre-marked, except a custom `VERBIX_VOICE_DIR`, which the shell replaces with `~/.verbix/voice`.

## 13. Packaging and AppImage

Build prerequisites: the Rust toolchain, `cargo-tauri`, and `uv`; no Node or npm.

Build script responsibilities, in order:

1. `uv run scripts/fetch_voice_models.py --bundle-tts` to provision the verified TTS assets.
2. `uv build` to produce the wheel. `_build_backend.py` fails the build if a TTS file is missing or empty.
3. Provision a relocatable CPython 3.13 with `uv python install --install-dir`, normalize its directory to `<resources>/voice/python`, and verify `<resources>/voice/python/bin/python3` runs from that path.
4. `uv pip install --python <resources>/voice/python/bin/python3 --target <resources>/voice/site-packages dist/verbix-*.whl`.
5. Copy `scripts/fetch_voice_models.py` to `<resources>/voice/fetch_voice_models.py`, and copy `THIRD-PARTY-NOTICES.md`, `DISTRIBUTION-TERMS.md`, and `LICENSES/` to `<resources>/licenses/`.
6. Smoke-check `PYTHONPATH=<resources>/voice/site-packages <resources>/voice/python/bin/python3 -c "import verbix.cli"`.
7. `cargo tauri build --bundles appimage`.

Tauri configuration:

- `productName`: `Verbix`; identifier: `com.verbix.app`; version matches `pyproject.toml`.
- `frontendDist`: `panel`; no dev server, no before-build command.
- `bundle.targets`: `["appimage"]` only.
- Bundle `<resources>/voice` and `<resources>/licenses` as application resource directories.
- `withGlobalTauri`: true; minimal capability set; no shell, HTTP, or updater plugins.
- Icons and tray icons are original artwork.

Artifact: `src-tauri/target/release/bundle/appimage/Verbix_<version>_amd64.AppImage`.

Runtime prerequisites on the user's machine:

- WebKitGTK 4.1 and GTK 3, per ADR-0001's accepted dependency.
- FUSE 2 for AppImage, or `APPIMAGE_EXTRACT_AND_RUN=1`.
- `libportaudio2` for audio.
- `bubblewrap` for the worker `bash` tool. Without it, Python already omits `bash` and prints why (`README.md`, Platform support); the shell keeps running.
- AppIndicator/StatusNotifier support for the GNOME tray.

No system Python, `uv`, or repository checkout is needed at runtime. The AppImage runs from any path, including paths with spaces. Data stays in `~/.verbix`. The existing source workflow (`uv sync`, `uv run scripts/fetch_voice_models.py`) is unchanged. No signing, notarization, auto-update, deb/rpm/Flatpak, or store build.

## 14. macOS and Windows seams

All shared shell code must compile on Linux, macOS, and Windows. CI runs `cargo check` and unit tests on all three. Only Linux produces a bundle.

| Concern | Linux | macOS / Windows |
|---|---|---|
| Tray, panel, single instance | shared Tauri code | shared Tauri code |
| Global shortcut | plugin on X11/XWayland; desktop fallback on Wayland | plugin; no shipped artifact |
| Autostart | XDG desktop entry | interface stub returns `Unsupported`; UI disabled |
| IPC | Unix domain socket | loopback TCP via the existing endpoint file (`src/verbix/ipc/server.py:156-170`) |
| Process stop | process group, SIGINT then SIGTERM/SIGKILL | `Child::kill`; full tree handling deferred |
| Bundle | AppImage | none |

Platform-specific code lives behind `platform.rs` and `autostart.rs`. No `target_os` checks in shared state, tray, panel, IPC, or setup code. Python must not assume Linux; this spec adds no Python platform behavior.

## 15. Security, privacy, licensing

- Provider key: mode `0600`, never in argv, logs, snapshots, or the panel after save. Environment wins over the file, as today.
- Shell state and logs: mode `0600`. Child log is pruned by the existing retention policy.
- IPC: create `~/.local/share/verbix` as `0700` before spawn, and narrow an existing directory to `0700`, so the UDS is not reachable by other users. Do not change the protocol or socket path.
- Child invocation: fixed executable and arguments. No shell, no user text in the command line.
- Panel: no remote content, strict CSP, `textContent` only. No provider or tool logic in the panel.
- The shell logs state transitions and error text, not transcripts, captions, task results, or keys. Shell log lines pass the same key redaction as the child log. The child log may contain conversation; it is local, mode `0600`, and retention-pruned.
- AppImage resources are read-only. The shell writes data under `~/.verbix`; it creates `~/.config/verbix` as an empty runtime cwd and writes `~/.config/autostart/verbix.desktop` only when autostart is enabled.
- License review before the first release build:
  - Scan the full Cargo tree and the bundled Python wheels; record every dependency, direct and transitive.
  - Tauri 2, `tauri-plugin-single-instance`, `tauri-plugin-global-shortcut`: MIT/Apache-2.0; record in `THIRD-PARTY-NOTICES.md`.
  - `serde`, `serde_json`, `tokio`, `libc`: permissive; record.
  - Bundled CPython and its runtime libraries: PSF and permissive; record.
  - WebKitGTK, GTK, and AppIndicator libraries: verify dynamic-link and notice obligations for the AppImage. If unclear, mark `UNRESOLVED` and block release; do not assume allowed.
  - Existing voice assets stay recorded: Supertonic 3 TTS (BigScience OpenRAIL-M, `DISTRIBUTION-TERMS.md`, `LICENSES/`) and Nemotron 3.5 ASR (OpenMDW-1.1). The M1 voice preset's voice-likeness clearance must close before release; if not, mark `UNRESOLVED` and block release.
  - No new third-party model, voice, font, icon, or audio asset. Tray and panel icons are original work.
  - `THIRD-PARTY-NOTICES.md`, `DISTRIBUTION-TERMS.md`, and `LICENSES/` ship inside the AppImage resources.

## 16. Failure modes

| Failure | Shell behavior |
|---|---|
| STT model missing at launch | `SetupRequired`; panel opens; no child |
| Config or env STT path is not a file | setup shows the path; no download; no child |
| Provider key missing | `SetupRequired`; no child |
| Child exits before IPC is ready | retry with backoff; tray `error` |
| IPC connects but ping fails | close, stop child, retry |
| Child crashes repeatedly | 5 attempts, then `Failed`; **Restart voice service** |
| Socket file is stale | Python unlinks it; shell keeps retrying |
| IPC line malformed | drop line; 3 in a row reconnect |
| Toggle while disconnected | panel shows setup or service error; no queue |
| STT download fails | step stays open; last stderr line; Retry |
| `auth.json` invalid or unreadable | setup shows error; do not overwrite |
| Key file is a symlink | refuse to write; show error |
| Autostart entry stale after moving AppImage | tray check off/on rewrites current path |
| Shortcut refused or Wayland | show fallback command; tray still toggles |
| GNOME shows no tray icon | second launch opens panel; fallback shortcut toggles once bound |
| Quit while download or turn is running | kill child/download; exit; no orphan |
| Shell crashes | Linux parent-death signal stops the service; next launch starts clean |
| `shell.json` corrupt | treat setup as incomplete; overwrite on Finish |

## 17. Test plan

TDD per slice: failing test first, then minimal code, then green suite. The full Python suite and `ruff check` stay green because the Python side is unchanged.

Rust unit tests, no display:

- `ipc`: request ids, response matching, event routing, malformed lines, EOF, write failure.
- `service`: spawn a fake service, readiness timeout, restart on crash, no restart on quit, stop deadline, log truncation.
- `state`: tray precedence, task row caps, caption finalization, snapshot contains no secret.
- `setup`: auth JSON preserve/replace/atomic/permissions, symlink refusal, provider-selection fallback, config/env authoritative model paths, model detection, corrupt `shell.json`, download failure.
- `autostart`: quoted path, `%` escaping, enable/disable, stale path rewrite.
- `command`: argument parsing and `--version`/`--help` before GUI.

Integration, Linux:

- `tests/fake_voice_service.py` speaks the existing JSONL protocol on a temp socket: ping, toggle, `floor_state`, `text_delta`, `user_transcript`, task events. Rust connects, toggles, observes states, kills it, and asserts no restart.
- A fake `fetch_voice_models.py` writes a small `.gguf`; setup advances without the network.
- AppImage smoke: `APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a Verbix_<version>_amd64.AppImage --version`; launch a second instance and verify forwarding; kill and check `pgrep -af 'verbix.cli'` is empty.

Manual acceptance on Linux:

- GNOME Wayland (with and without AppIndicator), KDE Plasma Wayland, and X11.
- AppImage in a directory with spaces.
- Fresh home: first-run key, model download, autostart, hidden relaunch at login.
- Shortcut toggle and Wayland fallback.
- Voice turn: transcript, caption, speaking, idle.
- Worker task: running row, spoken announcement, completed row.
- Quit: tray disappears, child exits, no orphan; relaunch is single instance.

## 18. Slices

### Slice 1: walking skeleton

Prerequisite: `~/.verbix` already provisioned with the existing `uv sync` and `uv run scripts/fetch_voice_models.py`, including the STT model. First-run setup lands in Slice 3; Slice 1 treats setup as complete.

Build:

- `src-tauri` scaffold, static empty panel hidden.
- Tray icon with Open, Toggle, Quit.
- Build script producing the bundled Python runtime and AppImage.
- Spawn `<resources>/voice/python/bin/python3 -m verbix.cli --serve --voice`.
- IPC connect + ping, tray toggle via `toggle`, floor events to tray state.
- Quit stops the child; crash restarts with backoff.
- README GPUI section replaced with the Tauri/ADR-0001 pointer.

Acceptance:

1. Launch AppImage. No panel appears; tray icon appears within 2 s.
2. Voice service child exists; IPC connects and pings within 30 s.
3. Tray **Toggle listening** sends `toggle`; returned action is `recording`; tray changes to listening, then speaking, then idle with real events.
4. Tray **Quit** removes the tray, stops the child within 7 s, and leaves no `verbix.cli` process.
5. Killing the child makes the tray show error, then the shell respawns it and reconnects.
6. `cargo tauri build --bundles appimage` produces an AppImage that runs from a path with spaces.
7. `APPIMAGE_EXTRACT_AND_RUN=1 xvfb-run -a <AppImage> --version` prints the version and exits 0.

### Slice 2: panel

- Static panel and snapshot event.
- Transcript, streaming caption, task rows.
- Close hides; Escape hides; tray/second launch shows.
- Tray working state.

Acceptance: a real turn fills transcript and caption; a worker completion adds a completed row and is spoken; closing the panel leaves the service running.

### Slice 3: first-run setup and autostart

- Setup detection and `shell.json`.
- Key file write, STT download with progress/error, autostart choice.
- Tray **Launch at login** and **Setup…**.
- Service start gated on setup.

Acceptance: fresh home reaches a working voice service through the panel without a terminal; relaunch at login is hidden; moving the AppImage and toggling autostart repairs the entry.

### Slice 4: shortcut, single instance, release hardening

- Global shortcut and Wayland fallback display.
- Single-instance forwarding for all arguments.
- CI matrix, license notices, AppImage smoke test.
- Failure-mode polish.

Acceptance: full v1 checklist below.

## 19. v1 acceptance

- The AppImage runs from any path and writes state to `~/.verbix`, plus the autostart entry when chosen.
- First run reaches a ready provider key (detected or entered), has the STT model ready (detected or downloaded), offers autostart, and starts voice.
- Later launches start hidden in the tray and connect to the existing voice service over the existing IPC.
- The tray shows idle, listening, speaking, working, and error correctly.
- The global shortcut toggles; on Wayland the fallback command is shown and works.
- The panel shows floor state, final transcripts, live captions, and session task rows.
- Completed worker results are spoken in the existing voice and appear as completed rows.
- Quit stops the child and leaves no orphan. A second launch is forwarded to the running shell.
- macOS and Windows compile in CI; only the Linux AppImage ships.
- Python tests and `ruff check` stay green.
- License review is complete, with `THIRD-PARTY-NOTICES.md`, `DISTRIBUTION-TERMS.md`, and `LICENSES/` shipped, no `UNRESOLVED` dependency, and the M1 voice-likeness clearance closed.
