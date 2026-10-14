# Verbix Architecture Re-Audit, Cross-Platform Portability & Unification Spec

**Date**: 2026-10-14  
**Status**: Ready for Final Spec Approval  
**Target**: `HEAD` (`7a72adb`)

---

## 1. Ground-Truth Stack, Versions & Native Dependencies at `HEAD`

Verified directly from `pyproject.toml`, `uv.lock`, `.venv/lib/python3.13/site-packages/`, `scripts/fetch_voice_models.py`, `THIRD-PARTY-NOTICES.md`, and `src/verbix/cli.py`:

| Component | Actual Stack at `HEAD` | Repo Grounding (`file:line`) | Wheel / Platform Reality (`uv.lock` & `.venv`) |
|---|---|---|---|
| **Python & Build** | Python `>=3.13`, `uv_build>=0.12.19,<0.13.0` | `pyproject.toml:9, 31` | `sherpa-onnx` and `pywebrtc-audio` publish wheels for `cp313` and `cp314` only (no `cp315` wheels). Constrain `requires-python = ">=3.13,<3.15"`. |
| **Default LLM** | `openrouter/google/gemini-3.5-flash-lite` via `litellm>=1.75.5,!=1.82.7,!=1.82.8` | `src/verbix/cli.py:71, 208`, `pyproject.toml:12` | Pure Python (`py3-none-any.whl`). |
| **TTS Engine** | **Supertonic 3 INT8** (`csukuangfj2/sherpa-onnx-supertonic-3-tts-int8-2026-05-11`, `BigScience OpenRAIL-M`) via `sherpa-onnx>=1.12.0` (`SherpaTTS`), resampling 44.1 kHz to 24 kHz `f32` mono and trimming silence. | `scripts/fetch_voice_models.py:22-60`, `src/verbix/voice/sherpa_tts.py:74-182`, `THIRD-PARTY-NOTICES.md:14-20` | Audit #1 wrongly assumed Kokoro TTS. `sherpa-onnx==1.13.8` & `sherpa-onnx-core==1.13.8` (`uv.lock:1793-1840`) ship wheels for Linux (`x86_64`, `aarch64`), macOS (`arm64`, `x86_64`), and Windows (`amd64`, `arm64`). |
| **STT Engine (Active)** | **Nemotron 3.5 ASR streaming 0.6B GGUF** (`handy-computer/nemotron-3.5-asr-streaming-0.6b-gguf`, `nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf`, `OpenMDW-1.1`) via `transcribe-cpp>=0.4.0` (`TranscribeSTT`). | `scripts/fetch_voice_models.py:61-70`, `src/verbix/voice/transcribe_stt.py:168-239`, `src/verbix/cli.py:380` | `transcribe-cpp-native==0.4.0` (`uv.lock:1984-1994`) ships wheels for Linux (`x86_64`, `aarch64`), macOS (`arm64`, `x86_64`), and Windows (`amd64`). No `win_arm64` or `musllinux` wheels. |
| **STT Engine (Retired)** | **Canary 180M Flash INT8** (`SherpaSTT`), marked `RETIRED` in notices and no longer fetched by `fetch_voice_models.py`. | `src/verbix/voice/sherpa_stt.py:170-357`, `THIRD-PARTY-NOTICES.md:39-44` | Unwired in CLI; `STTSession` (`sherpa_stt.py:43-168`) is still imported by `tests/test_stt_session_contract.py` and `tests/test_incremental_stt.py`. |
| **VAD Segmenter** | **Silero VAD** (`silero_vad.onnx`, MIT) via `sherpa-onnx` (`SileroSegmenter`), 512-sample (32 ms) windows at 16 kHz. | `src/verbix/voice/sherpa_vad.py:13-18, 38-152`, `scripts/fetch_voice_models.py:71-82` | Uses `sherpa-onnx`. |
| **Echo Cancellation** | **WebRTC AudioProcessor** via `pywebrtc-audio>=0.2.0` (`EchoCanceller`), 10 ms (160-sample) frames at 16 kHz. | `src/verbix/voice/aec.py:10-135`, `uv.lock:1488-1510` | Ships wheels for Linux (`x86_64`, `aarch64`), macOS (`arm64`, `x86_64`), and Windows (`amd64`). No `win_arm64` wheels. |
| **Audio Capture & Sinks** | `sounddevice>=0.5.6` (`PTTCapture` + `CallbackAudioSink`) and legacy `SubprocessAudioSink` (`sinks.py`). | `src/verbix/voice/capture.py:51-183`, `src/verbix/voice/callback_sink.py:44-192`, `src/verbix/voice/sinks.py:12-180` | Verified in `.venv/.../sounddevice.py:63-92` & `uv.lock:1861-1874`: macOS and Windows wheels bundle `libportaudio.{dylib,dll}`; Linux uses `py3-none-any.whl` and requires system `libportaudio.so.2`. |
| **Search (`ripgrep`)** | `ripgrep>=15.1.0` used by `grep.py` and `glob.py`. | `pyproject.toml:19`, `uv.lock:1665-1672`, `src/verbix/tools/shared.py:122-130` | **Broken packaging**: Contradicts `THIRD-PARTY-NOTICES.md:62`. Only `macosx_11_0_arm64` and `manylinux_2_39_x86_64` wheels exist on PyPI; fails `uv sync` without Rust on Windows, Linux `aarch64`, macOS `x86_64`, and Linux with glibc < 2.39. |
| **Screenshots (`pillow`)** | `pillow>=12.3.0` (`PIL.ImageGrab.grab()`). | `src/verbix/tools/screenshot.py:10, 83`, `.venv/.../PIL/ImageGrab.py:33-153` | Full wheel coverage (`uv.lock:1158-1217`). Uses XCB/`gnome-screenshot`/`grim`/`spectacle` on Linux, `screencapture` on macOS, Win32 GDI on Windows. |

*Unverified items*: The raw `.wav` dump of the live `"He wits up."` turn is not in the repository (`--dump-mic-dir` is opt-in at `src/verbix/cli.py:157-161`).

---

## 2. Validated Findings & Architectural Critique

### 2.1 Confirmed & Corrected Findings from Audit #1

1. **Four-Way Agent Loop Fragmentation (`VoiceAgent`, `WorkerAgent`, `AnnouncerAgent`, `DefaultAgent`)**:
   - **Grounded at**: `src/verbix/agent/voice.py:83-447` (447 lines), `src/verbix/agent/worker.py:61-375` (375 lines), `src/verbix/agent/announcer.py:19-46` (46 lines), and `src/verbix/agent/agents/default.py:111-688` (688 lines).
   - **Correction to Audit #1**: In production (`src/verbix/cli.py:217, 230`), `build_runtime()` passes `LitellmModel` instances, and `adapt_provider()` (`src/verbix/runtime/foreground.py:335-336`) returns `LitellmModel` unchanged. `VoiceAgent` calls `LitellmModel.query_stream()` (`litellm_model.py:165-288`), whereas `WorkerAgent` calls synchronous `LitellmModel.query()` (`litellm_model.py:138-163`) via `asyncio.to_thread`. Now that `LitellmModel.query_stream()` enforces `stream_idle_timeout_s = 60.0` (`litellm_model.py:80, 176`), all three flavors can run on a single streaming `Agent` engine.

2. **Control Signal Leak (`TurnComplete(final_text="RepeatedToolCall")`) & `"role": "exit"` History Poisoning**:
   - **Grounded at**: `src/verbix/agent/voice.py:215-224, 415-417`, `src/verbix/voice/output.py:130-141`, `src/verbix/cli.py:325-335`, and `src/verbix/runtime/announcer.py:155-158`.
   - `VoiceAgent.step_stream()` stuffs machine exit tokens (`"LimitsExceeded"`, `"TimeExceeded"`, `"RepeatedToolCall"`) into `TurnComplete.final_text`. `SpeechOutput.emit()` synthesizes `event.final_text` aloud when no `TextDelta` preceded it; `CliPrinter.emit()` prints it to `stdout`; and `drain_pending_announcements()` appends it to `runtime.foreground.messages`.
   - Furthermore, `VoiceAgent.step_stream()` appends `{"role": "exit", "content": ...}` to `self.messages` (`voice.py:216, 222, 415`). Because `runtime.foreground.messages` persists across turns, `LitellmModel._prepare_messages_for_api()` (`litellm_model.py:133-136`) sends `"role": "exit"` to the provider API on the next turn, causing strict OpenAI-compatible providers to reject subsequent turns with `400 Bad Request`.

3. **Tight Tool Polling (`status` Called 3x in One Turn) & `Runtime.status()` Blind Spot**:
   - **Grounded at**: `src/verbix/agent/voice.py:211-423`, `src/verbix/agent/config/voice.yaml:33`, and `src/verbix/runtime/runtime.py:367-397`.
   - `VoiceAgent` has `step_limit: 0` and no per-turn cap on `status`, keeping `status` in `tool_defs` across every loop iteration of the same turn.
   - Additionally, when `Runtime.status(task_id=None, full=False)` (`runtime.py:367-397`) is called without a `task_id`, it **ignores `full=True`** and **never calls `task.get_status()`**—returning only `- task-1: '<objective>' (running for Xs, started at HH:MM:SS)` (`runtime.py:387`) without the worker's `"Recent actions"` summary (`src/verbix/runtime/worker.py:304-321`).

4. **STT Noise & False Delegation (`"He wits up."` -> `spawn`)**:
   - **Grounded at**: `src/verbix/agent/config/voice.yaml:17-27` and `src/verbix/tools/spawn.py:21-44`.
   - In `voice.yaml:27`, *"Ambiguous input gets brief clarification, never continuation"* sits under `# INTERRUPTIONS` rather than `# TOOLS`, and neither `voice.yaml` nor `SpawnTool`'s schema description states that speech-to-text garbles, greetings, or ambiguous phrases without a concrete user task request must never be delegated to `spawn` or turned into speculative log/workspace investigations.
   - **Dropped from Audit #1**: Audit #1's `_SPECULATIVE_OBJECTIVE_RE` regex filter in `SpawnTool.execute()` is dropped; pattern-matching English phrasing in a tool handler is brittle and violates *"models decide, runtime coordinates, tools act"*.

5. **Mandatory Sandboxing (`BashTool` & `sandbox.py`)**:
   - **Grounded at**: `src/verbix/tools/__init__.py:62`, `src/verbix/tools/bash.py:38-68`, `src/verbix/tools/sandbox.py:27-169`, and `src/verbix/agent/environments/local.py:1-213`.
   - `BashTool` currently defaults to `sandboxed=False`, calling `execute_unsandboxed()` (`sandbox.py:27-75`).
   - Even when `execute_sandboxed()` (`sandbox.py:77-169`) is used on Linux:
     1. It mounts `--proc /proc` **without `--unshare-pid`** (`sandbox.py:115-116`), exposing host `ps aux` processes.
     2. It passes `preexec_fn=os.setsid` (`sandbox.py:149`, unsafe in multithreaded asyncio processes compared to `start_new_session=True`) and passes no `env=` (`sandbox.py:144-150`), leaking `OPENROUTER_API_KEY`, `VERBIX_API_KEY`, and host `HOME` into the sandbox.
     3. It binds `workspace_path` read-write (`sandbox.py:121-123`) without masking `<workspace>/.env` or `<workspace>/.verbix/auth.json` (which `read`/`write`/`edit` block at `src/verbix/tools/shared.py:87-98`, but `bash` can `cat` directly).
   - **Dropped from Audit #1**: Audit #1's regex command filter (`validate_workspace_command`) is dropped; isolation must be enforced structurally by `bwrap` namespaces, masked secret mounts, and sanitized child `env`.

### 2.2 Additional Codebase Findings Missed by Audit #1

6. **`WorkerAgent` + `LitellmModel` Uses `FormatError` Exception-as-Control-Flow (`src/verbix/agent/worker.py:169-249`, `src/verbix/agent/models/litellm_model.py:147-155`, `src/verbix/agent/models/utils/actions_toolcall.py:48-62`)**:
   - When `WorkerAgent` finishes tool execution and outputs a plain-text conclusion, `LitellmModel._parse_actions()` calls `parse_toolcall_actions()`, which unconditionally raises `FormatError(assistant_msg, user_msg)` (`actions_toolcall.py:61`). `WorkerAgent.step()` catches `FormatError` (`worker.py:176-189`), checks `if self._tools_executed > 0`, and treats that exception as normal task completion, while `worker.py:228-248` duplicates the same logic for non-raising models (`ProviderStreamAdapter`).

7. **`LocalEnvironment` Zombie Leak & Subprocess Orphan Bug (`src/verbix/runtime/workers.py:121-125, 180`, `src/verbix/runtime/runtime.py:454-461`, `src/verbix/agent/environments/local.py:26-113`)**:
   - `spawn_worker()` creates a `LocalEnvironment` per task and appends it to `runtime._environments` so `runtime.aclose()` can call `env.kill_all()`. Because `WorkerAgent` runs `BashTool` via `tools/sandbox.py` instead of `self.env`, `runtime.aclose()` kills **zero** running `BashTool` subprocesses, and `LocalEnvironment.get_template_vars()` (`local.py:111-112`) merges `dict(os.environ)` (including all `*_API_KEY` secrets) into `WorkerAgent.get_template_vars()`.

8. **Stale `finish` Tool Remnants & Unmarked Announcer Ledger (`src/verbix/runtime/foreground.py:88-89`, `src/verbix/runtime/worker.py:144-207`, `src/verbix/runtime/announcer.py:135-205`, `src/verbix/runtime/runtime.py:433-453`)**:
   - `DEFAULT_WORKER_PROMPT` (`foreground.py:88-89`) still instructs the worker to `"call the finish tool"` (removed in `4cc3ccc`).
   - `WorkerTask._run()` (`runtime/worker.py:144-207`) retains a dead 64-line `else:` branch polling `_is_finish_tool_message` for 30 rounds.
   - `Runtime._announce_ids()`, `Runtime._speak_batch()`, and `speak_batch()` (`runtime.py:433-453`, `announcer.py:164-205`) are dead duplicates of `drain_pending_announcements()`, while `drain_pending_announcements()` (`announcer.py:135-162`) never calls `runtime.ledger.mark_heard(task_id)`.

9. **Audio Pipeline, AEC & Continuous Voice Loop Issues (`src/verbix/voice/callback_sink.py:69-120`, `src/verbix/voice/output.py:460, 478`, `src/verbix/voice/pipeline.py:381-407`, `src/verbix/voice/transcribe_stt.py:112-165`)**:
   - **Missing `blocksize` on `CallbackAudioSink` (`callback_sink.py:69-75`)**: Unlike `PTTCapture` (`capture.py:116`, `blocksize = sample_rate // 50` = 20 ms), `CallbackAudioSink` omits `blocksize`, allowing variable 50–100 ms host buffers that degrade barge-in fade-out latency.
   - **Per-sample Python `float` boxing in `CallbackAudioSink` (`callback_sink.py:86-120`)**: Unpacks bytes into a `deque[float]` (`24,000` Python floats/sec) and pops them in a list comprehension inside the real-time PortAudio C callback under `self._lock`.
   - **AEC Reference Premature Enqueue (`output.py:460, 478`, `callback_sink.py:143-156`, `aec.py:107-108`)**: `SpeechOutput._feed()` feeds entire sentences to `EchoCanceller.feed_reference()` at enqueue time (`<1 ms`) rather than as `CallbackAudioSink` dequeues frames to the DAC; any inter-sentence underrun pads `_ref_buffer` with zeros and shifts echo cancellation alignment.
   - **Continuous Loop `_live_stt` Pollution (`pipeline.py:381-407`)**: `_run_continuous_loop()` calls `await self._setup_turn_listening()` at line 382 *before* `execute_voice_turn()` speaks. During playback, `feed_barge_in()` feeds AEC residual audio into `self._segmenter`. When `execute_voice_turn()` completes without barge-in, `finally:` (`pipeline.py:406`: `if self._is_continuous_active and self._live_stt is None:`) skips resetting `_live_stt`, leaving a dirty segmenter for the next user turn.
   - **Hypothesis — `TranscribeSession` Unbounded Pre-Speech Buffer (`transcribe_stt.py:112-165`)**: `TranscribeSession.feed()` appends every raw chunk (`self._buf.extend(chunk)`) from the start of listening while discarding the speech spans returned by `self._segmenter.pop_segments()` (`transcribe_stt.py:123`). We will test whether accumulating committed VAD spans (`pop_segments()` + `flush()`) reproduces and fixes pre-speech noise vulnerability without regressing single-pass transcription accuracy.

---

## 3. Platform Support Tiers & Audio/Sandbox Constraints

1. **Platform Tiers**:
   - **Supported Tier — Linux (`x86_64` & `aarch64`, Python 3.13–3.14)**: Full support (Text REPL, Voice PTT & Continuous + AEC + Barge-in, IPC, all 9 worker tools including `bash` sandboxed via `bwrap`). Requires system packages `bubblewrap` (`bwrap`) and `libportaudio2` (`portaudio`). If `bwrap` is missing or blocked by kernel unprivileged userns policy, `bash` is omitted from worker tools and the user is told why at startup.
   - **Experimental Tier — Windows (`amd64`) & macOS (`arm64` & `x86_64`)**: Documented in `README.md` as experimental and untested by the maintainer. No `bash` tool (no Windows or macOS sandbox backend is built; `probe_sandbox()` returns `available=False`, `bash` is omitted from worker tools, and startup prints why). Workers use the 8 cross-platform tools (`read`, `write`, `edit`, `grep`, `glob`, `web_search`, `web_fetch`, `screenshot`).
   - **Unsupported**: Windows `arm64`, `musllinux`, Python 3.15+.
2. **Single `sounddevice` Audio Path**:
   - `SubprocessAudioSink` (`src/verbix/voice/sinks.py`), `select_sink()`, and `--audio-sink` are removed. `CallbackAudioSink` is the sole hardware `AudioSink` across all platforms.
3. **Mandatory Sandboxing With Capability-Driven Tool Description**:
   - `execute_unsandboxed()` is deleted and `BashTool` has no `sandboxed` toggle—there is no code path in Verbix that runs worker shell commands directly on the host.
   - `probe_sandbox()` returns a `SandboxStatus` object reporting `available: bool`, `backend: str | None`, `reason: str`, and `capabilities_description: str` (e.g., for `bwrap`: workspace and `/tmp` read-write, system paths read-only, isolated PID namespace, `.env`/`auth.json` masked, network disabled or enabled per `allow_network`). `BashTool.description` is dynamically constructed from `SandboxStatus.capabilities_description` so the model knows the exact sandbox boundaries.

---

## 4. Unified Implementation Sequence

Every bug-fix step begins by writing a failing test that reproduces the issue against current code before applying the fix. Any finding that fails to reproduce in a test is dropped.

### Step 1: CI Matrix (Linux, Windows, macOS) & Python Version Bound
- **Files**: `.github/workflows/ci.yml`, `pyproject.toml`.
- **Changes**:
  - Constrain `requires-python = ">=3.13,<3.15"` in `pyproject.toml`.
  - Add `.github/workflows/ci.yml` running `uv sync`, `uv run ruff check src tests`, and `uv run pytest -q` on Python 3.13 across `ubuntu-latest`, `macos-latest`, and `windows-latest`.
  - In the `ubuntu-latest` job, install `bubblewrap` and `libportaudio2` and run `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` so `bwrap` namespace tests execute cleanly under Ubuntu 24.04.
  - Set `continue-on-error: true` on the `macos-latest` and `windows-latest` matrix entries initially so they run informationally until Step 5 lands.
- **Verification**: Validate workflow YAML and run `uv run pytest -q` + `uv run ruff check src tests`.

### Step 2: Drop `ripgrep` PyPI Dependency & Add `pathspec`-Backed Stdlib Search Fallback
- **Files**: `pyproject.toml`, `uv.lock`, `THIRD-PARTY-NOTICES.md`, `src/verbix/tools/shared.py`, `src/verbix/tools/grep.py`, `src/verbix/tools/glob.py`, `tests/test_tools_grep_glob.py`.
- **Changes**:
  - Remove `ripgrep>=15.1.0` from `pyproject.toml` and `THIRD-PARTY-NOTICES.md`; add pure-Python `pathspec>=0.12.1` (`py3-none-any.whl`, license: `MPL-2.0`, recorded in `THIRD-PARTY-NOTICES.md`) and update `uv.lock`.
  - In `GrepTool` and `GlobTool`, pass `--path-separator=/` whenever invoking a system `rg` binary found via `shutil.which("rg")`.
  - When `rg` is not on `PATH`, fall back to a shared pure-Python workspace file iterator `iter_workspace_files(target: Path, workspace: Path) -> list[Path]` in `src/verbix/tools/shared.py` (no host-side `git` subprocess):
    - **How `.gitignore` is handled**: Walk `target` with `os.walk` (pruning `.git`, `.verbix`, `.venv`, `node_modules`, `__pycache__` in-place) and match relative paths against `.gitignore` / `.ignore` files compiled with `pathspec.GitIgnoreSpec`.
    - In `GrepTool`'s fallback, skip binary files (any file containing `\x00` in its first 4096 bytes) and match lines using `re.compile(pattern, flags=0 if case_sensitive else re.IGNORECASE)`, returning `rel_posix_path:line_no:line_text` identical to `rg --line-number --no-heading`.
- **Verification**: Failing test first with `find_ripgrep_binary` monkeypatched to `None`, verifying `.gitignore` exclusions via `pathspec`, binary file skipping, case sensitivity, and glob filtering in both `GrepTool` and `GlobTool`; then green `uv run pytest tests/test_tools_grep_glob.py -q`.

### Step 3: Audio Path Consolidation (Three Sub-Steps, Each Behind Its Own Test)
- **Step 3a — Delete `SubprocessAudioSink`, Fail Fast on Missing PortAudio & Set 20 ms `blocksize`**:
  - *Test*: Failing tests asserting: (1) `CallbackAudioSink` passes `blocksize=sample_rate // 50` (`480` at 24 kHz) to `sd.OutputStream`; (2) when `sounddevice` fails to import or PortAudio fails to open, `CallbackAudioSink` / `_build_voice_pipeline()` fails fast with a `RuntimeError` containing a clear install hint (`libportaudio2` / `portaudio`); (3) `--audio-sink` is removed from `build_parser()`.
  - *Changes*: Delete `src/verbix/voice/sinks.py`, remove `select_sink()` from `src/verbix/voice/callback_sink.py`, remove `--audio-sink` from `src/verbix/cli.py`, wire `CallbackAudioSink` directly in `_build_voice_pipeline()` with the clear PortAudio install hint on failure, and update `tests/test_callback_sink.py`, `tests/test_speech_output.py`, `tests/test_config_scopes.py`, and `tests/test_voice_loop_barge.py`.
- **Step 3b — Contiguous Float32 Buffer in `CallbackAudioSink`**:
  - *Test*: Failing test verifying `CallbackAudioSink` stores pending samples in a contiguous `np.float32` buffer (no `deque` of Python `float` objects) while preserving exact `played_samples`, 10 ms fade-out on `stop()`, and short-tail linear ramp to zero.
  - *Changes*: Rewrite `CallbackAudioSink._pending` in `src/verbix/voice/callback_sink.py` to use a contiguous `np.ndarray` (`dtype=np.float32`) with read-offset slicing inside `_callback`.
- **Step 3c — AEC Reference Realignment**:
  - *Test*: Failing test reproducing AEC reference lead/drift when multiple sentences are enqueued into `CallbackAudioSink` ahead of DAC playback (verifying that `on_played` hands frames off via a lock-free queue and background thread as `_callback` actually dequeues them to the DAC, with zero logging or heavy work in `_callback`, stops feeding immediately on `interrupt()`/`stop()`, and configures `EchoCanceller(stream_delay_ms=20)`).
  - *Changes*: Move `on_audio` invocation from `SpeechOutput._feed()` enqueue time into `CallbackAudioSink` via a `queue.SimpleQueue[bytes | None]` and a background daemon thread so `_callback` only calls `put_nowait`, and pass `stream_delay_ms=20` to `EchoCanceller`.
- **Verification**: `uv run pytest tests/test_callback_sink.py tests/test_speech_output.py tests/test_aec_barge_in.py tests/test_voice_loop_barge.py -q`.

### Step 4: Mandatory Sandbox Seam (`bwrap` Only), Capability Reporting & Startup `bash` Gating
- **Files**: `src/verbix/tools/sandbox.py`, `src/verbix/tools/bash.py`, `src/verbix/tools/__init__.py`, `src/verbix/runtime/workers.py`, `src/verbix/runtime/runtime.py`, `src/verbix/cli.py`, `tests/test_tools.py`, `tests/test_worker_tools.py`.
- **Changes**:
  - Write failing tests reproducing: (a) `execute_sandboxed` leaking host `*_API_KEY` env vars and `<workspace>/.env` / `<workspace>/.verbix/auth.json` contents; (b) `BashTool` description not reflecting sandbox capabilities; (c) worker tool registry still offering `bash` when `probe_sandbox()` is unavailable.
  - Delete `execute_unsandboxed()` from `src/verbix/tools/sandbox.py` and remove the `sandboxed` parameter from `BashTool.__init__()`.
  - Add `SandboxStatus(available: bool, backend: str | None, reason: str, capabilities: str)` and `probe_sandbox(allow_network: bool = False) -> SandboxStatus` in `src/verbix/tools/sandbox.py`:
    - On Linux: checks `shutil.which("bwrap")` and runs a cached smoke probe (`bwrap --ro-bind / / -- true`). Reports capabilities: workspace and `/tmp` read-write, system paths read-only, isolated PID namespace (`--unshare-pid`), masked `.env`/`auth.json`, and network policy (`isolated` vs `enabled`).
    - On non-Linux (or failed probe): returns `SandboxStatus(available=False, backend=None, reason=..., capabilities="")`.
  - Build `BashTool.description` dynamically from `probe_sandbox(allow_network=self.allow_network).capabilities`.
  - Harden `execute_sandboxed()` in `src/verbix/tools/sandbox.py`:
    - Use `start_new_session=True` instead of `preexec_fn=os.setsid`.
    - Add `--unshare-pid` to `bwrap`.
    - Mask `<workspace>/.env` and `<workspace>/.verbix/auth.json` with `--ro-bind /dev/null <path>` when present.
    - Pass an allowlist-built child `env` (forwarding only `PATH`, `LANG`, `TERM` from `os.environ` when present, plus `HOME=<workspace>/.verbix/home`, `TMPDIR=/tmp`, `HISTFILE=/dev/null`, `PAGER=cat`, `MANPAGER=cat`, `PIP_PROGRESS_BAR=off`, `TQDM_DISABLE=1`) rather than a denylist of secret names.
    - Track active subprocess `pgid`s in `sandbox.py` (`kill_active_processes()`) and call it from `runtime.aclose()`.
  - When `not probe_sandbox().available`:
    - `build_runtime()` (`cli.py`) logs a one-line notice to `stderr`: `[verbix] Note: 'bash' tool disabled ({status.reason}).`
    - `spawn_worker()` / `build_worker_registry()` (`runtime/workers.py`) omits `"bash"` from worker tools.
    - `Runtime._build_grounded_prompt()` (`runtime/runtime.py`) omits `Shell: bash.`.
- **Verification**: `uv run pytest tests/test_tools.py tests/test_worker_tools.py tests/test_runtime_worker.py -q`.

### Step 5: Independent Bug Fixes (Each With Its Own Reproduction Test First)
- **Step 5a — Typed `TurnComplete`, No `"role": "exit"`, and Spoken Fallback (`core/events.py`, `voice/output.py`, `cli.py`, `ipc/server.py`, `agent/voice.py`)**:
  - *Tests*:
    1. Reproduce `SpeechOutput.emit(TurnComplete(final_text="RepeatedToolCall"))` synthesizing `"RepeatedToolCall"` and `CliPrinter.emit()` printing it.
    2. Reproduce `"role": "exit"` poisoning `VoiceAgent.messages` by asserting directly on the payload built by `LitellmModel._prepare_messages_for_api(agent.messages)` after a limit/repeat exit.
    3. Reproduce empty `_delivered_text` on a foreground breaker producing silence instead of the spoken fallback.
  - *Changes*: Add `exit_reason` and `is_error` to `TurnComplete`, and make `spoken_error_fallback: str` per-flavor (`"Sorry — I ran into a problem. Could you say that again?"` on `VoiceConfig`, `""` on `AnnouncerConfig` so the announcer never says `"Could you say that again?"` or writes it into `foreground.messages`). Remove `"role": "exit"` insertions; when a breaker trips with `_delivered_text == ""`, emit `spoken_error_fallback` only if non-empty. Ensure `SpeechOutput.emit()` and `CliPrinter.emit()` never speak or print machine exit codes.
- **Step 5b — Per-Turn Tool Gating in `VoiceAgent` & `Runtime.status()` Fidelity (`agent/voice.py`, `agent/config/voice.yaml`, `runtime/runtime.py`)**:
  - *Tests*:
    1. Reproduce `VoiceAgent` calling `status` 3 times in a single turn because `status` remains in `tool_defs` after the first call.
    2. Reproduce `runtime.status()` (with `task_id=None`) omitting active task `"Recent actions"` and ignoring `full=True`.
  - *Changes*:
    1. Add `max_calls_per_tool_per_turn: dict[str, int]` to `VoiceConfig` (`{"status": 1, "cancel": 1}` in `voice.yaml`) and enforce it in `VoiceAgent.step_stream()` (pruning exhausted tools from `get_tool_definitions()` for the rest of the turn, and canonicalizing `repeat_history` signatures with `json.dumps(args, sort_keys=True)`), completely independent of Step 7's engine unification.
    2. Update `Runtime.status()` (`src/verbix/runtime/runtime.py:367-397`) to include active tasks' recent action lines (and honor `full=True` when `task_id` is omitted).
- **Step 5c — Announcer `mark_heard()`, Dead Code Removal & Prompt/Tool Hardening (`runtime/announcer.py`, `runtime/runtime.py`, `runtime/foreground.py`, `agent/config/voice.yaml`, `tools/spawn.py`)**:
  - *Tests*: Reproduce `drain_pending_announcements()` leaving `ledger.get(task_id).delivery == "pending"` and `DEFAULT_WORKER_PROMPT` referencing `"finish"`.
  - *Changes*: Call `runtime.ledger.mark_heard(task_id)` in `drain_pending_announcements()`, delete dead `speak_batch()` / `Runtime._speak_batch()` / `Runtime._announce_ids()`, remove `"call the finish tool"` from `DEFAULT_WORKER_PROMPT`, and add STT-noise / no-speculative-delegation rules to `# TOOLS` in `voice.yaml` and `SpawnTool` description.
- **Step 5d — Console `drain_stdin` Portability & Platform Tier Docs (`cli.py`, `README.md`)**:
  - *Test*: Reproduce `drain_stdin()` calling `select.select` on Windows (`os.name == "nt"`).
  - *Changes*: Use `msvcrt.kbhit()` / `msvcrt.getwch()` on Windows in `drain_stdin()` (deferring Windows IPC changes to keep IPC minimal and POSIX UDS untouched), and document the platform support tiers in `README.md` (Linux supported; Windows and macOS experimental, untested, `bash` disabled, IPC `--serve`/`--toggle` POSIX-only).
- **Verification**: `uv run pytest tests/test_events.py tests/test_speech_output.py tests/test_voice_agent.py tests/test_runtime_e2e.py tests/test_announcer_config.py tests/test_ipc_and_cli.py tests/test_prompt_invariants.py -q`.

### Step 6: Voice-Loop & `TranscribeSession` Fixes (`voice/pipeline.py`, `voice/transcribe_stt.py`)
- **Step 6a — Continuous Loop `_live_stt` Reset After Playback (`src/verbix/voice/pipeline.py:381-407`)**:
  - *Test*: Reproduce `_run_continuous_loop()` feeding playback audio into `self._live_stt.feed_barge_in()` during `execute_voice_turn()` and failing to reset `self._live_stt` in `finally:` when playback finishes without barge-in.
  - *Changes*: Ensure `_run_continuous_loop()` cancels the old `_live_stt` session before opening the new one after `execute_voice_turn()` finishes (when no barge-in occurred), and seeds the new session with `self._mic_lookback` so user speech starting at the tail end of playback is never dropped.
- **Step 6b — Hypothesis Test: `TranscribeSession` Span-Only Accumulation (`src/verbix/voice/transcribe_stt.py:112-165`)**:
  - *Hypothesis & Test*: Write a test feeding pre-speech noise/silence chunks followed by a VAD-detected speech span into `TranscribeSession`. Check whether `finalize()` passes the unsegmented pre-speech buffer into `self._stt.transcribe()`.
  - *Changes*: If reproduced and accumulating VAD spans (`pop_segments()` + `flush()`) passes all existing `tests/test_transcribe_stt.py` and `tests/test_stt_session_contract.py` tests, update `TranscribeSession` to transcribe committed VAD spans; if not reproduced or if it breaks single-pass transcription contracts, drop 6b.
- **Verification**: `uv run pytest tests/test_conversational_voice_loop.py tests/test_audio_loop_states.py tests/test_transcribe_stt.py tests/test_stt_session_contract.py -q`.

### Go/No-Go Gate Before Step 7: Reassess Engine Unification vs. Direct Legacy Deletion
- **Gate Criteria**: After Steps 1–6 are complete and green, measure the remaining duplication between `VoiceAgent` (`src/verbix/agent/voice.py`) and `WorkerAgent` (`src/verbix/agent/worker.py`) once `DefaultAgent` (Step 8) is slated for removal:
  - If merging `VoiceAgent` and `WorkerAgent` into `src/verbix/agent/engine.py` reduces net complexity without risking regressions in worker tool enforcement or voice pre-tool buffering, proceed with **Step 7** followed by **Step 8**.
  - If `VoiceAgent` and `WorkerAgent` are cleaner kept separate after Step 5's bug fixes, **skip Step 7** and go straight to **Step 8** (deleting `DefaultAgent`, `LocalEnvironment`, `FinishTool`, and `tests/test_miniswe_*.py`).
  - Pause at this gate to report the diff/line metrics and confirm Go/No-Go with the engineer before executing Step 7.

### Step 7 (Conditional on Go/No-Go Gate): Engine Unification ("One Engine, Three Flavors")
- **Files**: `src/verbix/agent/engine.py`, `src/verbix/agent/config/voice.yaml`, `src/verbix/agent/config/worker.yaml`, `src/verbix/agent/config/announcer.yaml`, `src/verbix/runtime/foreground.py`, `src/verbix/runtime/announcer.py`, `src/verbix/runtime/workers.py`, `src/verbix/runtime/worker.py`.
- **Tests**: Failing tests reproducing: (a) Worker not streaming live `TextDelta` events during `step_stream()`; (b) Worker requiring `FormatError` exception-as-control-flow to complete on plain text after tool execution.
- **Changes**:
  - Implement unified `Agent` and `AgentConfig` in `src/verbix/agent/engine.py` driving Foreground, Announcer, and Worker via `model.query_stream()`.
  - Flavor configuration knobs:
    - `speech_chunking: bool` (`True` for Foreground/Announcer; `False` for Worker).
    - `turn_scoped_limits: bool` (`True` for Foreground/Announcer; `False` for Worker).
    - `require_tool_before_finish: bool` (`True` for Worker; `False` for Foreground/Announcer).
    - `max_calls_per_tool_per_turn: dict[str, int]` (`{"status": 1, "cancel": 1}` in `voice.yaml`; `{}` for Worker/Announcer).
    - Canonicalized `repeat_limit: int = 3` (`json.dumps(args, sort_keys=True)`), preserving the 3-strike worker error observation injection (`bae14b1`) before hard abort on the 4th strike.
  - Collapse `WorkerTask._run()` (`src/verbix/runtime/worker.py:123-208`) to stream directly from `self.session.step_stream(self.description)`.
  - Keep compatibility re-exports (`VoiceAgent`, `WorkerAgent`, `AnnouncerAgent`, `DefaultAgent`) in place during Step 7 so all existing test files remain green prior to Step 8.
- **Verification**: `uv run pytest -q` and `uv run ruff check src tests`.

### Step 8: Dedicated Commit — Delete Legacy Mini-SWE-Agent Surface
- **Files**: `src/verbix/agent/`, `src/verbix/tools/`, `src/verbix/runtime/`, `tests/`.
- **Changes**:
  - Audit `tests/test_miniswe_agent_turn.py`, `tests/test_miniswe_environment.py`, `tests/test_miniswe_tools_and_model.py`, and `tests/test_local_environment.py`. Port any behavior that `VoiceAgent` / `WorkerAgent` (or unified `Agent`) and `sandbox.py` must retain (e.g., multimodal observation expansion, cost tracking, turn-boundary recovery, and the `bae14b1` worker circuit-breaker tests) into `tests/test_worker_agent.py`, `tests/test_voice_agent.py`, or `tests/test_tools.py`.
  - Delete:
    - `src/verbix/agent/agents/default.py` and `src/verbix/agent/agents/`
    - `src/verbix/agent/environments/local.py` and `src/verbix/agent/environments/`
    - `src/verbix/tools/finish.py` (and `"finish"` from `TOOL_CATALOG` in `src/verbix/tools/__init__.py`)
    - Dead `else:` branch in `WorkerTask._run()` (`src/verbix/runtime/worker.py:144-207`) polling `_is_finish_tool_message`
    - Obsolete `tests/test_miniswe_*.py` and `tests/test_local_environment.py`
    - *(If Step 7 was executed, also delete `voice.py`, `worker.py`, `announcer.py`, `actions_toolcall.py`, `retry.py`, `LitellmModel.query()`, and `ProviderStreamAdapter.query()`).*
- **Verification**: `uv run pytest -q` and `uv run ruff check src tests`.

### Step 9: Dedicated Commit — Retire `src/verbix/voice/sherpa_stt.py` (Canary 180M Flash)
- **Files**: `src/verbix/voice/sherpa_stt.py`, `src/verbix/voice/transcribe_stt.py`, `tests/test_stt_session_contract.py`, `tests/test_incremental_stt.py`, `tests/test_sherpa_stt.py`, `README.md`.
- **Changes**:
  - Move `STTSession` and `_Segment` (imported by `tests/test_stt_session_contract.py` and `tests/test_incremental_stt.py`) from `src/verbix/voice/sherpa_stt.py` into `src/verbix/voice/transcribe_stt.py`.
  - Update imports in `tests/test_stt_session_contract.py` and `tests/test_incremental_stt.py`, and update stale `SherpaSTT` / Canary references in `README.md`.
  - Delete `src/verbix/voice/sherpa_stt.py` and `tests/test_sherpa_stt.py`.
- **Verification**: `uv run pytest -q` and `uv run ruff check src tests`.
