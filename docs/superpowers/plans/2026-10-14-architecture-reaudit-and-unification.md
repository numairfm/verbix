# Verbix Architecture Re-Audit, Portability & Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate control-signal audio leaks, unbounded/unsandboxed worker execution, tight `status` polling, broken native wheel dependencies, and legacy Mini-SWE-Agent/Canary drift while establishing a single `sounddevice` audio path and a 3-OS CI matrix.

**Architecture:** Fix independent bugs with reproduction tests first (search fallback, single `sounddevice` sink with contiguous float32 buffering and DAC-aligned AEC reference, mandatory `bwrap` sandbox with capability reporting and startup tool gating, typed `TurnComplete` without `"role": "exit"`, per-turn `status`/`cancel` caps in `VoiceAgent`, and continuous-loop STT session reset), pause at a Go/No-Go gate before optional engine unification, and finish with dedicated commits deleting the legacy Mini-SWE-Agent and retired Canary STT surfaces.

**Tech Stack:** Python `>=3.13,<3.15`, `pydantic>=2.12`, `litellm>=1.75.5`, `sherpa-onnx>=1.12.0` (Supertonic 3 INT8 TTS + Silero VAD), `transcribe-cpp>=0.4.0` (Nemotron 3.5 ASR GGUF), `pywebrtc-audio>=0.2.0`, `sounddevice>=0.5.6`, `pathspec>=0.12.1`, Linux `bubblewrap` (`bwrap`).

**Spec:** `docs/superpowers/specs/2026-10-14-architecture-reaudit-and-unification-design.md`

## Global Constraints

- **Python floor & ceiling:** `requires-python = ">=3.13,<3.15"` in `pyproject.toml`.
- **Platform tiers:** Linux (`x86_64`, `aarch64`) is the only supported tier; Windows (`amd64`) and macOS (`arm64`, `x86_64`) are experimental (`bash` disabled, untested, documented as such).
- **Mandatory sandboxing:** No code path may run worker shell commands directly on the host (`execute_unsandboxed` and `LocalEnvironment` deleted; no `sandboxed=False` flag on `BashTool`). Where `probe_sandbox().available` is `False`, `"bash"` is omitted from worker tools and `build_runtime()` prints `[verbix] Note: 'bash' tool disabled (<reason>).` to `stderr`.
- **Single `sounddevice` audio path:** `SubprocessAudioSink` (`src/verbix/voice/sinks.py`), `select_sink()`, and `--audio-sink` are deleted. Missing PortAudio fails fast with a clear install hint (`libportaudio2` / `portaudio`).
- **Model-visible roles & spoken text:** Message roles in `agent.messages` must be strictly `system | user | assistant | tool` (never `"exit"`). `TurnComplete.final_text` carries human-facing prose only; machine exit states ride on `TurnComplete.exit_reason` and `TurnComplete.is_error`.
- **Verification discipline:** Every headline bug fix begins with a failing test that reproduces the bug on current code. Full `set -o pipefail; uv run pytest -q` and `uv run ruff check src tests` must pass before claiming any task complete.

## Review Focus

1. **Negation rules (`!important.py`), nested `.gitignore`, and non-UTF-8 files in pure-Python search fallback (Task 2):** `iter_workspace_files()` must honor `.gitignore` negation patterns via `pathspec.GitIgnoreSpec` without spawning `git`, and `GrepTool` must decode text with `errors="replace"` while skipping binary files (`\x00` in first 4096 bytes).
2. **Empty or unaligned byte chunks in `CallbackAudioSink._enqueue()` (Task 3b):** Passing `b""` or a byte slice not divisible by sample width (4 bytes for `f32`, 2 bytes for `s16`) must truncate cleanly to whole samples without raising `ValueError` or corrupting the contiguous `np.float32` buffer.
3. **Symlinks inside workspace targeting host secrets in `bwrap` and file tools (Task 4):** `<workspace>/.env` and `<workspace>/.verbix/auth.json` must be masked with `--ro-bind /dev/null` inside `bwrap` when present, and child `env` must strip `VERBIX_API_KEY` plus all `*_API_KEY`, `*_SECRET*`, and `*_TOKEN*` variables.
4. **Partial speech delivered (`_delivered_text != ""`) before a foreground circuit breaker trips (Task 5a):** If the foreground agent already emitted spoken `TextDelta` chunks before hitting `RepeatedToolCall` or `LimitsExceeded`, it must preserve `_delivered_text` in `self.messages` and `TurnComplete.final_text` rather than overwriting heard speech with the generic apology string.
5. **Reordered JSON keys or whitespace in repeated tool calls (Task 5b):** Calling a tool with `{"full": false, "task_id": "task-1"}` and then `{"task_id": "task-1", "full": false}` must produce the same canonical signature `(name, json.dumps(args, sort_keys=True))`.

---

### Task 1: CI Matrix (Linux, Windows, macOS) & Python Version Bound

**Files:**
- Create: `.github/workflows/ci.yml`
- Modify: `pyproject.toml:9`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Consumes: `pyproject.toml` project metadata.
- Produces: `.github/workflows/ci.yml` matrix across `ubuntu-latest`, `macos-latest` (`continue-on-error: true`), and `windows-latest` (`continue-on-error: true`), plus `requires-python = ">=3.13,<3.15"`.

- [ ] **Step 1: Write the failing test in `tests/test_cli_smoke.py`**

```python
def test_pyproject_python_ceiling_and_ci_workflow():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.13,<3.15"

    ci_file = root / ".github" / "workflows" / "ci.yml"
    assert ci_file.is_file()
    ci_text = ci_file.read_text(encoding="utf-8")
    assert "ubuntu-latest" in ci_text
    assert "macos-latest" in ci_text
    assert "windows-latest" in ci_text
    assert "apparmor_restrict_unprivileged_userns=0" in ci_text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli_smoke.py::test_pyproject_python_ceiling_and_ci_workflow -v`
Expected: FAIL (`AssertionError: assert '>=3.13' == '>=3.13,<3.15'`)

- [ ] **Step 3: Update `pyproject.toml:9` and create `.github/workflows/ci.yml`**

In `pyproject.toml:9`, set `requires-python = ">=3.13,<3.15"`.
Create `.github/workflows/ci.yml` with a matrix over `os: [ubuntu-latest, macos-latest, windows-latest]`, setting `continue-on-error: ${{ matrix.os != 'ubuntu-latest' }}`, installing `bubblewrap` and `libportaudio2` plus running `sudo sysctl -w kernel.apparmor_restrict_unprivileged_userns=0` on `ubuntu-latest`, and executing `uv sync --locked`, `uv run ruff check src tests`, and `uv run pytest -q`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli_smoke.py::test_pyproject_python_ceiling_and_ci_workflow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .github/workflows/ci.yml tests/test_cli_smoke.py
git commit -m "ci: add Linux/macOS/Windows CI matrix and cap requires-python below 3.15"
```

---

### Task 2: Drop `ripgrep` Wheel & Add `pathspec`-Backed Search Fallback

**Files:**
- Modify: `pyproject.toml:10-25`
- Modify: `THIRD-PARTY-NOTICES.md:59-64`
- Modify: `src/verbix/tools/shared.py:122-131`
- Modify: `src/verbix/tools/grep.py:45-101`
- Modify: `src/verbix/tools/glob.py:35-95`
- Test: `tests/test_tools_grep_glob.py`

**Interfaces:**
- Consumes: `pathspec.GitIgnoreSpec` from `pathspec>=0.12.1`.
- Produces:
  - `find_ripgrep_binary() -> str | None` in `src/verbix/tools/shared.py` (returns `None` when `rg` is not on `PATH`/venv).
  - `iter_workspace_files(target: Path, workspace: Path, glob_pattern: str | None = None) -> list[Path]` in `src/verbix/tools/shared.py` (walks with `os.walk`, prunes `.git`, `.verbix`, `.venv`, `node_modules`, `__pycache__`, and filters via `pathspec.GitIgnoreSpec` from `.gitignore`/`.ignore` files in `workspace` and subdirectories, without spawning `git`).

- [ ] **Step 1: Write the failing reproduction tests in `tests/test_tools_grep_glob.py`**

```python
@pytest.mark.asyncio
async def test_grep_and_glob_stdlib_fallback_honors_gitignore_and_skips_binary(
    sample_workspace: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("verbix.tools.grep.find_ripgrep_binary", lambda: None)
    monkeypatch.setattr("verbix.tools.glob.find_ripgrep_binary", lambda: None)

    (sample_workspace / ".gitignore").write_text("ignored_dir/\n*.log\n!keep.log\n", encoding="utf-8")
    ignored = sample_workspace / "ignored_dir"
    ignored.mkdir()
    (ignored / "secret.py").write_text("def hello_secret(): pass\n", encoding="utf-8")
    (sample_workspace / "drop.log").write_text("hello drop\n", encoding="utf-8")
    (sample_workspace / "keep.log").write_text("hello keep\n", encoding="utf-8")
    (sample_workspace / "binary.py").write_bytes(b"\x00\x01\x02hello binary\n")

    ctx = ToolContext(workspace=sample_workspace)

    glob_res = await GlobTool().execute({"pattern": "**/*"}, ctx)
    assert glob_res.is_error is False
    glob_lines = glob_res.observation.splitlines()
    assert "sub/test.py" in glob_lines
    assert "keep.log" in glob_lines
    assert "drop.log" not in glob_lines
    assert not any("ignored_dir" in line for line in glob_lines)

    grep_res = await GrepTool().execute({"pattern": "hello"}, ctx)
    assert grep_res.is_error is False
    assert "main.py:1:def hello():" in grep_res.observation
    assert "sub/test.py:1:def test_hello():" in grep_res.observation
    assert "keep.log:1:hello keep" in grep_res.observation
    assert "drop.log" not in grep_res.observation
    assert "secret.py" not in grep_res.observation
    assert "binary.py" not in grep_res.observation
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tools_grep_glob.py::test_grep_and_glob_stdlib_fallback_honors_gitignore_and_skips_binary -v`
Expected: FAIL (`TypeError` or `error: failed to execute ripgrep` because `GrepTool` has no fallback when `find_ripgrep_binary()` returns `None`).

- [ ] **Step 3: Replace `ripgrep` with `pathspec>=0.12.1` and implement `iter_workspace_files()` + `GrepTool`/`GlobTool` fallbacks**

1. In `pyproject.toml:10-25`, remove `"ripgrep>=15.1.0"` and add `"pathspec>=0.12.1"`. Run `uv lock && uv sync`.
2. In `THIRD-PARTY-NOTICES.md:59-64`, replace the `ripgrep` section with `pathspec` (`MPL-2.0`, pure-Python gitignore pattern matching by Caleb P. Burns).
3. In `src/verbix/tools/shared.py`:
   - Change `find_ripgrep_binary() -> str | None` to return `None` when neither the venv binary nor `shutil.which("rg")` exists.
   - Implement `iter_workspace_files(target: Path, workspace: Path, glob_pattern: str | None = None) -> list[Path]` using `os.walk` (pruning `.git`, `.verbix`, `.venv`, `node_modules`, `__pycache__`), compiling `.gitignore` and `.ignore` lines with `pathspec.GitIgnoreSpec.from_lines()`, and optionally filtering relative POSIX paths with `pathspec.GitIgnoreSpec.from_lines([glob_pattern])` when `glob_pattern` is given.
4. In `src/verbix/tools/grep.py` and `src/verbix/tools/glob.py`:
   - When `rg_bin := find_ripgrep_binary()` is non-`None`, pass `"--path-separator=/"` in `cmd`.
   - When `rg_bin is None` (or `create_subprocess_exec` raises `OSError`), run `iter_workspace_files()` in `GlobTool` and `GrepTool` (in `GrepTool`, skip files where `raw[:4096]` contains `b"\x00"`, decode with `utf-8` `errors="replace"`, and match lines with `re.compile`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools_grep_glob.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock THIRD-PARTY-NOTICES.md src/verbix/tools/shared.py src/verbix/tools/grep.py src/verbix/tools/glob.py tests/test_tools_grep_glob.py
git commit -m "feat(tools): replace broken ripgrep wheel with pathspec-backed search fallback"
```

---

### Task 3a: Delete `SubprocessAudioSink`, Fail Fast on Missing PortAudio & Set 20 ms `blocksize`

**Files:**
- Delete: `src/verbix/voice/sinks.py`
- Modify: `src/verbix/voice/callback_sink.py:1-192`
- Modify: `src/verbix/cli.py:144-149, 367-400`
- Modify: `tests/test_callback_sink.py`
- Modify: `tests/test_speech_output.py:85-95, 270-295`
- Modify: `tests/test_config_scopes.py:200-235`
- Modify: `tests/test_voice_loop_barge.py:75-85`

**Interfaces:**
- Consumes: `sounddevice.OutputStream`.
- Produces: `CallbackAudioSink(sample_rate: int = 24000, channels: int = 1, audio_format: str = "f32")` in `src/verbix/voice/callback_sink.py` passing `blocksize=sample_rate // 50` to `sd.OutputStream` and raising `RuntimeError` with a PortAudio install hint (`libportaudio2` / `portaudio`) when `sounddevice` cannot load or open a stream.

- [ ] **Step 1: Write the failing tests in `tests/test_callback_sink.py`**

```python
def test_callback_sink_sets_20ms_blocksize_and_fails_fast_with_portaudio_hint(monkeypatch):
    import sys
    import types
    import importlib

    recorded_kwargs = {}

    class FakeStream:
        def __init__(self, *a, **k):
            recorded_kwargs.update(k)

        def start(self):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from verbix.voice.callback_sink import CallbackAudioSink

    sink = CallbackAudioSink(sample_rate=24000)
    assert recorded_kwargs.get("blocksize") == 480
    sink.close()

    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(RuntimeError, match="libportaudio2|portaudio"):
        CallbackAudioSink(sample_rate=24000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_callback_sink.py::test_callback_sink_sets_20ms_blocksize_and_fails_fast_with_portaudio_hint -v`
Expected: FAIL (`AssertionError: assert None == 480`)

- [ ] **Step 3: Delete `src/verbix/voice/sinks.py`, remove `select_sink` and `--audio-sink`, and set `blocksize` + PortAudio install hint**

1. Delete `src/verbix/voice/sinks.py`.
2. In `src/verbix/voice/callback_sink.py`:
   - Remove `from verbix.voice.sinks import SubprocessAudioSink` and `select_sink()`.
   - In `CallbackAudioSink.__init__()`, catch `(ImportError, OSError)` on `import sounddevice as sd` (or stream open failure) and raise `RuntimeError("Audio output unavailable (PortAudio missing or no output device). On Linux, install 'libportaudio2' (Debian/Ubuntu) or 'portaudio' (Fedora/Arch).") from e`.
   - Pass `blocksize=sample_rate // 50` to `sd.OutputStream(...)`.
3. In `src/verbix/cli.py`:
   - Remove the `--audio-sink` argument (`cli.py:144-149`).
   - In `_build_voice_pipeline()` (`cli.py:372-386`), instantiate `CallbackAudioSink(sample_rate=24000, audio_format="f32")` directly.
4. Update `tests/test_callback_sink.py`, `tests/test_speech_output.py`, `tests/test_config_scopes.py`, and `tests/test_voice_loop_barge.py` to remove `SubprocessAudioSink`, `select_sink`, and `--audio-sink` references.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_callback_sink.py tests/test_speech_output.py tests/test_config_scopes.py tests/test_voice_loop_barge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git rm src/verbix/voice/sinks.py
git add src/verbix/voice/callback_sink.py src/verbix/cli.py tests/test_callback_sink.py tests/test_speech_output.py tests/test_config_scopes.py tests/test_voice_loop_barge.py
git commit -m "refactor(voice): remove SubprocessAudioSink and set 20ms blocksize on CallbackAudioSink"
```

---

### Task 3b: Contiguous `np.float32` Buffer Rewrite in `CallbackAudioSink`

**Files:**
- Modify: `src/verbix/voice/callback_sink.py:44-160`
- Test: `tests/test_callback_sink.py`

**Interfaces:**
- Consumes: `numpy.ndarray` (`dtype=np.float32`).
- Produces: `CallbackAudioSink._buf: np.ndarray` storing contiguous `float32` samples instead of `deque[float]`, handling empty/unaligned byte slices safely.

- [ ] **Step 1: Write the failing test in `tests/test_callback_sink.py`**

```python
def test_callback_sink_uses_contiguous_float32_buffer_and_handles_unaligned_bytes(monkeypatch):
    import numpy as np

    sink = _stubbed_sink(monkeypatch)
    assert isinstance(sink._buf, np.ndarray)
    assert sink._buf.dtype == np.float32

    # Unaligned 6-byte chunk for f32 (1 full float32 + 2 trailing bytes) and empty chunk
    raw = np.array([0.5], dtype=np.float32).tobytes() + b"\xff\xff"
    sink._enqueue(b"")
    sink._enqueue(raw)
    assert len(sink._buf) == 1

    out = np.zeros((4, 1), dtype=np.float32)
    sink._callback(out, 4, None, None)
    assert math.isclose(float(out[0, 0]), 0.5, rel_tol=1e-6)
    assert sink.played_samples == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_callback_sink.py::test_callback_sink_uses_contiguous_float32_buffer_and_handles_unaligned_bytes -v`
Expected: FAIL (`AttributeError: 'CallbackAudioSink' object has no attribute '_buf'`)

- [ ] **Step 3: Replace `_pending: deque[float]` with `_buf: np.ndarray` in `src/verbix/voice/callback_sink.py`**

In `CallbackAudioSink`:
- Initialize `self._buf = np.zeros(0, dtype=np.float32)`.
- In `_enqueue(self, data: bytes) -> None`: truncate `data` to `n * 4` (for `f32`) or `n * 2` (for `s16`), convert via `np.frombuffer(..., dtype=np.float32)` (or `np.frombuffer(..., dtype=np.int16).astype(np.float32) / 32768.0`), and concatenate onto `self._buf` under `self._lock`.
- In `_callback(self, outdata: Any, frames: int, _time: Any, _status: Any) -> None`: slice `taken = self._buf[:k]`, advance `self._buf = self._buf[k:]`, apply the linear ramp `(1.0 - (pos + np.arange(k, dtype=np.float32)) / denom)` when `self._stopping` is `True`, and clear `self._buf = np.zeros(0, dtype=np.float32)` when `self._fade_left == 0`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_callback_sink.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/callback_sink.py tests/test_callback_sink.py
git commit -m "perf(voice): replace deque[float] with contiguous float32 buffer in CallbackAudioSink"
```

---

### Task 3c: AEC Reference Realignment to DAC Dequeue (Queue + Worker Thread + `stream_delay_ms`)

**Files:**
- Modify: `src/verbix/voice/callback_sink.py:47-140`
- Modify: `src/verbix/voice/output.py:290-294, 353-360, 414, 430, 460, 478`
- Modify: `src/verbix/voice/pipeline.py:44-49`
- Test: `tests/test_aec_barge_in.py`

**Interfaces:**
- Consumes: `CallbackAudioSink(on_played: Callable[[bytes], None] | None = None)`, `EchoCanceller(sample_rate: int = 16000, stream_delay_ms: int = 20)`.
- Produces: Minimal PortAudio `_callback` that hands dequeued PCM bytes to `self._played_queue: queue.SimpleQueue[bytes | None]` (`put_nowait`, zero logging or resampling inside the real-time callback), drained by a daemon worker thread calling `on_played`, plus `EchoCanceller(stream_delay_ms=20)` in `VoicePipeline`.

- [ ] **Step 1: Write the failing test in `tests/test_aec_barge_in.py`**

```python
def test_callback_sink_feeds_aec_reference_via_worker_thread_on_dac_dequeue(monkeypatch):
    import sys
    import time
    import types

    class FakeStream:
        def __init__(self, *a, **k):
            pass
        def start(self):
            pass
        def stop(self):
            pass
        def close(self):
            pass

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = FakeStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    from verbix.voice.callback_sink import CallbackAudioSink

    fed_bytes = bytearray()
    sink = CallbackAudioSink(sample_rate=24000, on_played=fed_bytes.extend)

    # Enqueue 2400 samples (100ms); nothing has been pulled by DAC yet
    audio = np.ones(2400, dtype=np.float32).tobytes()
    sink._enqueue(audio)
    assert len(fed_bytes) == 0

    # DAC pulls 480 samples (20ms); worker thread drains queue and feeds reference hook
    out = np.zeros((480, 1), dtype=np.float32)
    sink._callback(out, 480, None, None)
    deadline = time.monotonic() + 1.0
    while len(fed_bytes) < 480 * 4 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(fed_bytes) == 480 * 4
    sink.close()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_aec_barge_in.py::test_callback_sink_feeds_aec_reference_via_worker_thread_on_dac_dequeue -v`
Expected: FAIL (`TypeError: CallbackAudioSink.__init__() got an unexpected keyword argument 'on_played'`)

- [ ] **Step 3: Wire `SimpleQueue` + worker thread in `CallbackAudioSink`, `SpeechOutput`, and `stream_delay_ms=20` in `VoicePipeline`**

1. In `src/verbix/voice/callback_sink.py`:
   - Add `on_played: Callable[[bytes], None] | None = None` to `CallbackAudioSink.__init__()`.
   - When `on_played is not None`, create `self._played_queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()` and start a daemon `threading.Thread(target=self._drain_played, daemon=True)` that loops on `item = self._played_queue.get()`, breaks on `None`, and calls `self.on_played(item)`.
   - Inside `_callback()`: if `k > 0 and self._played_queue is not None`, call `self._played_queue.put_nowait(taken.tobytes())` with zero logging, resampling, or heavy work.
   - In `close()`: push `None` to `self._played_queue` to stop the daemon thread cleanly.
2. In `src/verbix/voice/output.py`:
   - In `SpeechOutput.__init__()`, if `self.on_audio is not None` and `hasattr(self.sink, "on_played")`: set `self.sink.on_played = self._notify_ref` and skip calling `self._notify_ref` inside `_feed()` / `play_cue()` so reference audio is never fed at enqueue time.
3. In `src/verbix/voice/pipeline.py:48`:
   - Pass `stream_delay_ms=20` when constructing `EchoCanceller(stream_delay_ms=20)` to match the 20 ms output blocksize.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_aec_barge_in.py tests/test_speech_output.py tests/test_callback_sink.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/callback_sink.py src/verbix/voice/output.py tests/test_aec_barge_in.py
git commit -m "fix(voice): align AEC speaker reference with CallbackAudioSink DAC dequeue"
```

---

### Task 4: Mandatory `bwrap` Sandbox Seam, Capability-Driven `BashTool.description`, Secret Stripping & Startup `bash` Gating

**Files:**
- Modify: `src/verbix/tools/sandbox.py:1-169`
- Modify: `src/verbix/tools/bash.py:1-82`
- Modify: `src/verbix/tools/__init__.py:58-126`
- Modify: `src/verbix/runtime/workers.py:23-45, 95-186`
- Modify: `src/verbix/runtime/runtime.py:151-169, 454-461`
- Modify: `src/verbix/cli.py:186-243`
- Test: `tests/test_tools.py`, `tests/test_worker_tools.py`

**Interfaces:**
- Consumes: `bwrap` binary on Linux.
- Produces:
  - `SandboxStatus(available: bool, backend: str | None, reason: str, capabilities: str)` in `src/verbix/tools/sandbox.py`.
  - `probe_sandbox(allow_network: bool = False) -> SandboxStatus` in `src/verbix/tools/sandbox.py`.
  - `execute_sandboxed(command: str, config: SandboxConfig) -> SandboxResult` in `src/verbix/tools/sandbox.py` (with `--unshare-pid`, masked `.env`/`auth.json`, secret-stripped `env`, `start_new_session=True`, and active `pgid` tracking).
  - `kill_active_processes() -> None` in `src/verbix/tools/sandbox.py`.
  - `BashTool(allow_network: bool = False, extra_binds: ... = None)` (no `sandboxed` parameter; `.description` built from `probe_sandbox(allow_network=self.allow_network).capabilities`).

- [ ] **Step 1: Write the failing reproduction tests in `tests/test_tools.py` and `tests/test_worker_tools.py`**

```python
# In tests/test_tools.py:
@pytest.mark.asyncio
async def test_bash_tool_is_always_sandboxed_strips_secrets_and_masks_dotenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from verbix.tools.sandbox import probe_sandbox

    status = probe_sandbox()
    if not status.available:
        pytest.skip(f"Sandbox unavailable: {status.reason}")

    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".env").write_text("OPENROUTER_API_KEY=sk-dotenv-secret\n", encoding="utf-8")
    (workspace / ".verbix").mkdir()
    (workspace / ".verbix" / "auth.json").write_text('{"openrouter":{"key":"sk-auth-secret"}}', encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env-secret")
    monkeypatch.setenv("VERBIX_API_KEY", "sk-verbix-secret")

    tool = BashTool()
    assert "bubblewrap" in tool.description.lower()
    ctx = ToolContext(workspace=workspace)

    # 1. Host env vars outside the allowlist (PATH, LANG, TERM) must not leak into child env
    monkeypatch.setenv("CUSTOM_UNLISTED_VAR", "should-not-appear")
    env_res = await tool.execute({"command": "env"}, ctx)
    assert "sk-env-secret" not in env_res.observation
    assert "sk-verbix-secret" not in env_res.observation
    assert "should-not-appear" not in env_res.observation

    # 2. Workspace .env and .verbix/auth.json must be masked as /dev/null inside bwrap
    secret_res = await tool.execute({"command": "cat .env .verbix/auth.json"}, ctx)
    assert secret_res.observation.strip() == ""

    # 3. Cannot write outside workspace
    esc_res = await tool.execute({"command": "touch /usr/escape_test"}, ctx)
    assert esc_res.is_error is True


def test_worker_registry_omits_bash_when_sandbox_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    from verbix.runtime.workers import build_worker_registry
    from verbix.tools.sandbox import SandboxStatus

    monkeypatch.setattr(
        "verbix.tools.sandbox.probe_sandbox",
        lambda allow_network=False: SandboxStatus(
            available=False, backend=None, reason="no sandbox on host", capabilities=""
        ),
    )
    reg = build_worker_registry(workspace=tmp_path)
    names = {d["function"]["name"] for d in reg.get_definitions()}
    assert "bash" not in names
    assert "read" in names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools.py::test_bash_tool_is_always_sandboxed_strips_secrets_and_masks_dotenv tests/test_tools.py::test_worker_registry_omits_bash_when_sandbox_unavailable -v`
Expected: FAIL (`sk-env-secret` appears in `env_res.observation` because `BashTool` runs unsandboxed by default and leaks `os.environ`).

- [ ] **Step 3: Implement mandatory `bwrap` sandbox seam, capability reporting, secret masking, and startup gating**

1. In `src/verbix/tools/sandbox.py`:
   - Delete `execute_unsandboxed()`.
   - Add `@dataclass(frozen=True) class SandboxStatus: available: bool; backend: str | None; reason: str; capabilities: str`.
   - Implement `probe_sandbox(allow_network: bool = False) -> SandboxStatus`:
     - On `sys.platform != "linux"`: return `SandboxStatus(False, None, f"no sandbox backend for {sys.platform}", "")`.
     - On Linux: check `shutil.which("bwrap")` and run a cached `subprocess.run([bwrap_path, "--ro-bind", "/", "/", "--", "true"], capture_output=True, timeout=5)`. When `returncode == 0`, return `SandboxStatus(True, "bwrap", "ok", "Executes a bash command in the workspace inside a Linux bubblewrap (bwrap) sandbox: workspace and /tmp are read-write, system paths are read-only, PID namespace is isolated, .env and auth.json are masked, and network is " + ("enabled." if allow_network else "disabled."))`.
   - In `execute_sandboxed(command: str, config: SandboxConfig) -> SandboxResult`:
     - Add `"--unshare-pid"` to `cmd_args`.
     - For `secret_rel` in `(".env", ".verbix/auth.json")`: if `(workspace_path / secret_rel).is_file()`, append `["--ro-bind", "/dev/null", str(workspace_path / secret_rel)]` after the workspace `--bind`.
     - Build child `env` from an explicit allowlist: forward only `PATH`, `LANG`, `TERM` from `os.environ` (when present, defaulting `PATH="/usr/bin:/bin"`, `LANG="C.UTF-8"`), plus `HOME=str(workspace_path / ".verbix" / "home")` (creating the directory), `TMPDIR="/tmp"`, `HISTFILE="/dev/null"`, `PAGER="cat"`, `MANPAGER="cat"`, `PIP_PROGRESS_BAR="off"`, `TQDM_DISABLE="1"`.
     - Replace `preexec_fn=os.setsid` with `start_new_session=True` and `env=child_env`.
     - Track active `pgid` in module-level `_ACTIVE_PGIDS: set[int]` and expose `kill_active_processes() -> None`.
2. In `src/verbix/tools/bash.py`:
   - Remove `sandboxed: bool` from `BashTool.__init__()` and remove the `else: execute_unsandboxed(...)` branch.
   - Make `description` a `@property` returning `probe_sandbox(allow_network=self.allow_network).capabilities or "Executes a bash command in the workspace inside a mandatory OS sandbox."`.
3. In `src/verbix/runtime/workers.py`, `src/verbix/runtime/runtime.py`, and `src/verbix/cli.py`:
   - In `build_worker_registry()` and `spawn_worker()` (`workers.py`), filter `"bash"` out of the tool list when `not probe_sandbox().available`.
   - In `Runtime._build_grounded_prompt()` (`runtime.py:161`), include `" Shell: bash."` only when `probe_sandbox().available` is `True`.
   - In `Runtime.aclose()` (`runtime.py:454-461`), call `kill_active_processes()` from `verbix.tools.sandbox`.
   - In `build_runtime()` (`cli.py`), if `not probe_sandbox().available`, write `[verbix] Note: 'bash' tool disabled ({status.reason}).\n` to `sys.stderr`.
4. Remove `test_unsandboxed_bash_execution` from `tests/test_tools.py` (since unsandboxed execution is now forbidden).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools.py tests/test_worker_tools.py tests/test_runtime_worker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/sandbox.py src/verbix/tools/bash.py src/verbix/runtime/workers.py src/verbix/runtime/runtime.py src/verbix/cli.py tests/test_tools.py tests/test_worker_tools.py
git commit -m "feat(sandbox): enforce mandatory bwrap sandbox, mask secrets, and gate bash at startup"
```

---

### Task 5a: Typed `TurnComplete`, No `"role": "exit"`, and Spoken Fallback

**Files:**
- Modify: `src/verbix/core/events.py:44-46`
- Modify: `src/verbix/agent/voice.py:212-225, 410-422`
- Modify: `src/verbix/voice/output.py:130-142`
- Modify: `src/verbix/cli.py:325-342`
- Modify: `src/verbix/ipc/server.py:43-45`
- Test: `tests/test_events.py`, `tests/test_speech_output.py`, `tests/test_voice_agent.py`

**Interfaces:**
- Consumes: `LitellmModel._prepare_messages_for_api`.
- Produces:
  - `ExitReason = Literal["ok", "terminal_tool", "repeated_tool_call", "limits_exceeded", "time_exceeded", "format_error", "exception"]` in `src/verbix/core/events.py`.
  - `TurnComplete(final_text: str, exit_reason: ExitReason = "ok", is_error: bool = False)` in `src/verbix/core/events.py`.

- [ ] **Step 1: Write the failing reproduction tests in `tests/test_voice_agent.py` and `tests/test_speech_output.py`**

```python
# In tests/test_voice_agent.py:
@pytest.mark.asyncio
async def test_voice_agent_repeat_breaker_never_emits_exit_role_to_api_and_speaks_fallback():
    from verbix.agent.models.litellm_model import LitellmModel

    mock_model = MagicMock()

    async def repeat_stream(messages, tools=None, **kw):
        yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")

    mock_model.query_stream = repeat_stream
    tools = ToolRegistry()
    tools.register(
        "status",
        {"type": "function", "function": {"name": "status"}},
        lambda a: ToolResult("running", is_error=False),
    )

    agent = VoiceAgent(model=mock_model, tools=tools)
    events = [e async for e in agent.step_stream("what?")]

    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "repeated_tool_call"
    assert tc.is_error is True
    assert "RepeatedToolCall" not in tc.final_text
    assert "Sorry" in tc.final_text

    # Assert on the exact payload built by LitellmModel._prepare_messages_for_api
    litellm_model = LitellmModel(model_name="openrouter/google/gemini-3.5-flash-lite")
    api_payload = litellm_model._prepare_messages_for_api(agent.messages)
    roles = {m.get("role") for m in api_payload}
    assert "exit" not in roles
    assert roles <= {"system", "user", "assistant", "tool"}

    # Announcer must have empty spoken_error_fallback so it never says "Could you say that again?"
    announcer = AnnouncerAgent(model=mock_model, config=AnnouncerConfig(step_limit=1, turn_scoped_limits=False))
    announcer.n_calls = 1
    ann_events = [e async for e in announcer.step_stream("debrief")]
    ann_tc = [e for e in ann_events if isinstance(e, TurnComplete)][-1]
    assert ann_tc.exit_reason == "limits_exceeded"
    assert ann_tc.final_text == ""
    assert not any("Could you say that again" in str(m.get("content", "")) for m in announcer.messages)


@pytest.mark.asyncio
async def test_voice_agent_preserves_already_delivered_speech_on_limit_exit():
    mock_model = MagicMock()
    calls = 0

    async def partial_then_tool_stream(messages, tools=None, **kw):
        nonlocal calls
        calls += 1
        yield TextDelta(text="I checked your workspace right now. ")
        yield TextDelta(text="Let me also check status.")
        yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")

    mock_model.query_stream = partial_then_tool_stream
    tools = ToolRegistry()
    tools.register("status", {"type": "function", "function": {"name": "status"}}, lambda a: ToolResult("ok"))

    config = VoiceConfig(step_limit=1, turn_scoped_limits=True)
    agent = VoiceAgent(model=mock_model, tools=tools, config=config)
    events = [e async for e in agent.step_stream("check")]

    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "limits_exceeded"
    assert tc.is_error is True
    assert "LimitsExceeded" not in tc.final_text
    assert "I checked your workspace right now." in tc.final_text


# In tests/test_speech_output.py:
@pytest.mark.asyncio
async def test_speech_output_never_synthesizes_error_turn_complete_tokens():
    from verbix.core.events import TurnComplete
    from verbix.voice.base import MockAudioSink, MockTextToSpeech
    from verbix.voice.output import SpeechOutput

    tts = MockTextToSpeech()
    sink = MockAudioSink(playback_delay=0.01)
    out = SpeechOutput(tts=tts, sink=sink)

    await out.emit(TurnComplete(final_text="RepeatedToolCall", exit_reason="repeated_tool_call", is_error=True))
    await asyncio.sleep(0.05)
    assert tts.synthesized_texts == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_voice_agent.py::test_voice_agent_repeat_breaker_never_emits_exit_role_to_api_and_speaks_fallback tests/test_speech_output.py::test_speech_output_never_synthesizes_error_turn_complete_tokens -v`
Expected: FAIL (`TypeError: TurnComplete.__init__() got an unexpected keyword argument 'exit_reason'`)

- [ ] **Step 3: Implement typed `TurnComplete`, remove `"role": "exit"`, and filter machine exits in `SpeechOutput` and `CliPrinter`**

1. In `src/verbix/core/events.py:44-46`: add `ExitReason` literal and `exit_reason: ExitReason = "ok"`, `is_error: bool = False` to `TurnComplete`.
2. In `src/verbix/agent/voice.py` and `src/verbix/agent/announcer.py`:
   - Add `spoken_error_fallback: str = "Sorry — I ran into a problem. Could you say that again?"` to `VoiceConfig`, and set `spoken_error_fallback: str = ""` on `AnnouncerConfig` (`src/verbix/agent/announcer.py`) so the announcer never speaks `"Could you say that again?"` or writes it into `foreground.messages`.
   - When `step_limit`/`cost_limit` (`exit_reason="limits_exceeded"`), `wall_time_limit_seconds` (`exit_reason="time_exceeded"`), or `_REPEAT_LIMIT` (`exit_reason="repeated_tool_call"`) trips:
     - Never append `{"role": "exit", ...}` to `self.messages`.
     - Repair `self.messages` via `validate_and_repair_history(self.messages, delivered_text=self._delivered_text)`.
     - If `self._delivered_text.strip()` is non-empty, set `final = self._delivered_text`; otherwise set `final = self.config.spoken_error_fallback` and append `{"role": "assistant", "content": final}` to `self.messages` only when `final` is non-empty.
     - Yield `TurnComplete(final_text=final, exit_reason=..., is_error=True)`.
   - On `res.is_terminal` (`voice.py:419`), yield `TurnComplete(final_text=res.observation, exit_reason="terminal_tool", is_error=res.is_error)`.
   - On `except Exception` (`voice.py:439`), use `note = self.config.spoken_error_fallback`, append `{"role": "assistant", "content": note}` only when `note` is non-empty, and yield `TurnComplete(final_text=note, exit_reason="exception", is_error=True)`.
3. In `src/verbix/voice/output.py:130-141` and `src/verbix/cli.py:325-341`:
   - Guard against any residual internal control tokens (`"RepeatedToolCall"`, `"LimitsExceeded"`, `"TimeExceeded"`, `"RepeatedFormatError"`) so `SpeechOutput.emit()` and `CliPrinter.emit()` never synthesize or print them even if an unmigrated caller emits them.
4. Update `tests/test_voice_agent.py::test_voice_agent_limits_exceeded` to assert `e.exit_reason == "limits_exceeded"` and `e.is_error is True`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_events.py tests/test_voice_agent.py tests/test_speech_output.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/core/events.py src/verbix/agent/voice.py src/verbix/voice/output.py src/verbix/cli.py src/verbix/ipc/server.py tests/test_events.py tests/test_voice_agent.py tests/test_speech_output.py
git commit -m "fix(agent,voice): separate TurnComplete exit_reason from spoken final_text and drop exit role"
```

---

### Task 5b: Per-Turn Tool Gating in `VoiceAgent` & `Runtime.status()` Fidelity

**Files:**
- Modify: `src/verbix/agent/voice.py:56-82, 160-170, 208-235, 354-420`
- Modify: `src/verbix/agent/config/voice.yaml:29-35`
- Modify: `src/verbix/runtime/runtime.py:367-398`
- Test: `tests/test_voice_agent.py`, `tests/test_runtime_e2e.py`

**Interfaces:**
- Consumes: `WorkerTask.get_status(full: bool = False) -> str`.
- Produces:
  - `VoiceConfig.max_calls_per_tool_per_turn: dict[str, int] = Field(default_factory=lambda: {"status": 1, "cancel": 1})` in `src/verbix/agent/voice.py` and `src/verbix/agent/config/voice.yaml`.
  - `Runtime.status(task_id: str | None = None, full: bool = False) -> str` including active task progress lines even when `task_id` is omitted.

- [ ] **Step 1: Write the failing reproduction tests in `tests/test_voice_agent.py` and `tests/test_runtime_e2e.py`**

```python
# In tests/test_voice_agent.py:
@pytest.mark.asyncio
async def test_voice_agent_prunes_status_after_one_call_in_same_turn():
    seen_tools_per_call = []

    async def fake_stream(messages, tools=None, **kw):
        tool_names = [t["function"]["name"] for t in (tools or [])]
        seen_tools_per_call.append(tool_names)
        if len(seen_tools_per_call) == 1:
            yield ToolCallDelta(index=0, call_id="c1", name="status", arguments_delta="{}")
        else:
            yield TextDelta(text="Task-1 is currently running pytest.")
            yield TurnComplete(final_text="Task-1 is currently running pytest.")

    mock_model = MagicMock()
    mock_model.query_stream = fake_stream

    tools = ToolRegistry()
    for name in ("spawn", "status", "cancel"):
        tools.register(
            name,
            {"type": "function", "function": {"name": name}},
            lambda a: ToolResult("task-1 running: Called bash"),
        )

    agent = VoiceAgent(model=mock_model, tools=tools, config=VoiceConfig(tools=["spawn", "status", "cancel"]))
    events = [e async for e in agent.step_stream("what is happening?")]

    assert seen_tools_per_call[0] == ["spawn", "status", "cancel"]
    assert "status" not in seen_tools_per_call[1]
    assert "spawn" in seen_tools_per_call[1]
    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "ok"
    assert "Task-1 is currently running pytest." in tc.final_text


@pytest.mark.asyncio
async def test_voice_agent_canonicalizes_json_args_in_repeat_history():
    calls = 0

    async def reordered_stream(messages, tools=None, **kw):
        nonlocal calls
        calls += 1
        arg_variants = [
            '{"a": 1, "b": 2}',
            '{"b": 2, "a": 1}',
            '{ "a" : 1 , "b" : 2 }',
        ]
        yield ToolCallDelta(index=0, call_id=f"c{calls}", name="spawn", arguments_delta=arg_variants[(calls - 1) % 3])

    mock_model = MagicMock()
    mock_model.query_stream = reordered_stream
    tools = ToolRegistry()
    tools.register("spawn", {"type": "function", "function": {"name": "spawn"}}, lambda a: ToolResult("started"))

    agent = VoiceAgent(model=mock_model, tools=tools, config=VoiceConfig(tools=["spawn"], max_calls_per_tool_per_turn={}))
    events = [e async for e in agent.step_stream("go")]
    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "repeated_tool_call"
    assert calls == 3


# In tests/test_runtime_e2e.py:
def test_runtime_status_without_task_id_includes_recent_actions_and_honors_full(tmp_path):
    from verbix.runtime.worker import WorkerTask

    runtime = Runtime(workspace=tmp_path, provider=MagicMock())
    session = MagicMock(spec=["messages"])
    session.messages = [
        {"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "read", "arguments": '{"path": "README.md"}'}}]},
        {"role": "tool", "content": "showing lines 1-10 of 10\n# Verbix"},
    ]
    t = WorkerTask(id="task-1", description="read docs", session=session)
    t._status = "running"
    runtime.tasks["task-1"] = t

    concise = runtime.status()
    assert "Called read" in concise

    full = runtime.status(full=True)
    assert 'read({"path": "README.md"})' in full
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_voice_agent.py::test_voice_agent_prunes_status_after_one_call_in_same_turn tests/test_voice_agent.py::test_voice_agent_canonicalizes_json_args_in_repeat_history tests/test_runtime_e2e.py::test_runtime_status_without_task_id_includes_recent_actions_and_honors_full -v`
Expected: FAIL (`"status"` is still in `seen_tools_per_call[1]`, `reordered_stream` loops endlessly, and `"Called read"` is missing from `runtime.status()`).

- [ ] **Step 3: Implement `max_calls_per_tool_per_turn` in `VoiceAgent` and active task action reporting in `Runtime.status()`**

1. In `src/verbix/agent/config/voice.yaml:33`, add:
   ```yaml
   max_calls_per_tool_per_turn:
     status: 1
     cancel: 1
   ```
2. In `src/verbix/agent/voice.py`:
   - Add `max_calls_per_tool_per_turn: dict[str, int] = Field(default_factory=lambda: {"status": 1, "cancel": 1})` to `VoiceConfig`.
   - Track `per_turn_counts: dict[str, int] = {}` inside `step_stream()`.
   - Update `get_tool_definitions(self, excluded: set[str] | None = None)` to filter out any tool name where `per_turn_counts.get(name, 0) >= self.config.max_calls_per_tool_per_turn.get(name, 999999)`.
   - Before executing a tool call in `step_stream()`, if `per_turn_counts.get(name, 0) >= cap`, skip execution and record `ToolResult(f"Tool '{name}' was already called in this turn. Respond to the user directly using the previous result.", is_error=True)`.
   - Canonicalize `sig` in `VoiceAgent.step_stream()` (`voice.py:409`) as `(name, json.dumps(args, sort_keys=True))`.
3. In `src/verbix/runtime/runtime.py:382-388`:
   - For each active task `t` in `Runtime.status(task_id=None, full=False)`, call `t.get_status(full=full)` so active task progress (`Recent actions:` or `Full history:`) is included when `task_id` is omitted.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_voice_agent.py tests/test_runtime_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/agent/voice.py src/verbix/agent/config/voice.yaml src/verbix/runtime/runtime.py tests/test_voice_agent.py tests/test_runtime_e2e.py
git commit -m "fix(agent,runtime): enforce per-turn status/cancel cap and include worker actions in status()"
```

---

### Task 5c: Announcer `mark_heard()`, Dead Code Removal & Prompt/Tool Hardening

**Files:**
- Modify: `src/verbix/runtime/announcer.py:135-205`
- Modify: `src/verbix/runtime/runtime.py:52-68, 433-453`
- Modify: `src/verbix/runtime/foreground.py:54-90`
- Modify: `src/verbix/agent/config/voice.yaml:16-24`
- Modify: `src/verbix/tools/spawn.py:16-36`
- Test: `tests/test_announcer_config.py`, `tests/test_prompt_invariants.py`

**Interfaces:**
- Consumes: `TaskLedger.mark_heard(task_id: str)`.
- Produces: `drain_pending_announcements(runtime)` marking delivered tasks as `"heard"`, updated `DEFAULT_WORKER_PROMPT` without `"finish"`, and STT-ambiguity rules in `voice.yaml` and `SpawnTool`.

- [ ] **Step 1: Write the failing reproduction tests in `tests/test_announcer_config.py` and `tests/test_prompt_invariants.py`**

```python
# In tests/test_announcer_config.py:
@pytest.mark.asyncio
async def test_drain_pending_announcements_marks_ledger_heard(tmp_path):
    from verbix.runtime.announcer import drain_pending_announcements
    from verbix.runtime.runtime import Runtime

    rt = Runtime(workspace=tmp_path, provider=MagicMock())
    rt.ledger.register_task("task-1", "test objective")
    rt.ledger.mark_completed("task-1", "all good")
    rt._pending_announcements.append(("task-1", "all good"))

    async def fake_submit(prompt, output, agent=None):
        return "Task one is done: all good."

    rt.submit = fake_submit
    await drain_pending_announcements(rt)
    assert rt.ledger.get("task-1").delivery == "heard"


# In tests/test_prompt_invariants.py:
def test_default_worker_prompt_does_not_reference_removed_finish_tool():
    assert "call the finish tool" not in DEFAULT_WORKER_PROMPT.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_announcer_config.py::test_drain_pending_announcements_marks_ledger_heard tests/test_prompt_invariants.py::test_default_worker_prompt_does_not_reference_removed_finish_tool -v`
Expected: FAIL (`delivery` is `"pending"` instead of `"heard"`, and `"call the finish tool"` is in `DEFAULT_WORKER_PROMPT`).

- [ ] **Step 3: Implement `ledger.mark_heard()` in `drain_pending_announcements()`, remove dead `speak_batch()`, and update prompts**

1. In `src/verbix/runtime/announcer.py`:
   - In `drain_pending_announcements(runtime)` (`announcer.py:155-158`), when `final_text` is non-empty and `hasattr(runtime, "ledger")`, call `runtime.ledger.mark_heard(task_id)` for each `task_id, _` in `batch`.
   - Delete dead `speak_batch()` (`announcer.py:164-205`) and remove it from `src/verbix/runtime/runtime.py` (`_announce_ids`, `_speak_batch`, `__all__`). Update `test_speak_batch_shares_snapshot_and_writes_back` in `tests/test_announcer_config.py` to test `drain_pending_announcements`.
2. In `src/verbix/runtime/foreground.py:88-89`:
   - Change `DEFAULT_WORKER_PROMPT` to: `"You are an autonomous background worker executing tasks for Verbix.\nDiscover the necessary environment and tools, execute actions directly, verify outcomes, and conclude with an information-dense factual summary of results upon completion."`
   - Update `test_default_worker_prompt_invariants` in `tests/test_prompt_invariants.py:53` accordingly.
3. In `src/verbix/agent/config/voice.yaml:17` and `src/verbix/tools/spawn.py:29-32`:
   - Add explicit rule in `voice.yaml` under `# TOOLS`: `- User input arrives via live speech-to-text: never call spawn on greetings, short phonetic fragments, or ambiguous transcriptions. If what the user wants done is unclear, ask a brief one-sentence question instead of spawning a task to investigate.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_announcer_config.py tests/test_prompt_invariants.py tests/test_runtime_e2e.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/runtime/announcer.py src/verbix/runtime/runtime.py src/verbix/runtime/foreground.py src/verbix/agent/config/voice.yaml src/verbix/tools/spawn.py tests/test_announcer_config.py tests/test_prompt_invariants.py
git commit -m "fix(runtime,prompts): mark announced tasks heard, drop dead speak_batch, and harden STT spawn rules"
```

---

### Task 5d: Console `drain_stdin()` Windows Portability & Platform Tier Docs

**Files:**
- Modify: `src/verbix/cli.py:245-253`
- Modify: `README.md:88-95, 211-221`
- Modify: `.github/workflows/ci.yml`
- Test: `tests/test_ipc_and_cli.py`

**Interfaces:**
- Consumes: `msvcrt` on `os.name == "nt"`, `select.select` on POSIX.
- Produces: Cross-platform `drain_stdin() -> None` in `src/verbix/cli.py` and documented platform tiers in `README.md`.

- [ ] **Step 1: Write the failing test in `tests/test_ipc_and_cli.py`**

```python
def test_drain_stdin_uses_msvcrt_on_windows(monkeypatch):
    import sys
    import types
    from verbix.cli import drain_stdin

    select_called = False

    def exploding_select(*a, **k):
        nonlocal select_called
        select_called = True
        raise AssertionError("select.select must not be called on Windows stdin")

    chars = ["a", "\n"]
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.kbhit = lambda: bool(chars)
    fake_msvcrt.getwch = lambda: chars.pop(0)

    monkeypatch.setattr("verbix.cli.os.name", "nt")
    monkeypatch.setattr("verbix.cli.select.select", exploding_select)
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)

    drain_stdin()
    assert select_called is False
    assert chars == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_ipc_and_cli.py::test_drain_stdin_uses_msvcrt_on_windows -v`
Expected: FAIL (`AssertionError: select.select must not be called on Windows stdin`)

- [ ] **Step 3: Update `drain_stdin()` in `src/verbix/cli.py` and document platform tiers in `README.md`**

1. In `src/verbix/cli.py:245-253`:
   ```python
   def drain_stdin() -> None:
       """Discards already-buffered stdin lines so stale Enters never become phantom commands."""
       if os.name == "nt":
           try:
               import msvcrt

               while msvcrt.kbhit():
                   msvcrt.getwch()
           except (ImportError, OSError):
               pass
           return
       try:
           while select.select([sys.stdin], [], [], 0)[0]:
               if not os.read(sys.stdin.fileno(), 4096):
                   break
       except (OSError, ValueError):
           pass
   ```
2. In `README.md`:
   - Document platform support tiers under `## Platform support`: Linux (`x86_64`, `aarch64`) is the supported tier (requires `bubblewrap` and `libportaudio2`); Windows (`amd64`) and macOS (`arm64`, `x86_64`) are experimental and untested (`bash` is disabled because no sandbox backend is enabled there; workers use the 8 cross-platform tools).
   - Remove stale `--audio-sink` references from `README.md:32`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ipc_and_cli.py tests/test_cli_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/cli.py README.md tests/test_ipc_and_cli.py
git commit -m "fix(cli,docs): make drain_stdin cross-platform and document platform support tiers"
```

---

### Task 6a: Continuous Voice Loop `_live_stt` Reset After Playback

**Files:**
- Modify: `src/verbix/voice/pipeline.py:381-408`
- Test: `tests/test_conversational_voice_loop.py`

**Interfaces:**
- Consumes: `VoicePipeline._setup_turn_listening()`.
- Produces: Clean `_live_stt` reset in `VoicePipeline._run_continuous_loop()` after `execute_voice_turn()` finishes without barge-in so playback residual audio fed via `feed_barge_in()` never leaks into the next listening turn.

- [ ] **Step 1: Write the failing reproduction test in `tests/test_conversational_voice_loop.py`**

```python
@pytest.mark.asyncio
async def test_continuous_loop_resets_stt_session_after_normal_playback_completion():
    from unittest.mock import AsyncMock, MagicMock
    from verbix.runtime.floor import FloorArbiter
    from verbix.voice.base import MockAudioSink, MockTextToSpeech
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer

    sessions = []

    def make_session(**kw):
        s = MagicMock()
        s.polluted_by_barge_in = False
        s.speech_duration = 1.0
        s.feed = MagicMock()
        def _barge(chunk):
            s.polluted_by_barge_in = True
            return False
        s.feed_barge_in = _barge
        s.finalize = AsyncMock(return_value="hello there")
        s.cancel = MagicMock()
        sessions.append(s)
        return s

    stt = MagicMock()
    stt.open_session = MagicMock(side_effect=make_session)

    runtime = MagicMock()
    runtime.floor = FloorArbiter()
    runtime.is_idle = True
    runtime.pending_continuation = None

    capture = MagicMock()
    capture.sample_rate = 16000
    capture.is_recording = False
    capture.start_recording = AsyncMock()
    capture.stop_recording = AsyncMock(return_value=b"")

    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=AudioPlayer(tts=MockTextToSpeech(), sink=MockAudioSink()),
        capture=capture,
    )

    async def fake_submit(text, output, **kw):
        runtime.floor.set_state("SPEAKING")
        # Simulate mic chunk arriving while Verbix is speaking (no barge-in triggered)
        pipeline.capture.on_chunk(b"\x01\x00" * 160)
        return "Hi!"

    runtime.submit = fake_submit

    await pipeline.start_continuous_session()
    pipeline._speech_endpoint_event.set()
    await asyncio.sleep(0.05)

    # After Turn 1 playback completes normally without barge-in, the old session must be
    # cancelled and the active _live_stt waiting for Turn 2 MUST be a fresh session
    # seeded with any tail lookback audio!
    assert sessions[-2].cancel.called is True
    assert pipeline._live_stt is sessions[-1]
    assert pipeline._live_stt.polluted_by_barge_in is False
    assert pipeline._live_stt.feed.called is True
    await pipeline.stop_continuous_session()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_conversational_voice_loop.py::test_continuous_loop_resets_stt_session_after_normal_playback_completion -v`
Expected: FAIL (`AssertionError: assert True is False` because `pipeline._live_stt` was opened before `execute_voice_turn`, fed via `feed_barge_in` during playback, and skipped in `finally:` because `self._live_stt is not None`).

- [ ] **Step 3: Reset `_live_stt` in `VoicePipeline._run_continuous_loop()` when playback finishes without barge-in**

In `src/verbix/voice/pipeline.py:272-290, 381-408`:
- In `_setup_turn_listening()`: always cancel the old `self._live_stt` (`if self._live_stt is not None: self._live_stt.cancel()`) before calling `self.stt.open_session(...)`, and if `self._mic_lookback` is non-empty, feed `self._mic_lookback` into the newly opened session and clear `self._mic_lookback = b""` so speech starting at the tail end of playback is never dropped.
- Track whether `_interrupt_active_speech()` occurred during `turn_task` (via `self._barge_in_triggered = False` cleared before `execute_voice_turn` and set to `True` in `_interrupt_active_speech()`).
- In `finally:` of `_run_continuous_loop()` (`pipeline.py:405-407`), if `self._is_continuous_active` and (`self._live_stt is None` or not `self._barge_in_triggered`), call `await self._setup_turn_listening()` so the old session is cancelled, any tail lookback is seeded, and the next turn starts cleanly.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_conversational_voice_loop.py tests/test_aec_barge_in.py tests/test_audio_loop_states.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/pipeline.py tests/test_conversational_voice_loop.py
git commit -m "fix(voice): reset continuous-mode STT session after normal playback completion"
```

---

### Task 6b: Hypothesis Check — `TranscribeSession` VAD Span Accumulation

**Files:**
- Modify (if hypothesis holds): `src/verbix/voice/transcribe_stt.py:84-166`
- Test: `tests/test_transcribe_stt.py`, `tests/test_stt_session_contract.py`

**Interfaces:**
- Consumes: `SileroSegmenter.pop_segments() -> list[bytes]` and `SileroSegmenter.flush() -> list[bytes]`.
- Produces: `TranscribeSession.finalize() -> str` passing only VAD-committed speech spans to `self._stt.transcribe()` when a segmenter is attached and produces spans.

- [ ] **Step 1: Write the hypothesis reproduction test in `tests/test_transcribe_stt.py`**

```python
@pytest.mark.asyncio
async def test_transcribe_session_drops_pre_speech_noise_outside_vad_spans():
    from unittest.mock import MagicMock
    from verbix.voice.transcribe_stt import TranscribeSession

    transcribed_buffers = []
    fake_stt = MagicMock()
    fake_stt.transcribe = lambda audio, sr: (transcribed_buffers.append(audio) or "hey what's up")

    speech_span = b"\x10\x00" * 1600  # 100ms span
    fake_seg = MagicMock()
    fake_seg.pop_segments = MagicMock(side_effect=[[], [], [speech_span]])
    fake_seg.flush = MagicMock(return_value=[])

    sess = TranscribeSession(stt=fake_stt, segmenter=fake_seg, sample_rate=16000)
    pre_noise = b"\x01\x00" * 1600
    sess.feed(pre_noise)
    sess.feed(pre_noise)
    sess.feed(speech_span)

    res = await sess.finalize()
    assert res == "hey what's up"
    assert transcribed_buffers == [speech_span]
```

- [ ] **Step 2: Run test to verify whether the hypothesis reproduces**

Run: `uv run pytest tests/test_transcribe_stt.py::test_transcribe_session_drops_pre_speech_noise_outside_vad_spans -v`
Expected: FAIL (`len(transcribed_buffers[0]) == 9600` instead of `3200` because `pre_noise` chunks were appended to `self._buf`).

- [ ] **Step 3: Accumulate VAD spans in `TranscribeSession` and verify all existing STT contract tests**

In `src/verbix/voice/transcribe_stt.py:84-166`:
- Add `self._spans: list[bytes] = []` to `TranscribeSession.__init__()`.
- In `feed()`, extend `self._spans.extend(spans)` when `self._segmenter.pop_segments()` returns spans.
- In `finalize()`, collect `flushed = self._segmenter.flush()` and `self._spans.extend(flushed)`. If `self._spans` is non-empty, set `audio = b"".join(self._spans)` (falling back to `bytes(self._buf)` only when `self._segmenter is None` or when a mock segmenter in legacy unit tests does not implement `pop_segments`/`flush`). Clear `self._spans.clear()` on `cancel()`/`finalize()`.
- *(If this change breaks any existing contract in `tests/test_transcribe_stt.py` or `tests/test_stt_session_contract.py`, revert and drop Task 6b per spec).*

- [ ] **Step 4: Run all STT tests to verify they pass**

Run: `uv run pytest tests/test_transcribe_stt.py tests/test_stt_session_contract.py tests/test_incremental_stt.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/transcribe_stt.py tests/test_transcribe_stt.py
git commit -m "fix(stt): transcribe VAD-committed speech spans in TranscribeSession"
```

---

### Go/No-Go Gate Before Task 7: Reassess Engine Unification

- [ ] **Gate Check: Compare `src/verbix/agent/voice.py` and `src/verbix/agent/worker.py` after Tasks 1–6**

Run: `wc -l src/verbix/agent/voice.py src/verbix/agent/worker.py src/verbix/agent/announcer.py`
Evaluate:
1. Does unifying `VoiceAgent` and `WorkerAgent` into `src/verbix/agent/engine.py` eliminate meaningful duplication beyond what Task 8 (`DefaultAgent` deletion) already removes?
2. Report the remaining duplication and line counts to the engineer and get an explicit **Go** (execute Task 7 then Task 8) or **No-Go** (skip Task 7 and go directly to Task 8).

---

### Task 7 (Conditional on Go at Gate): Engine Unification ("One Engine, Three Flavors")

**Files:**
- Create: `src/verbix/agent/engine.py`
- Modify: `src/verbix/agent/voice.py`, `src/verbix/agent/worker.py`, `src/verbix/agent/announcer.py`
- Modify: `src/verbix/runtime/worker.py:123-208`
- Test: `tests/test_worker_agent.py`, `tests/test_voice_agent.py`, `tests/test_worker_watch.py`

**Interfaces:**
- Consumes: `LitellmModel.query_stream`, `ToolRegistry`, `compact_messages`, `validate_and_repair_history`, `split_chunks`.
- Produces: `Agent` and `AgentConfig` in `src/verbix/agent/engine.py` supporting `speech_chunking`, `turn_scoped_limits`, `require_tool_before_finish`, `max_calls_per_tool_per_turn`, `repeat_limit`, `step_stream()`, and `run()`.

- [ ] **Step 1: Write the failing test in `tests/test_worker_agent.py`**

```python
@pytest.mark.asyncio
async def test_worker_streams_live_events_and_completes_without_format_error_exception():
    from verbix.core.events import TextDelta, TurnComplete

    mock_model = MagicMock()
    calls = 0

    async def fake_stream(messages, tools=None, **kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            yield ToolCallDelta(index=0, call_id="c1", name="read", arguments_delta='{"path": "a.txt"}')
        else:
            yield TextDelta(text="File a.txt contains ")
            yield TextDelta(text="42 lines.")
            yield TurnComplete(final_text="File a.txt contains 42 lines.")

    mock_model.query_stream = fake_stream
    tools = ToolRegistry()
    tools.register("read", {"type": "function", "function": {"name": "read"}}, lambda a: ToolResult("42 lines"))

    agent = WorkerAgent(model=mock_model, tools=tools)
    events = [e async for e in agent.step_stream("check a.txt")]
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert deltas == ["File a.txt contains ", "42 lines."]
    tc = [e for e in events if isinstance(e, TurnComplete)][-1]
    assert tc.exit_reason == "ok"
    assert tc.final_text == "File a.txt contains 42 lines."
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_worker_agent.py::test_worker_streams_live_events_and_completes_without_format_error_exception -v`
Expected: FAIL (`AttributeError: 'WorkerAgent' object has no attribute 'step_stream'`)

- [ ] **Step 3: Implement `Agent` and `AgentConfig` in `src/verbix/agent/engine.py` and wire `VoiceAgent`, `WorkerAgent`, and `AnnouncerAgent`**

1. Create `src/verbix/agent/engine.py` with `AgentConfig` and `Agent` implementing `step_stream()`, `step()`, and `run()`, supporting:
   - `speech_chunking: bool` (`True` for `VoiceConfig`/`AnnouncerConfig`; `False` for `WorkerConfig`).
   - `require_tool_before_finish: bool` (`True` for `WorkerConfig`; `False` for `VoiceConfig`/`AnnouncerConfig`).
   - 3-strike identical tool signature breaker (`repeat_mode="inject_error"` for Worker matching `bae14b1`, `repeat_mode="abort_turn"` for Voice).
2. Re-export `VoiceAgent`, `WorkerAgent`, and `AnnouncerAgent` as thin config-preset subclasses of `Agent` so existing imports remain intact until Task 8.

- [ ] **Step 4: Run full test suite to verify it passes**

Run: `uv run pytest tests/test_voice_agent.py tests/test_worker_agent.py tests/test_announcer_config.py tests/test_runtime_worker.py tests/test_worker_watch.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/verbix/agent/engine.py src/verbix/agent/voice.py src/verbix/agent/worker.py src/verbix/agent/announcer.py src/verbix/runtime/worker.py tests/test_worker_agent.py
git commit -m "refactor(agent): unify Voice, Worker, and Announcer on one streaming Agent engine"
```

---

### Task 8: Dedicated Commit — Port Required Tests & Delete Legacy Mini-SWE-Agent Surface

**Files:**
- Delete: `src/verbix/agent/agents/default.py`, `src/verbix/agent/agents/__init__.py`
- Delete: `src/verbix/agent/environments/local.py`, `src/verbix/agent/environments/__init__.py`
- Delete: `src/verbix/tools/finish.py`
- Delete: `src/verbix/agent/models/utils/actions_toolcall.py`, `src/verbix/agent/models/utils/retry.py`
- Delete: `tests/test_miniswe_agent_turn.py`, `tests/test_miniswe_environment.py`, `tests/test_miniswe_tools_and_model.py`, `tests/test_local_environment.py`
- Modify: `src/verbix/agent/__init__.py`, `src/verbix/tools/__init__.py`, `src/verbix/runtime/foreground.py`, `src/verbix/runtime/workers.py`, `src/verbix/runtime/worker.py`, `src/verbix/agent/models/litellm_model.py`
- Modify: `tests/test_turn_boundary.py`, `tests/test_agent_limits_and_caps.py`, `tests/test_announcer_config.py`, `tests/test_ipc_and_cli.py`, `tests/test_tools.py`, `tests/test_runtime_worker.py`

**Interfaces:**
- Consumes: `VoiceAgent` / `WorkerAgent` / `AnnouncerAgent` (or unified `Agent`).
- Produces: Clean `src/verbix/agent/` package with zero `DefaultAgent`, `LocalEnvironment`, `FinishTool`, or `actions_toolcall` remnants.

- [ ] **Step 1: Port any remaining test coverage from `tests/test_miniswe_*.py` and `DefaultAgent`-coupled tests**

1. Update `tests/test_turn_boundary.py`, `tests/test_agent_limits_and_caps.py`, `tests/test_announcer_config.py:57-94`, `tests/test_runtime_worker.py:275-296`, and `tests/test_ipc_and_cli.py:127-182` to use `VoiceAgent` / `WorkerAgent` (and `read`/`bash` instead of `FinishTool`).
2. Update `tests/test_tools.py:183-204` (`test_multimodal_expansion_is_on_by_default`) to test `expand_multimodal_content` via `LitellmModel.format_message` without importing `actions_toolcall`.
3. Verify that `tests/test_worker_agent.py` retains `test_worker_circuit_breaker_intercepts_repeated_calls` and `test_worker_circuit_breaker_resets_on_different_arguments` (from `bae14b1`).

- [ ] **Step 2: Run the migrated tests to verify they pass before deleting legacy files**

Run: `uv run pytest tests/test_turn_boundary.py tests/test_agent_limits_and_caps.py tests/test_announcer_config.py tests/test_ipc_and_cli.py tests/test_tools.py tests/test_worker_agent.py -v`
Expected: PASS

- [ ] **Step 3: Delete legacy Mini-SWE-Agent files and dead methods**

1. Delete `src/verbix/agent/agents/default.py`, `src/verbix/agent/agents/__init__.py`, `src/verbix/agent/environments/local.py`, `src/verbix/agent/environments/__init__.py`, `src/verbix/tools/finish.py`, `src/verbix/agent/models/utils/actions_toolcall.py`, `src/verbix/agent/models/utils/retry.py`, `tests/test_miniswe_agent_turn.py`, `tests/test_miniswe_environment.py`, `tests/test_miniswe_tools_and_model.py`, and `tests/test_local_environment.py`.
2. Remove `"finish"` and `FinishTool` from `src/verbix/tools/__init__.py`.
3. Remove `LocalEnvironment` imports/usages from `src/verbix/runtime/foreground.py` and `src/verbix/runtime/workers.py`.
4. Remove the dead `_is_finish_tool_message` / `_MAX_TURNS_WITHOUT_FINISH` branch from `src/verbix/runtime/worker.py:18-32, 144-207`.
5. Remove `LitellmModel._query()`, `query()`, `_parse_actions()`, `format_observation_messages()` from `src/verbix/agent/models/litellm_model.py` (if Task 7 unified `WorkerAgent` on `query_stream`).

- [ ] **Step 4: Run the full test suite and linter**

Run: `set -o pipefail; uv run pytest -q && uv run ruff check src tests`
Expected: PASS with 0 errors.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(agent): remove legacy DefaultAgent, LocalEnvironment, FinishTool, and Mini-SWE tests"
```

---

### Task 9: Dedicated Commit — Retire `src/verbix/voice/sherpa_stt.py` (Canary 180M Flash)

**Files:**
- Delete: `src/verbix/voice/sherpa_stt.py`, `tests/test_sherpa_stt.py`
- Modify: `src/verbix/voice/transcribe_stt.py`
- Modify: `tests/test_stt_session_contract.py`, `tests/test_incremental_stt.py`
- Modify: `README.md:89-90, 156, 189`

**Interfaces:**
- Consumes: `SileroSegmenter`.
- Produces: `STTSession` and `_Segment` hosted in `src/verbix/voice/transcribe_stt.py` (imported by `tests/test_stt_session_contract.py` and `tests/test_incremental_stt.py`), with `src/verbix/voice/sherpa_stt.py` and `tests/test_sherpa_stt.py` deleted.

- [ ] **Step 1: Move `STTSession` and `_Segment` to `src/verbix/voice/transcribe_stt.py` and update test imports**

1. Move `_Segment` and `STTSession` from `src/verbix/voice/sherpa_stt.py:32-168` into `src/verbix/voice/transcribe_stt.py`.
2. Update `from verbix.voice.sherpa_stt import STTSession` to `from verbix.voice.transcribe_stt import STTSession` in `tests/test_stt_session_contract.py` and `tests/test_incremental_stt.py`.
3. Update `README.md:89-90, 156, 189` to reference `TranscribeSTT` (Nemotron 3.5 ASR GGUF via `transcribe.cpp`) instead of `SherpaSTT` (Canary).

- [ ] **Step 2: Delete `src/verbix/voice/sherpa_stt.py` and `tests/test_sherpa_stt.py`**

Run: `git rm src/verbix/voice/sherpa_stt.py tests/test_sherpa_stt.py`

- [ ] **Step 3: Run full test suite and linter to verify no remaining references**

Run: `set -o pipefail; uv run pytest -q && uv run ruff check src tests`
Expected: PASS with 0 errors.

- [ ] **Step 4: Commit**

```bash
git add src/verbix/voice/transcribe_stt.py tests/test_stt_session_contract.py tests/test_incremental_stt.py README.md
git commit -m "chore(voice): retire legacy Canary sherpa_stt module and preserve STTSession contract"
```
