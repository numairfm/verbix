# Remove Continuous Voice Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove continuous voice mode and all continuous-only code, dependencies (`pywebrtc-audio`), model weights (`silero_vad.onnx`), and tests from Verbix while preserving push-to-talk (PTT) and background announcement behavior.

**Architecture:** Push-to-talk (`VoicePipeline.toggle` -> `start_recording` -> `stop_recording_and_process` -> `execute_voice_turn`) becomes the sole voice interaction path. `SpeechOutput` (sentence streaming, warm prefetch, earcon cues, and `played_samples`-based 3-way `InterruptionSnapshot`), `CallbackAudioSink` (10ms fade-out and `played_samples`), `PTTCapture` (DC-bias/high-pass conditioning), `TranscribeSTT.transcribe()` (single-pass GGUF ASR with click/noise gate), and `FloorArbiter` (`IDLE`, `LISTENING`, `SPEAKING`) remain intact without continuous-only hooks.

**Tech Stack:** Python 3.13+, `pytest`, `ruff`, `sounddevice`, `transcribe-cpp`, `sherpa-onnx`.

**Spec:** `docs/superpowers/specs/2026-10-15-remove-continuous-voice-mode-design.md`

## Global Constraints

- Pre-deletion commit `0cf15ec` is tagged as `continuous-mode-v1`.
- No behavior changes to PTT.
- Remove continuous-only code completely with no stubs or commented-out code.
- Keep `--voice-mode` as a temporary clear-error shim (annotated for removal in a later release) that exits with code `2` and a clear message when given anything other than `"ptt"`.
- Every commit must leave the full test suite (`uv run pytest -q`) green and `uv run ruff check src tests` clean.

## Review Focus

1. **PTT continuation stash on Enter-key interrupt**: Pressing Enter during playback in PTT mode stashes `snapshot.unplayed_text` on `runtime.pending_continuation`, and saying `"continue"` or `"go on"` on a subsequent PTT turn prepends `[Stashed unplayed context: ...]` and clears `pending_continuation`.
2. **PTT low-energy noise gate**: `TranscribeSTT.transcribe()` rejects low-energy noise and short clicks via `_has_min_voiced_duration()` without invoking `self._model.session()`.
3. **PTT floor announcement gating**: `FloorArbiter.can_announce()` returns `True` only when `self._state == FloorState.IDLE and not self.interrupt_requested` (never during `LISTENING` or `SPEAKING`), and `FloorArbiter.acquire()` always exits to `FloorState.IDLE`.
4. **Zero residual references to removed symbols**: `grep` across `src/` and `tests/` finds zero references to `set_user_speaking`, `user_speaking`, `continuous_mode`, `on_played`, `on_audio`, `open_session`, `STTSession`, `TranscribeSession`, `SileroSegmenter`, `EnergyVAD`, `EchoCanceller`, `buffer_window`, or `last_user_speech_at`.
5. **Graceful CLI error on `--voice-mode continuous`**: Running `verbix --voice --voice-mode continuous` prints a clear human-readable error to `stderr` and exits with code `2`.

---

### Task 1: Remove Continuous-Only Voice/Runtime Code, Port PTT Continuation Stash, and Migrate Voice/Runtime Tests

**Files:**
- Delete: `src/verbix/voice/aec.py`, `src/verbix/voice/sherpa_vad.py`, `src/verbix/voice/vad.py`
- Delete: `tests/test_conversational_voice_loop.py`, `tests/test_aec_barge_in.py`, `tests/test_sherpa_vad.py`, `tests/test_incremental_stt.py`, `tests/test_stt_session_contract.py`
- Modify: `src/verbix/voice/pipeline.py`, `src/verbix/voice/transcribe_stt.py`, `src/verbix/voice/base.py`, `src/verbix/voice/output.py`, `src/verbix/voice/callback_sink.py`, `src/verbix/voice/capture.py`, `src/verbix/runtime/floor.py`, `src/verbix/runtime/runtime.py`, `src/verbix/runtime/announcer.py`
- Modify: `tests/test_draft_shelf.py`, `tests/test_stt_nospeech.py`, `tests/test_mic_dump.py`, `tests/test_barge_in_heard.py`, `tests/test_capture_warmup.py`, `tests/test_sounddevice_capture.py`, `tests/test_speech_output.py`, `tests/test_transcribe_stt.py`, `tests/test_runtime_e2e.py`, `tests/test_idle_drained_announcements.py`, `tests/test_voice_loop_barge.py`, `tests/test_audio_loop_states.py`, `tests/test_ledger_and_floor.py`, `tests/test_ipc_and_cli.py`

**Interfaces:**
- Produces:
  - `VoicePipeline(runtime, stt, player, capture=None, printer=None, dump_mic_dir=None)` in `src/verbix/voice/pipeline.py` with `toggle()`, `start_recording()`, `stop_recording_and_process()`, `execute_voice_turn()`, `delivered_text()`, `_stash_interruption()`.
  - `TranscribeSTT.transcribe(audio_data: bytes, sample_rate: int = 16000) -> str` in `src/verbix/voice/transcribe_stt.py` (no `STTSession`, `TranscribeSession`, or `open_session`).
  - `SpeechOutput(tts, sink, printer=None)` in `src/verbix/voice/output.py` (no `on_audio`).
  - `CallbackAudioSink(sample_rate=24000, channels=1, audio_format="f32")` in `src/verbix/voice/callback_sink.py` (no `on_played`).
  - `PTTCapture(sample_rate=16000, channels=1)` in `src/verbix/voice/capture.py` (no `on_chunk`).
  - `FloorArbiter.can_announce() -> bool` (`self._state == FloorState.IDLE and not self.interrupt_requested`) and `FloorArbiter.acquire()` (always exits to `FloorState.IDLE`) in `src/verbix/runtime/floor.py`.
  - `Runtime.__init__` in `src/verbix/runtime/runtime.py` without `buffer_window`.

- [ ] **Step 1: Port `tests/test_draft_shelf.py`, `tests/test_stt_nospeech.py`, and `tests/test_audio_loop_states.py` to PTT semantics**
  - Update `tests/test_draft_shelf.py` to drive PTT (`await pipeline.toggle()` / `await pipeline.stop_recording_and_process()`) using `stt.transcribe` instead of `start_continuous_session()` and `open_session()`.
  - Update `tests/test_stt_nospeech.py` to test `TranscribeSTT.transcribe()` skipping `self._model.session()` on low-energy noise via `_has_min_voiced_duration`.
  - Add `test_pipeline_logs_user_transcript_ptt` to `tests/test_audio_loop_states.py` and `test_floor_can_announce_only_when_idle_and_not_interrupted` to `tests/test_ledger_and_floor.py`.

- [ ] **Step 2: Delete continuous-only modules and test files**
  - Delete `src/verbix/voice/aec.py`, `src/verbix/voice/sherpa_vad.py`, `src/verbix/voice/vad.py`.
  - Delete `tests/test_conversational_voice_loop.py`, `tests/test_aec_barge_in.py`, `tests/test_sherpa_vad.py`, `tests/test_incremental_stt.py`, `tests/test_stt_session_contract.py`.

- [ ] **Step 3: Remove continuous-only code from `voice/` and `runtime/` and wire PTT continuation stash**
  - In `src/verbix/voice/pipeline.py`: remove continuous-mode methods/fields, call `self._stash_interruption(snapshot)` in `toggle()` on playback interrupt, and apply `pending_continuation` on bare `"continue"` / `"go on"` in `stop_recording_and_process()`.
  - In `src/verbix/voice/transcribe_stt.py` & `src/verbix/voice/base.py`: remove `_Segment`, `STTSession`, `TranscribeSession`, `open_session()`, and `STTSessionProtocol`.
  - In `src/verbix/voice/output.py`, `src/verbix/voice/callback_sink.py`, `src/verbix/voice/capture.py`: remove `on_audio`/`_emit_ref`/`_notify_ref`, `on_played`/`_played_queue`/`_played_thread`, and `on_chunk`.
  - In `src/verbix/runtime/floor.py`, `src/verbix/runtime/announcer.py`, `src/verbix/runtime/runtime.py`: remove `user_speaking`, `continuous_mode`, `set_user_speaking()`, and `buffer_window`; make `can_announce()` check `self._state == FloorState.IDLE and not self.interrupt_requested` and `acquire()` exit to `FloorState.IDLE`.

- [ ] **Step 4: Update remaining test files to remove deleted symbol references**
  - Update `tests/test_mic_dump.py`, `tests/test_barge_in_heard.py`, `tests/test_capture_warmup.py`, `tests/test_sounddevice_capture.py`, `tests/test_speech_output.py`, `tests/test_transcribe_stt.py`, `tests/test_runtime_e2e.py`, `tests/test_idle_drained_announcements.py`, `tests/test_voice_loop_barge.py`, `tests/test_ipc_and_cli.py`.

- [ ] **Step 5: Run full test suite and linter and commit**
  - Run: `uv run pytest -q && uv run ruff check src tests`
  - Expected: All tests pass, 0 lint errors.

---

### Task 2: Add CLI `--voice-mode` Error Shim, Remove `pywebrtc-audio` & `silero_vad.onnx`, and Update Docs

**Files:**
- Delete: `LICENSES/mit-silero.md`
- Modify: `src/verbix/cli.py`, `pyproject.toml`, `uv.lock`, `scripts/fetch_voice_models.py`, `THIRD-PARTY-NOTICES.md`, `README.md`
- Test: `tests/test_ipc_and_cli.py`, `tests/test_fetch_voice_models.py`

**Interfaces:**
- Produces:
  - Temporary `--voice-mode` shim in `src/verbix/cli.py` that fails with exit code `2` and `"Error: Continuous voice mode has been removed. Verbix uses push-to-talk (--voice).\n"` on `stderr` when `args.voice_mode != "ptt"`.
  - `MANIFEST` in `scripts/fetch_voice_models.py` containing only `"tts"` and `"asr"`.

- [ ] **Step 1: Write failing tests in `tests/test_ipc_and_cli.py` and `tests/test_fetch_voice_models.py`**
  - In `tests/test_ipc_and_cli.py`: add `test_cli_voice_mode_continuous_fails_with_clear_error` asserting `async_main(build_parser().parse_args(["--voice", "--voice-mode", "continuous"]))` returns `2` and writes `"Continuous voice mode has been removed"` to `stderr`.
  - In `tests/test_fetch_voice_models.py`: assert `"vad" not in mod.MANIFEST` and update `fake_manifest` to `"tts"` and `"asr"`.

- [ ] **Step 2: Run tests to verify they fail**
  - Run: `uv run pytest tests/test_ipc_and_cli.py::test_cli_voice_mode_continuous_fails_with_clear_error tests/test_fetch_voice_models.py -v`
  - Expected: FAIL.

- [ ] **Step 3: Implement CLI shim, remove `pywebrtc-audio` and `silero_vad.onnx`, and update docs**
  - Update `src/verbix/cli.py` to remove the continuous loop branch and add the `--voice-mode != "ptt"` error check (`# TODO(release): remove --voice-mode shim in a later release`).
  - Remove `"pywebrtc-audio>=0.2.0"` from `pyproject.toml` and run `uv sync`.
  - Remove `"vad"` from `scripts/fetch_voice_models.py`, delete `LICENSES/mit-silero.md`, and update `THIRD-PARTY-NOTICES.md` and `README.md`.

- [ ] **Step 4: Run full test suite, linter, and symbol-removal grep, then commit**
  - Run: `uv run pytest -q && uv run ruff check src tests`
  - Expected: All tests pass, 0 lint errors, 0 references to removed symbols.
