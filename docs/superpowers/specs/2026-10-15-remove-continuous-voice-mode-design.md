# Design Spec: Remove Continuous Voice Mode (Push-to-Talk Only)

**Date:** 2026-10-15  
**Pre-deletion Git Tag:** `continuous-mode-v1` (at commit `0cf15ec`)  
**Status:** Ready for Review

---

## 1. Objective & Scope

Remove continuous (hands-free VAD + AEC) voice mode from Verbix so that push-to-talk (PTT) is the sole voice interaction mode. All infrastructure that serves PTT or background task announcements is preserved without behavior changes; all continuous-only code paths, dependencies, model weights, and tests are removed cleanly without stubs or dead flags.

---

## 2. Classification: Continuous-Only vs. Shared Infrastructure

### A. Continuous-Only (To Be Deleted)

| File / Symbol | Lines | Why It Is Continuous-Only |
| :--- | :---: | :--- |
| `src/verbix/voice/aec.py` (`EchoCanceller`) | 124 | WebRTC acoustic echo cancellation (`pywebrtc-audio`). PTT halts speaker playback before opening the microphone (`start_recording`), so speaker echo cancellation is never used in PTT. |
| `src/verbix/voice/sherpa_vad.py` (`SileroSegmenter`) | 167 | Silero VAD streaming speech segmenter (`silero_vad.onnx`). PTT recording boundaries are controlled explicitly by user toggle (`start_recording` / `stop_recording_and_process`). |
| `src/verbix/voice/vad.py` (`EnergyVAD`) | 68 | Energy-based VAD segmenter fallback for continuous mode. |
| `src/verbix/voice/pipeline.py`:<br/>- `is_continuous_active`, `start_continuous_session()`, `stop_continuous_session()`<br/>- `_interrupt_active_speech()`, `_is_speaking`, `_on_speech_endpoint()`<br/>- `_setup_turn_listening()`, `_run_continuous_loop()`<br/>- `_feed_speaker_reference()`, `aec` property/setter<br/>- `_live_stt`, `_continuous_task`, `_speech_endpoint_event`, `_gate_stt_feed`, `_barge_in_triggered`, `_loop`, `_mic_lookback`, `_tail_lookback`, `_dump_buf` | ~360 | Continuous VAD turn loop, acoustic barge-in detector, 300ms mic lookback/tail buffers, and AEC reference callback. |
| `src/verbix/voice/transcribe_stt.py`:<br/>- `_Segment`, `STTSession` (`lines 24-152`)<br/>- `TranscribeSession` (`lines 210-317`)<br/>- `TranscribeSTT.open_session()` (`lines 349-368`) | ~255 | Incremental VAD-span session wrapper. In PTT mode, `PTTCapture.stop_recording()` already returns the complete recorded `audio_bytes`, which `stop_recording_and_process()` passes directly to `TranscribeSTT.transcribe(audio_bytes, sample_rate)`. |
| `src/verbix/voice/base.py`:<br/>- `STTSessionProtocol` (`lines 17-51`) | 35 | Protocol for incremental VAD-driven STT sessions. `SpeechToText`, `TextToSpeech`, `AudioSink`, and `Mock*` classes remain. |
| `src/verbix/voice/output.py`:<br/>- `on_audio` parameter (`line 91, 96-100`)<br/>- `_sink_handles_ref`, `_emit_ref()`, `_notify_ref()` (`lines 306, 367-379, 459, 469`) | ~25 | Reference-audio tap that existed solely to feed played PCM into `EchoCanceller.feed_reference()`. |
| `src/verbix/voice/callback_sink.py`:<br/>- `on_played` parameter & property (`lines 55, 73-77, 88, 91-121`)<br/>- `_played_queue`, `_played_thread`, `_drain_played()`, `_stop_played_worker()`, queue enqueue in `_callback` (`lines 174-176, 187`) | ~45 | Background worker thread and queue that existed solely to forward DAC-dequeued PCM chunks to `SpeechOutput.on_audio` → `EchoCanceller`. |
| `src/verbix/voice/capture.py`:<br/>- `on_chunk` callback parameter (`lines 58, 62, 134-138`) | ~10 | Live per-chunk callback into `STTSession` / `_feed_chunk`. PTT reads the buffered recording returned by `await capture.stop_recording()`. |
| `src/verbix/runtime/floor.py`:<br/>- `user_speaking`, `continuous_mode`, `set_user_speaking()` (`lines 29-30, 67-74`)<br/>- `FloorState.LISTENING` branch in `can_announce()` and `acquire()` (`lines 80, 92-95`) | ~20 | Sticky VAD `user_speaking` latch and announcing while in continuous `LISTENING`. In PTT, `FloorState.LISTENING` means the user's mic is actively recording, so `can_announce()` is strictly `self._state == FloorState.IDLE and not self.interrupt_requested`, and `acquire()` always exits to `FloorState.IDLE`. |
| `src/verbix/runtime/runtime.py`:<br/>- `buffer_window: float = 0.3` parameter in `Runtime.__init__` (`line 100`) | ~3 | Unused quiet-window parameter. PTT announcements drain immediately whenever `floor.is_idle and not floor.interrupt_requested` (no `buffer_window` or `last_user_speech_at` timer is used). |
| `scripts/fetch_voice_models.py`:<br/>- `MANIFEST["vad"]` (`silero_vad.onnx`, `lines 72-82`) and `vad_dir` in `fetch_models()` | ~20 | Downloads `silero_vad.onnx`, which is unused without `SileroSegmenter`. |

---

### B. Shared Infrastructure (Preserved for PTT & Announcements)

1. **PTT Toggle, Enter-Key Barge-In, and Continuation Stash (`src/verbix/voice/pipeline.py`)**:
   - `VoicePipeline.toggle()`:
     - If `self.capture.is_recording`: calls `await self.stop_recording_and_process()`.
     - If `self._speech_output.is_playing or not self.runtime.is_idle`:
       1. Calls `snapshot = self._speech_output.interrupt()` to immediately halt speaker playback and capture the 3-way `InterruptionSnapshot`.
       2. Calls `self._stash_interruption(snapshot)` to store `snapshot.unplayed_text` in `self.runtime.pending_continuation`.
       3. Calls `await self.runtime.cancel_turn(heard=heard)` to trim `foreground.messages` to `confirmed_heard`.
       4. Calls `await self.start_recording()` and returns `{"action": "interrupted_and_recording"}`.
     - If idle: calls `await self.start_recording()` and returns `{"action": "recording"}`.
   - `VoicePipeline.stop_recording_and_process()`:
     - Stops capture (`audio_bytes = await self.capture.stop_recording()`), plays the earcon cue (`play_cue()`), transcribes `audio_bytes` via `self.stt.transcribe(audio_bytes)` (when `len(audio_bytes) >= 1600`), writes the optional `--dump-mic-dir` WAV (`_dump_wav`), and if `transcribed_text` is empty returns `FloorState.IDLE` without calling the model.
     - Preserves the bare `"continue"` / `"go on"` continuation consumption (`if norm in ("continue", "go on") and getattr(self.runtime, "pending_continuation", None):`) before spawning `self.execute_voice_turn(transcribed_text)`.
2. **Sentence Streaming, Sample-Accurate Interruption Snapshots, and Earcon Cue (`src/verbix/voice/output.py`)**:
   - `SpeechOutput`, `InterruptionSnapshot` (`confirmed_heard`, `interrupted_sentence`, `unplayed_text`), warm sentence prefetch (`_WARM_DEPTH = 2`), `_sent_spans`, `sink.played_samples` accounting, `interrupt()`, `delivered_text()`, `reset()`, and `play_cue()` remain intact.
3. **Click-Free Fade-Out & `played_samples` Accounting (`src/verbix/voice/callback_sink.py`)**:
   - `CallbackAudioSink`, `apply_fade()`, 10ms fade-out on `stop()`, and `played_samples` counter remain intact.
4. **DC-Offset / High-Pass Mic Conditioning (`src/verbix/voice/capture.py`)**:
   - `PTTCapture` and `_condition_s16le()` (60 Hz one-pole high-pass filter removing DMIC DC offset so `_has_min_voiced_duration` in `TranscribeSTT` works reliably) remain intact.
5. **Single-Pass GGUF ASR & Click Rejection (`src/verbix/voice/transcribe_stt.py`)**:
   - `TranscribeSTT`, `find_default_stt_model()`, and `_has_min_voiced_duration()` (150ms / 300 RMS gate rejecting Enter-key clicks without invoking neural ASR) remain intact.
6. **Floor States & Announcement Delivery (`src/verbix/runtime/floor.py`, `src/verbix/runtime/ledger.py`, `src/verbix/runtime/announcer.py`, `src/verbix/runtime/runtime.py`)**:
   - `FloorState` (`IDLE`, `LISTENING`, `SPEAKING`), `FloorArbiter.acquire()`, `request_interrupt()`, `TaskLedger` (`delivery` + `announce_attempts` cap), `drain_pending_announcements()`, and `Runtime._foreground_grounding()` remain intact.

---

## 3. Dependencies & Model Assets Leaving

1. **Python Dependency Leaving (`pyproject.toml` & `uv.lock`)**:
   - **`pywebrtc-audio>=0.2.0`** (`BSD-3-Clause`): Removed from `pyproject.toml` and `THIRD-PARTY-NOTICES.md`.
2. **Model Asset Leaving (`scripts/fetch_voice_models.py`, `THIRD-PARTY-NOTICES.md`, `LICENSES/`)**:
   - **`silero_vad.onnx`** (`MANIFEST["vad"]` in `scripts/fetch_voice_models.py`): Removed from `MANIFEST` and `fetch_models()`.
   - **`THIRD-PARTY-NOTICES.md` & `LICENSES/mit-silero.md`**: Remove the Silero VAD and `pywebrtc-audio` sections from `THIRD-PARTY-NOTICES.md` (and remove `LICENSES/mit-silero.md` and the retired Canary 180M Flash note).
3. **Platform & Python-Version Notes Re-evaluation (`pyproject.toml`, `README.md`, `.github/workflows/ci.yml`)**:
   - Removing `pywebrtc-audio` drops the WebRTC C++ extension wheel constraint.
   - Remaining native voice packages (`transcribe-cpp-native==0.4.0` and `sherpa-onnx==1.13.8`) publish prebuilt wheels for CPython 3.13 and 3.14 across Linux (`x86_64`), macOS (`arm64`), and Windows (`amd64`), so `requires-python = ">=3.13,<3.15"` in `pyproject.toml` and the 3-OS CI matrix in `.github/workflows/ci.yml` remain accurate.

---

## 4. CLI, Config, Env Vars & Documentation Behavior

1. **Graceful CLI Rejection of Continuous Mode (`src/verbix/cli.py`)**:
   - Keep `--voice-mode` in `build_parser()` as a temporary clear-error shim (`# TODO(release): remove --voice-mode shim in a later release`) with `default="ptt"` (accepting any string so `argparse` does not emit an obscure choice error) and help text: `"Voice interaction mode (only 'ptt' is supported; continuous mode has been removed)."`.
   - Early in `async_main()`, if `getattr(args, "voice_mode", "ptt") != "ptt"`: write a clear message to `sys.stderr`:
     `"Error: Continuous voice mode has been removed. Verbix uses push-to-talk (--voice).\n"`
     and return exit code `2`.
2. **Audit of Other CLI Flags, Config Keys, and Environment Variables**:
   - **CLI flags (`src/verbix/cli.py`)**: `--dump-mic-dir` serves PTT per-turn WAV dumps and stays; no other continuous-only CLI flags exist.
   - **Config keys (`src/verbix/config.py`)**: `_ALLOWED_KEYS = ("provider", "stt_model")` contains no continuous-only keys.
   - **Environment variables**: No continuous-only environment variables exist (`VERBIX_STT_MODEL` and `VERBIX_VOICE_DIR` serve PTT).
   - **Protocols (`src/verbix/voice/base.py`)**: `SpeechToText` protocol has only `transcribe(audio_data, sample_rate=16000) -> str` (no `open_session`); `STTSessionProtocol` is deleted and `open_session` is removed from all test doubles.
3. **`README.md`**:
   - Document push-to-talk (`uv run verbix --voice`) as the single voice mode and remove continuous-mode and `buffer_window` notes.

---

## 5. Complete Grep-Verified Test Suite Migration Plan (18 Files)

### Tests to Delete (5 Files — 100% Continuous-Only)
1. `tests/test_conversational_voice_loop.py` (porting PTT transcript logging to `tests/test_audio_loop_states.py`)
2. `tests/test_aec_barge_in.py`
3. `tests/test_sherpa_vad.py`
4. `tests/test_incremental_stt.py`
5. `tests/test_stt_session_contract.py`

### Tests to Port / Update (13 Files — Preserving PTT & Shared Coverage)
1. `tests/test_stt_nospeech.py`: Port `test_vad_gate_skips_model_on_low_energy_noise` from `TranscribeSession` to `TranscribeSTT.transcribe()` (verifying `_has_min_voiced_duration` skips the neural ASR model on low-energy noise in PTT).
2. `tests/test_draft_shelf.py`: Port all continuation-stash tests (`test_stash_then_continue_round_trips_on_real_runtime`, `test_empty_transcript_stays_silent_and_keeps_snapshot`, `test_interrupt_stashes_unplayed_outside_history`, `test_continue_hands_stashed_text_to_agent_and_clears`, `test_continue_variants_match_exactly_case_insensitive`, `test_near_continue_does_not_consume_stash`, `test_other_utterance_leaves_stash_alone`, `test_newer_interrupt_overwrites_stash`) to PTT (`pipeline.toggle()` / `stop_recording_and_process()`) and remove `open_session` / `feed_barge_in` / `on_chunk`.
3. `tests/test_mic_dump.py`: Keep the 4 PTT WAV dump tests (`test_cli_flag_defaults_off_and_parses`, `test_unset_writes_nothing_ptt`, `test_ptt_dumps_wav_named_for_transcript`, `test_dump_write_failure_never_breaks_turn`); remove `FakeLiveSession`, `open_session`, `on_chunk`, `test_continuous_dumps_finalized_turn_bytes`, and `test_barge_in_dumps_lookback_seed`.
4. `tests/test_barge_in_heard.py`: Remove unused `_StubSession` and `_StubSTT` (`open_session` / `feed_barge_in`).
5. `tests/test_capture_warmup.py`: Remove `test_on_chunk_cleared_on_session_end` (`on_chunk` and `_live_stt`).
6. `tests/test_sounddevice_capture.py`: Remove `on_chunk` argument from `PTTCapture` calls in `test_capture_lifecycle_with_sounddevice` and `test_audio_callback_never_raises_after_loop_teardown`.
7. `tests/test_speech_output.py`: Keep all sentence streaming, prefetch, earcon cue, and `played_samples` interruption snapshot tests (`lines 52-255`); remove `_RecordingAEC`, `on_chunk`, and the 5 `on_audio` reference-hook tests (`lines 28-50, 257-359`).
8. `tests/test_fetch_voice_models.py`: Remove `vad` (`silero-vad`) from `fake_manifest` and assertions.
9. `tests/test_transcribe_stt.py`: Keep `find_default_stt_model` tests; remove `TranscribeSession` and `open_session` tests (`lines 145-198`).
10. `tests/test_runtime_e2e.py`: Remove `buffer_window=...` from `Runtime(...)` calls (`lines 66, 101, 164, 298`) and update `test_scripted_live_session_replay_has_no_duplicate_spawns_and_no_double_spoken_results` (`lines 665-706`) to use PTT floor states (`FloorState.LISTENING` while recording, `FloorState.IDLE` when idle between turns, with no `set_user_speaking` calls).
11. `tests/test_idle_drained_announcements.py`: Remove `on_chunk` and `_aec` checks from `test_ptt_interrupted_announcement_with_short_capture_is_not_retried_and_folds_into_next_turn`.
12. `tests/test_ipc_and_cli.py` & `tests/test_voice_loop_barge.py`: Remove `buffer_window=0.02`, `fake_pipe.is_continuous_active`, and `FakeCapture.on_chunk`, and add `test_cli_voice_mode_continuous_fails_with_clear_error`.
13. `tests/test_audio_loop_states.py` & `tests/test_ledger_and_floor.py`: Port `test_pipeline_logs_user_transcript_ptt` into `tests/test_audio_loop_states.py` and add `test_floor_can_announce_only_when_idle_and_not_interrupted` in `tests/test_ledger_and_floor.py`.
