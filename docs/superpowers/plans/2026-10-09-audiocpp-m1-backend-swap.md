# M1 audio.cpp Backend Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap Verbix's voice backends to a resident audio.cpp server (Silero VAD gate, ASR transcription, Supertonic TTS) with push-to-talk behavior unchanged.

**Architecture:** A Python provisioner installs binaries + weights once; a supervisor owns the resident server process; three thin HTTP clients implement the existing `voice/base.py` protocols; pipeline loses only its live-STT branch.

**Tech Stack:** Python 3.13, httpx2, numpy, stdlib `wave`, sounddevice (optional `audio` extra), audio.cpp TAG `v0.9.1` server + GGUF weights.

**Spec:** `docs/superpowers/specs/2026-10-09-audiocpp-voice-backend-swap-design.md` — the plan argues from the spec; executors read both. Spec §N references below are to that file.

## Global Constraints

- audio.cpp release TAG `v0.9.1`; per-OS artifact SHA256 pinned statically in the provisioner manifest, verified on every download.
- No new Python dependencies: httpx2, numpy, stdlib `wave` only; sounddevice via the pre-existing optional `audio` extra.
- VAD threshold `0.5`, tunables default; STT `language="en"`; TTS voice `voice_id "M1"` (provisional); capture 16 kHz mono s16le.
- Timeouts: health 10 s total; VAD 15 s; STT `max(30 s, 4× duration)`; TTS 60 s/sentence. httpx2 carries them; timeouts enter the transport bucket.
- Error taxonomy (§8): transport/timeout → setup guidance; unknown-model message (any status) → setup guidance; other 4xx + 503 busy/insufficient-memory → error contract, NO respawn; other 5xx → one respawn, then contract.
- ASR stays `UNRESOLVED` until a package id is pinned; legal sign-off blocks release.
- Full suite + `ruff check` green before any completion claim; `set -o pipefail`, never bare-pipe `| tail`.

## Review Focus

- 30 s utterance through the VAD gate + STT in one HTTP call each: must transcribe fully, not truncate at body limits.
- Rapid double-toggle (open/close/open): second open must get a healthy server, never a half-dead port.
- Non-16 kHz device delivery (e.g. 48 kHz): must resample to the 16 kHz contract, never pass through silently.
- Non-English speech with `language="en"` pinned: must return English-oriented text (or empty), never raise.
- Server 503 `server_busy` under back-to-back turns: must surface the error contract, never respawn-loop.
- (Each line's test is added to the owning task below: long-audio → Task 6; double-toggle → Task 4; rate mismatch → Task 8; non-English → Task 6; 503 → Task 4.)

## File Structure

- Create `scripts/provision_voice.py` — one-command provision (stdlib only); importable layout + step functions (precedent: `fetch_tts_assets.py` path-insert).
- Create `src/verbix/voice/audiocpp_server.py` — `AudioCppServer(binary, config)`: `load()`, `aclose()`, `is_loaded`, `base_url`. No auth; binds 127.0.0.1 via `server.json`.
- Create `src/verbix/voice/audiocpp_vad.py` — `AudioCppVAD(base_url, model_id, threshold=0.5)`: `async is_speech(pcm) -> bool` via `POST /v1/tasks/run`.
- Create `src/verbix/voice/audiocpp_stt.py` — `AudioCppSTT(base_url, model_id, language="en")`: `async transcribe(audio, sample_rate=16000) -> str` via multipart `POST /v1/audio/transcriptions`.
- Create `src/verbix/voice/audiocpp_tts.py` — `AudioCppTTS(base_url, model_id, voice_id="M1")`: async-generator `synthesize(text)` yielding 24 kHz f32 chunks via `POST /v1/audio/speech`.
- Create `src/verbix/voice/wavutil.py` — `to_wav_bytes(pcm, sample_rate=16000, channels=1, width=2)`, `strip_wav(data)`, `ensure_f32_24k(pcm, sample_rate)`.
- Modify `src/verbix/voice/capture.py` — sounddevice backend last in detection order; hatch-open log line.
- Modify `src/verbix/voice/pipeline.py` — delete live branch; pre-submit failures return the error dict.
- Modify `src/verbix/voice/base.py` (mocks only), `src/verbix/voice/__init__.py` (re-exports), `src/verbix/cli.py` (wiring + error branches), `src/verbix/ipc/client.py` (raise on error status), `src/verbix/ipc/server.py` (toggle error mapping).
- Modify `pyproject.toml` + `uv.lock` (drop faster-whisper), `THIRD-PARTY-NOTICES.md`, create `LICENSES/` + `DISTRIBUTION-TERMS.md`, update `README.md` map.
- Delete: `voice/stt.py`, `voice/pocket_helper.py`, `voice/tts_assets.py`, three PocketTTS scripts, `tests/test_incremental_stt.py`.
- Rewrite: `tests/test_pocket_helper.py` (new-clients suite), `tests/test_tts_assets.py` (provisioner), `tests/test_stt_nospeech.py` (new STT), `tests/test_voice_loop_barge.py` (re-import), `tests/test_capture_warmup.py` (drop `_live_stt`).
- Create: `tests/fixtures/voice/` corpus (10 spec sentences), `scripts/wer.py` (word-split WER, validation-only).

---

### Task 1: Baselines + corpus + WER harness (BEFORE any deletion)

**Files:**
- Create: `tests/fixtures/voice/*.wav` (10 spec §5 sentences, recorded once on dev mic), `scripts/wer.py`
- Test: `tests/test_voice_corpus.py` (new, asserts the 10 fixtures exist and are 16 kHz mono)

**Interfaces:**
- Consumes: nothing (uses current `FasterWhisperSTT`, `PocketHelperEngine`).
- Produces: corpus paths + `wer.py: word_wer(ref, hyp) -> float` used by Task 11; baseline numbers below.

- [ ] **Step 1: Write `tests/test_voice_corpus.py`** asserting 10 named files exist under `tests/fixtures/voice/` and each is 16 kHz mono s16le (stdlib `wave`).
- [ ] **Step 2: Run it to verify it fails** — `uv run pytest tests/test_voice_corpus.py -v`, missing files.
- [ ] **Step 3: Record the corpus**, commit the WAVs, write `scripts/wer.py` with `def word_wer(ref: str, hyp: str) -> float` (word-split, `difflib.SequenceMatcher` opcodes).
- [ ] **Step 4: Measure baselines on this machine** (tiny.en WER via `wer.py`, transcription RTF, stop-to-transcript max over K=3, PocketTTS request-to-complete, silence/noise emptiness). Record every number + machine string + backend in the commit message.
- [ ] **Step 5: Run full check** — `uv run pytest tests/test_voice_corpus.py -v` PASS; `uv run ruff check src tests` clean.
- [ ] **Step 6: Commit** — `git add tests/fixtures/voice scripts/wer.py tests/test_voice_corpus.py && git commit -m "test: M1 validation corpus, WER harness, baselines (see message)"`.

### Task 2: Provisioner + live server + task-1 confirmations

**Files:**
- Create: `scripts/provision_voice.py`
- Test: rewrite `tests/test_tts_assets.py` → `tests/test_provision_voice.py` (layout paths, SHA-verify logic, server.json contents, gate logic — all against temp dirs + fakes, no downloads)

**Interfaces:**
- Consumes: nothing.
- Produces: layout `{root}/bin/audiocpp_server`, `{root}/models/<family>/`, `{root}/server.json` (bind 127.0.0.1, model entries incl. `silero-vad`), `provision(root, backend) -> paths`; `scripts/` importable pattern.

- [ ] **Step 1: Write failing tests** for layout resolution (`VERBIX_VOICE_DIR` override → cwd `voice-runtime` → `~/.verbix/voice`), SHA mismatch raising, `server.json` containing the `silero-vad` entry + bind address.
- [ ] **Step 2: Run to verify they fail** — `uv run pytest tests/test_provision_voice.py -v`.
- [ ] **Step 3: Implement `scripts/provision_voice.py`** (TAG `v0.9.1`, static SHA manifest, `--backend cpu|cuda|vulkan|metal` default auto, `--force-rebuild`; installs `verbix[audio]`; Python v2 model manager; Fedora rule per spec §6; exit 0 only after health + VAD/ASR/TTS round-trips).
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Run it live on Fedora, record the four task-1 confirmations into the spec** (VAD segment shape, Supertonic native rate/channels, WAV header size, `voice_id` JSON field). Amend the spec file with the values.
- [ ] **Step 6: Commit** — provisioner + tests + spec amendment.

### Task 3: wavutil

**Files:**
- Create: `src/verbix/voice/wavutil.py`
- Test: `tests/test_wavutil.py`

**Interfaces:**
- Consumes: nothing (stdlib `wave`, numpy).
- Produces: `to_wav_bytes(pcm, sample_rate=16000, channels=1, width=2) -> bytes`; `strip_wav(data) -> bytes`; `ensure_f32_24k(pcm, sample_rate) -> bytes` — used by Tasks 5–7.

- [ ] **Step 1: Write failing tests** — round-trip PCM↔WAV at 16 kHz mono s16le; strip recovers exact payload; 48 kHz→24 kHz conversion length math.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement the three functions** (16 kHz mono s16le capture contract).
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Commit.**

### Task 4: Supervisor

**Files:**
- Create: `src/verbix/voice/audiocpp_server.py`
- Test: `tests/test_audiocpp_server.py` (stub binary: a Python one-liner HTTP server answering `/health`; dead-binary path)

**Interfaces:**
- Consumes: provisioned `(binary, config)` paths from Task 2.
- Produces: `AudioCppServer(binary, config)` with `async load()`, `async aclose()`, `is_loaded: bool`, `base_url: str` — used by Tasks 5–7. Double-toggle test lives here (Review Focus: reopen after close gets a healthy server).

- [ ] **Step 1: Write failing tests** — missing binary raises with the provision command; `load()` waits for `/health` (10 s bound); kill mid-life → `load()` respawns healthy; close→open cycle healthy.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `AudioCppServer`** (spawn `[binary, --config config]`, poll health, no auth).
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Commit.**

### Task 5: VAD client

**Files:**
- Create: `src/verbix/voice/audiocpp_vad.py`
- Test: `tests/test_audiocpp_vad.py` (fake HTTP server asserting byte-level `tasks/run` shape incl. nested `options.threshold`)

**Interfaces:**
- Consumes: `base_url` (Task 4), `wavutil.to_wav_bytes` (Task 3), confirmed segment shape (Task 2).
- Produces: `AudioCppVAD(base_url, model_id, threshold=0.5)` with `async is_speech(pcm) -> bool` — used by Task 9.

- [ ] **Step 1: Write failing tests** — speech WAV → True, silence → False, exact request JSON asserted, 503 → error contract no respawn, timeout (15 s) → transport bucket.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `AudioCppVAD`** (httpx2, `audio_base64` WAV, segments → bool).
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Commit.**

### Task 6: STT client (and ASR pin)

**Files:**
- Create: `src/verbix/voice/audiocpp_stt.py`
- Test: rewrite `tests/test_stt_nospeech.py` (fake server: empty → `""`; multipart shape asserted; `language="en"` asserted)

**Interfaces:**
- Consumes: `base_url` (Task 4), `wavutil` (Task 3).
- Produces: `AudioCppSTT(base_url, model_id, language="en")` with `async transcribe(audio, sample_rate=16000) -> str` — used by Task 9. Long-audio test lives here (Review Focus: 30 s utterance transcribes fully). Non-English test lives here (returns text, never raises).

- [ ] **Step 1: Write failing tests** — transcription text returned; silence → `""`; multipart field names asserted; unknown-model message → setup-guidance error (not respawn); 30 s utterance transcribes fully (Review Focus); non-English audio returns text, never raises (Review Focus).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `AudioCppSTT`** (multipart `POST /v1/audio/transcriptions`).
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Evaluate the §5 shortlist against the Task 1 corpus** (each candidate via `model_id` config), pin the winner by package id in the provisioner manifest. ASR flips `UNRESOLVED`→pinned here.
- [ ] **Step 6: Commit** — client + tests + manifest pin.

### Task 7: TTS client

**Files:**
- Create: `src/verbix/voice/audiocpp_tts.py`
- Test: rewrite `tests/test_pocket_helper.py` → `tests/test_audiocpp_tts.py` (reuse its fake-HTTP-server pattern)

**Interfaces:**
- Consumes: `base_url` (Task 4), `wavutil.strip_wav/ensure_f32_24k` (Task 3), confirmed rate (Task 2).
- Produces: `AudioCppTTS(base_url, model_id, voice_id="M1")` with async-generator `synthesize(text)` yielding 24 kHz f32 chunks — used by Task 9.

- [ ] **Step 1: Write failing tests** — chunks yielded are headerless f32; `options.voice_id` field asserted (never top-level `"voice"`); 60 s timeout; 5xx → one respawn then error.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement `AudioCppTTS`** (`POST /v1/audio/speech`, strip + convert).
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Commit.**

### Task 8: Capture backend + startup gates

**Files:**
- Modify: `src/verbix/voice/capture.py`
- Test: rewrite `tests/test_capture_warmup.py` (drop `_live_stt`; keep on-chunk-clearing, frame-bytes, warm-rate-limit; add backend-order + hatch log-line tests)

**Interfaces:**
- Consumes: `verbix[audio]` extra (installed by Task 2).
- Produces: sounddevice backend last in detection order; 16 kHz mono s16le contract (resample on mismatch); hatch-open log line; no-backend → raise. Rate-mismatch test lives here (Review Focus: 48 kHz device resamples, never passes through).

- [ ] **Step 1: Write failing tests** — detection order `pw-record`→`arecord`→sounddevice (monkeypatched `shutil.which`/import); missing-everything raises with provision command; log line emitted on start; 48 kHz device delivery resamples to the 16 kHz contract (Review Focus).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** (lazy sounddevice import, format request + numpy resample fallback).
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Commit.**

### Task 9: Pipeline + CLI + IPC error contract

**Files:**
- Modify: `src/verbix/voice/pipeline.py`, `src/verbix/cli.py`, `src/verbix/ipc/server.py`, `src/verbix/ipc/client.py`, `src/verbix/voice/base.py` (mocks), `src/verbix/voice/__init__.py` (re-exports)
- Test: rewrite `tests/test_voice_loop_barge.py` (re-import, assertions unchanged); new `tests/test_voice_error_contract.py`

**Interfaces:**
- Consumes: `AudioCppVAD/STT/TTS` (Tasks 5–7), supervisor (Task 4).
- Produces: hatched-PTT flow on new backends; `{"action":"error"}` dict from toggle with floor `IDLE`; CLI prints + continues; IPC `status="error"` mapping; client raise path.

- [ ] **Step 1: Write failing tests** — pre-submit failure returns the §3 dict at both call sites, floor `IDLE`, no turn spawned; daemon toggle raise path; background failure logs with floor `IDLE`; pipeline never touches `open_session`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** — delete `pipeline.py:85-88` + finalize-preference branch; wire new clients in `cli.py` voice block including the provisioning check and the capture/sink startup gates (raise with guidance, making silent paths unreachable); `__getattr__` re-exports; mocks extended; `client.toggle()` raises on error status; daemon message generalized.
- [ ] **Step 4: Run tests** — PASS; `ruff check` clean.
- [ ] **Step 5: Commit.**

### Task 10: Cutover (deletions, deps, docs, licenses)

**Files:**
- Delete: `src/verbix/voice/stt.py`, `src/verbix/voice/pocket_helper.py`, `src/verbix/voice/tts_assets.py`, `scripts/setup_tts_runtime.sh`, `scripts/build_tts_helper.sh`, `scripts/fetch_tts_assets.py`, `tests/test_incremental_stt.py`
- Modify: `pyproject.toml` + `uv.lock` (drop faster-whisper, regen), `THIRD-PARTY-NOTICES.md`, `README.md` (module map)
- Create: `LICENSES/` (copies), `DISTRIBUTION-TERMS.md` (repo root)

**Interfaces:**
- Consumes: everything above (suite must be green before AND after).
- Produces: tree with no PocketTTS/faster-whisper references (`grep` clean); complete license set.

- [ ] **Step 1: Verify pre-delete green** — `uv run pytest -q` + `uv run ruff check src tests`.
- [ ] **Step 2: Execute deletions + dep drop + lockfile regen**; add license files + notices entry; update README map.
- [ ] **Step 3: Verify post-delete green** — full suite + `ruff`; `grep -rn "pocket_helper\|faster_whisper\|PocketHelperEngine\|FasterWhisperSTT\|tts_assets\|setup_tts_runtime" src tests scripts | grep -v Binary` returns nothing (excluding the plan-specified historical mentions, of which there are none in code).
- [ ] **Step 4: Commit.**

### Task 11: Validation + exit gate (Fedora-green)

**Files:** none (measurements only; numbers recorded in the commit message)

- [ ] **Step 1: Run the provisioner for real** — exit 0 with health + three round-trips on Fedora.
- [ ] **Step 2: Score the corpus** — WER vs Task 1 baselines via `scripts/wer.py` (median, K=3), RTF, stop-to-transcript max, TTS latency ratio, silence emptiness.
- [ ] **Step 3: Check every §5 bar** — WER ≤ tiny.en; RTF ≤ 0.5 CPU; p95/max ≤ 3 s; TTS ≤ 1.25×; silence empty. Any miss → fix or re-pin (ASR package, TTS package) and re-run; spec amended with the pinned ids. Confirm both §7 blockers are recorded in their final state (ASR pinned here; voice provenance + legal sign-off still open by design).
- [ ] **Step 4: End-to-end voice turns** — PTT turn, barge-in, error paths (kill server mid-turn → contract observed).
- [ ] **Step 5: Commit the validation record** (spec amendment with pinned ids + results; empty commit message body forbidden — numbers in the message).
