# sherpa-onnx voice swap — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace faster-whisper STT + bundled Silero VAD + PocketTTS.cpp helper with sherpa-onnx, in-process, one dependency.

**Architecture:** `sherpa-onnx` pip package runs TTS (Supertonic) and ASR (Canary) inside the Python process. The `voice/base.py` protocols are unchanged, so the pipeline, floor, runtime, CLI and IPC are untouched. All helper-process supervision is deleted.

**Tech Stack:** Python 3.13, `sherpa-onnx` (compiled wheel, ships Linux/macOS/Windows), ONNX INT8 models.

**Spec:** `docs/superpowers/specs/2026-10-09-voice-stack-decision.md` (engineer approved).

## Global Constraints

- `voice/base.py` protocols (`SpeechToText`, `TextToSpeech`, `AudioSink`) stay byte-identical — the flow must not change.
- TTS voice: Supertonic 3 `M1`, INT8 (`sherpa-onnx-supertonic-3-tts-int8-2026-05-11`), provisional pending the OpenRAIL-M gate.
- ASR: Canary 180M Flash INT8 (`sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8`). Dev-only, not bundled — no license gate.
- Both loaded in-process via `sherpa_onnx` — no subprocess, no HTTP, no supervisor, no token.
- Models download once to a single root (VERBIX_VOICE_DIR), fetched + checksum-verified by our own script.
- Deletions same change: `voice/pocket_helper.py`, `voice/tts_assets.py`, `scripts/setup_tts_runtime.sh`, `scripts/build_tts_helper.sh`, `scripts/fetch_tts_assets.py`.
- Full suite + `ruff check` green before any completion claim; `set -o pipefail`.
- Lazy imports only (`sherpa_onnx` imported inside constructors, like `sounddevice` today).

## Review Focus

- Model files missing at runtime → loud error naming the fetch command.
- `sherpa_onnx` import failure (no wheel for platform) → loud error, no silent fallback to a deleted backend.
- TTS chunk streaming must keep sentence-level latency: first chunk within ~250 ms of call.
- ASR on noise/silence must not hallucinate into a turn.
- Windows/macOS wheel availability for the pinned sherpa-onnx version (verify on PyPI, do not assume).

---

### Task 1: INT8 quality spike (decision input — do first)

**Files:** none committed. Scratch output to `/tmp/opencode/`.

- [ ] **Step 1:** `uv add sherpa-onnx` in a scratch venv (`/tmp/opencode/sherpa-spike`), pull both INT8 models, run the TTS on one sentence, save WAV to `/tmp/opencode/supertonic_int8.wav`. Record size + first-chunk ms.
- [ ] **Step 2:** Pull fp32 Supertonic ONNX, same sentence, save `/tmp/opencode/supertonic_fp32.wav`.
- [ ] **Step 3:** Run Canary INT8 on 5 spoken sentences (record with `pw-record`), record transcripts.
- [ ] **Step 4:** Stop. Engineer listens to both WAVs and reads the transcripts. Decision: INT8 or fp32 for shipping.

Report: numbers + file paths. No code committed.

### Task 2: Model fetch + provenance

**Files:**
- Create: `scripts/fetch_voice_models.py`
- Test: `tests/test_fetch_voice_models.py`

**Interfaces:**
- Produces: `fetch(root) -> Paths` — downloads pinned model dirs, verifies SHA256 from a static manifest, writes `provenance.json` (model id, license, source URL, date).

- [ ] **Step 1:** Write failing tests — layout resolves under VERBIX_VOICE_DIR; checksum mismatch raises; provenance file written.
- [ ] **Step 2:** Run to verify they fail.
- [ ] **Step 3:** Implement fetch (stdlib `urllib`, static SHA manifest, `huggingface.co` sources).
- [ ] **Step 4:** Run tests to verify they pass; `ruff check` clean.
- [ ] **Step 5:** Commit.

### Task 3: TTS client (Supertonic via sherpa-onnx)

**Files:**
- Create: `src/verbix/voice/sherpa_tts.py`
- Test: `tests/test_sherpa_tts.py`

**Interfaces:**
- Consumes: `voice/base.py` `TextToSpeech` protocol; `fetch()` paths from Task 2.
- Produces: `SherpaTTS(model_dir, voice_id="M1")` with async-generator `synthesize(text)` yielding 24 kHz f32 chunks (matching today's TTS contract exactly).

- [ ] **Step 1:** Write failing tests — fake `sherpa_onnx` module injected via `sys.modules`; chunk count/shape pinned; missing model dir raises with the fetch command; lazy import verified.
- [ ] **Step 2:** Run to verify they fail.
- [ ] **Step 3:** Implement (lazy `sherpa_onnx.OfflineTts`, sentence → samples → f32 24 kHz chunks, sample rate conversion if native is 44.1 kHz).
- [ ] **Step 4:** Run tests; `ruff check` clean.
- [ ] **Step 5:** Commit.

### Task 4: ASR client (Canary via sherpa-onnx)

**Files:**
- Create: `src/verbix/voice/sherpa_stt.py`
- Test: `tests/test_sherpa_stt.py`

**Interfaces:**
- Consumes: `voice/base.py` `SpeechToText` protocol; `fetch()` paths from Task 2.
- Produces: `SherpaSTT(model_dir)` with `async transcribe(audio, sample_rate=16000) -> str`; keeps `open_session` if the live session in `voice/stt.py` is to be preserved (see step 1 decision).

- [ ] **Step 1:** Decide from the Task 3/4 evidence: keep the existing live `open_session` (endpointing stays, M2 stays cheap) or drop it. Record the decision in the commit message.
- [ ] **Step 2:** Write failing tests — fake recognizer; text returned; empty audio → `""`; noise input behavior measured (Task 1 data) and pinned; missing model dir raises with the fetch command.
- [ ] **Step 3:** Run to verify they fail.
- [ ] **Step 4:** Implement; if keeping the session, port `STTSession`'s endpointing onto the new decode function unchanged.
- [ ] **Step 5:** Run tests; `ruff check` clean.
- [ ] **Step 6:** Commit.

### Task 5: Wire + delete (cutover)

**Files:**
- Modify: `src/verbix/cli.py` (voice block), `src/verbix/voice/__init__.py` (re-exports), `pyproject.toml` (drop `faster-whisper`, add `sherpa-onnx`), `README.md` (module map), `THIRD-PARTY-NOTICES.md` (Supertonic entry)
- Delete: `src/verbix/voice/pocket_helper.py`, `src/verbix/voice/tts_assets.py`, `scripts/setup_tts_runtime.sh`, `scripts/build_tts_helper.sh`, `scripts/fetch_tts_assets.py`
- Delete/rewrite: `tests/test_pocket_helper.py`, `tests/test_tts_assets.py` (their replacements are Tasks 2–4)

- [ ] **Step 1:** Verify green baseline: `uv run pytest -q` + `uv run ruff check src tests`.
- [ ] **Step 2:** Execute wiring + deletions + dep swap (regenerate lockfile).
- [ ] **Step 3:** Verify: full suite green, `ruff` no new errors, grep confirms zero live references to `pocket_helper|tts_assets|faster_whisper|PocketHelperEngine|setup_tts_runtime` in src/tests/scripts.
- [ ] **Step 4:** Run the real thing: `uv run scripts/fetch_voice_models.py`, then a live PTT voice turn, then a barge-in. Record both outcomes in the commit message.
- [ ] **Step 5:** Commit.

### Task 6: License gate (blocks release, not merge)

**Files:** `LICENSES/`, `DISTRIBUTION-TERMS.md` (drafts exist on the `m1-audiocpp-swap` branch — port or rewrite).

- [ ] **Step 1:** Port the license set + terms from the branch, updated for sherpa-onnx as the loader.
- [ ] **Step 2:** Engineer sign-off on the OpenRAIL-M terms + machine-speech disclosure wording. Until then the Supertonic TTS stays `UNRESOLVED` in the notices file.

---

## Not in scope (recorded for later)

- DC-bias conditioning (`capture.py`): unproven on main, cheap hygiene — separate 15-minute slice if wanted.
- M2 hands-free auto-submit (now cheaper: `STTSession` endpointing + VAD).
- M3 talk-over interruption (needs AEC regardless of backend).
- Windows/macOS hardware validation.
