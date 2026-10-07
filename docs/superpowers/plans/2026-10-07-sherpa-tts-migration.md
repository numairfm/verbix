# Sherpa TTS Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the torch-based PocketTTS backend with sherpa-onnx PocketTTS int8 behind the existing `TextToSpeech` protocol, cutting TTS memory ~3x with no UX change.

**Architecture:** New `SherpaTTSEngine` in `voice/sherpa_tts.py` mirrors `PocketTTSEngine`'s surface (lazy `load()`, `synthesize()` yielding 24 kHz f32 ~80 ms chunks, `aclose()`). Model weights (~200 MB) and the Alba reference clip download once into a cache dir on first `load()` with pinned byte-size checks. `cli.py` constructs it where it constructs `PocketTTSEngine` today; the torch path and its deps are deleted.

**Tech Stack:** sherpa-onnx==1.13.8 (pinned), numpy (already a main dep), stdlib urllib (no new fetch deps).

**Spec:** This plan IS the spec — decided in the audit discussion 2026-10-07. Measured baselines (same machine, same sentence, Alba voice): torch load 1054 MB / RTF 0.43x / import 406 MB; sherpa load 352 MB / RTF 0.37x / import 36 MB. STT stays faster-whisper (separate migration). Stock sherpa wheel (espeak statically inside) per user's decision; licensing follow-up tracked separately, not in this plan.

## Global Constraints

- Python >= 3.13; voice pipeline speaks 24 kHz f32 mono (`voice/player.py`, `voice/output.py` assume this — do not change chunk format).
- `TextToSpeech.synthesize(text: str) -> AsyncIterator[bytes]` (`voice/base.py:21`) is the only contract consumers use.
- No network or model weights in tests — ever. Fake the `sherpa_onnx` module via monkeypatched `sys.modules`; fetch tests use `file://` fixture URLs.
- `ruff check` clean on touched files; full suite green at the end of Task 3.

## Review Focus

- Empty/whitespace text yields zero chunks and never touches the model — test pinned in Task 1.
- Unknown voice name fails fast listing supported names (today exactly one: `alba`) — test pinned in Task 1.
- First run with no network fails with a clear "download failed, retry with network" error, not a traceback — test pinned in Task 2.
- Partial/corrupt download is detected by byte-size check and re-fetched, never loaded — test pinned in Task 2.
- Non-24 kHz sherpa output refuses loudly rather than playing chipmunk audio — test pinned in Task 1.

---

### Task 1: SherpaTTSEngine behind the TextToSpeech protocol

**Files:**
- Create: `src/verbix/voice/sherpa_tts.py`
- Test: `tests/test_sherpa_tts.py`

**Interfaces:**
- Consumes: `sherpa_onnx.OfflineTts`, `sherpa_onnx.OfflineTtsConfig`, `sherpa_onnx.OfflineTtsModelConfig`, `sherpa_onnx.OfflineTtsPocketModelConfig`, `sherpa_onnx.GenerationConfig` (attribute-style: construct empty, then set `reference_audio`, `reference_sample_rate`, `num_steps`).
- Produces: `SherpaTTSEngine(voice: str = "alba", num_steps: int = 2, num_threads: int = 2, sample_rate: int = 24000)` with `.channels == 1`, `.audio_format == "f32"`, `.sample_rate == 24000`, `.is_loaded: bool`, `async load() -> None` (idempotent), `async synthesize(text: str) -> AsyncIterator[bytes]` (24 kHz f32 mono, ~80 ms / 7680-byte frames; empty text yields nothing), `async aclose() -> None` (resets to unloaded). Model file paths and reference PCM bytes arrive via constructor-or-setter from Task 2 — Task 1 takes them as plain constructor args `model_dir: Path | None = None, reference_wav: Path | None = None` and raises `RuntimeError` naming the missing piece when unset.

- [ ] **Step 1: Write the failing tests** in `tests/test_sherpa_tts.py` (monkeypatch a fake `sherpa_onnx` module into `sys.modules` whose `OfflineTts.generate` returns canned 24 kHz float samples; never import the real package):

```python
def test_synthesize_yields_80ms_f32_frames(): ...
    # engine with fake backend emitting 24000 samples; expect 12 chunks of 7680 bytes, all float32-convertible
def test_empty_text_yields_nothing_and_never_loads(): ...
def test_unknown_voice_lists_supported_names(): ...
    # SherpaTTSEngine(voice="nope") + load() -> RuntimeError mentioning "alba"
def test_non_24k_output_refused(): ...
    # fake backend with sample_rate=16000 -> RuntimeError on synthesize
def test_aclose_resets_to_unloaded(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sherpa_tts.py -q`
Expected: FAIL with "sherpa_tts not defined" (import error on the missing module).

- [ ] **Step 3: Implement `SherpaTTSEngine` in `src/verbix/voice/sherpa_tts.py`**

`import sherpa_onnx` lazily inside `load()` (so text modes never pay the import) with a clear `ImportError` message (`pip install verbix[tts]`). `synthesize` runs blocking `tts.generate` in `asyncio.to_thread`, slices float32 samples into 1920-sample frames, yields `.tobytes()`. Reference PCM loaded once per `load()` from the `reference_wav` path (int16 mono wav → float32 / 32768).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sherpa_tts.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/sherpa_tts.py tests/test_sherpa_tts.py
git commit -m "feat: add SherpaTTSEngine behind TextToSpeech protocol"
```

### Task 2: Voice + model fetch with cache and size checks

**Files:**
- Create: `src/verbix/voice/sherpa_assets.py`
- Test: `tests/test_sherpa_assets.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure paths + urllib).
- Produces: `ensure_sherpa_pocket(model_dir: Path | None = None) -> Path` (downloads the 7 bundle files when missing, verifies pinned byte sizes, returns dir), `ensure_reference_voice(name: str, voices_dir: Path | None = None) -> Path` (downloads `alba` clip once, returns wav path; unknown name raises `RuntimeError` listing supported names). Defaults root at `$VERBIX_CACHE_DIR` or `~/.cache/verbix`. Pinned file sizes (bytes): `lm_main.int8.onnx` 76341079, `encoder.onnx` 72713165, `decoder.int8.onnx` 22693618, `text_conditioner.onnx` 16388343, `lm_flow.int8.onnx` 9962530, `vocab.json` 69478, `token_scores.json` 123616. Bundle base URL: `https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26` — NO, per-file mirror (measured 10x faster here): `https://huggingface.co/csukuangfj2/sherpa-onnx-pocket-tts-int8-2026-01-26/resolve/main/<file>` with the GitHub tarball as documented fallback in a code comment (provenance note). Alba clip candidates, verified in order by the implementer with `curl -fI` (first reachable wins, recorded in commit message): `https://huggingface.co/datasets/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav`, then `https://huggingface.co/kyutai/tts-voices/resolve/main/alba-mackenna/casual.wav`.

- [ ] **Step 1: Write the failing tests** in `tests/test_sherpa_assets.py` (serve fixtures over `file://` URLs only):

```python
def test_size_mismatch_refetches(tmp_path): ...
    # corrupt one cached file (wrong size) -> ensure_* redownloads from file:// fixture, sizes all pass
def test_no_network_fails_clearly(tmp_path, monkeypatch): ...
    # urlretrieve raising URLError -> RuntimeError mentioning "download failed" and the file name
def test_unknown_voice_name(tmp_path): ...
    # ensure_reference_voice("nope") -> RuntimeError mentioning "alba"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sherpa_assets.py -q`
Expected: FAIL with import error on the missing module.

- [ ] **Step 3: Implement `ensure_sherpa_pocket` + `ensure_reference_voice` in `src/verbix/voice/sherpa_assets.py`**

Stdlib `urllib.request.urlretrieve` only. Skip files already present with correct size. Wire Task 1's engine to these: `SherpaTTSEngine.load()` calls both helpers when `model_dir`/`reference_wav` are None (keep the explicit-path overrides for tests).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sherpa_assets.py tests/test_sherpa_tts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/sherpa_assets.py tests/test_sherpa_assets.py src/verbix/voice/sherpa_tts.py
git commit -m "feat: cache-checked sherpa model and voice fetch"
```

### Task 3: Wire cli, swap deps, delete torch path

**Files:**
- Modify: `src/verbix/cli.py` (`_build_voice_pipeline`, ~lines 280-309)
- Modify: `pyproject.toml` (`[project.optional-dependencies] tts`)
- Delete: `src/verbix/voice/pocket_tts.py`
- Test: full suite (no new test file; existing voice tests use mocks and must stay green)

**Interfaces:**
- Consumes: `SherpaTTSEngine` from Task 1 (constructor: `SherpaTTSEngine()` — defaults resolve everything at `load()`).
- Produces: identical voice-mode behavior with the torch package uninstallable.

- [ ] **Step 1: Swap the builder** — in `_build_voice_pipeline`, replace the `PocketTTSEngine` import/construction with `SherpaTTSEngine` (same lazy-import-inside-function pattern). No signature changes elsewhere: `AudioPlayer(tts=tts, sink=sink)` is untouched.

- [ ] **Step 2: Swap the dep and delete the old backend** — `pyproject.toml` `tts` extra becomes exactly `["sherpa-onnx>=1.13.8"]` (numpy already a main dep; pocket-tts and torch lines removed). Delete `src/verbix/voice/pocket_tts.py`. Confirm no remaining references: `grep -rn "PocketTTSEngine\|pocket_tts\|import torch" src/ tests/` must return nothing.

- [ ] **Step 3: Run the full suite plus lint**

Run: `.venv/bin/python -m pytest -q` (expected: all pass), `.venv/bin/ruff check src tests` (expected: no new errors vs. baseline).
Note: the developer venv here still has torch installed until the next `uv sync`; suite passing pre-sync is fine, but re-run pytest once after `uv sync --extra tts` in Task 4.

- [ ] **Step 4: Commit**

```bash
git add src/verbix/cli.py pyproject.toml uv.lock src/verbix/voice/pocket_tts.py
git commit -m "feat: default TTS to sherpa-onnx, drop torch backend"
```

### Task 4: Bench vs. baselines, attribution, docs

**Files:**
- Modify: `README.md` (voice section: engine + model + cache locations)
- Create: `THIRD-PARTY-NOTICES.md` (or append; exact filename at implementer discretion, recorded in commit)

**Interfaces:**
- Consumes: finished Tasks 1–3 on a fresh `uv sync --extra tts` env.
- Produces: recorded numbers + attribution + green suite on the clean env.

- [ ] **Step 1: Fresh env + full suite** — `uv sync --extra tts`, then `.venv/bin/python -m pytest -q` (expected: all pass; torch must be absent: `.venv/bin/python -c "import torch"` fails with ImportError).

- [ ] **Step 2: Bench the migration** — synthesize the зафиксированное sentence "Hello! This is a benchmark of the text to speech engine. How do I sound?" 3x through the real `SherpaTTSEngine`, recording import RSS, loaded RSS, wall time, RTF. Gates (from 2026-10-07 baselines, same machine): loaded RSS <= 400 MB (was 1054), RTF <= 0.5x (was 0.43), import delta <= 60 MB (was ~390). On miss: stop, report numbers, do not tune blindly.

- [ ] **Step 3: Attribution** — copy the bundle's `LICENSE` (CC-BY-4.0) reference into `THIRD-PARTY-NOTICES.md`: Kyutai PocketTTS weights via sherpa-onnx int8 export, links to both repos, "modified: int8 ONNX export", plus the pinned bundle release URL. One row per heavy model asset (add whisper tiny.en while there: Systran faster-whisper-tiny.en, Apache-2.0).

- [ ] **Step 4: Docs + commit** — README voice section names sherpa-onnx 1.13.8, the ~200 MB first-run download, and cache locations.

```bash
git add README.md THIRD-PARTY-NOTICES.md
git commit -m "docs: sherpa TTS bench, attribution, voice docs"
```

## Self-Review

1. **Spec coverage:** measured targets (RSS/RTF/import) → Task 4 gates; same voice → Task 2 clip + cache; same protocol/chunks → Task 1 frame math + 24 kHz guard; no torch → Task 3 grep + ImportError check; attribution → Task 4. STT untouched anywhere. Covered.
2. **Step scan:** each test step asserts exact values; each code step names exact signatures/files; verification steps name commands + expected output. The Alba URL has two ordered candidates with a decide-rule (first reachable via `curl -fI`) instead of an unverified pin — deliberate, flagged here.
3. **Type consistency:** `SherpaTTSEngine(...)` constructor shape identical in Tasks 1–3; `ensure_*` signatures identical in Tasks 2–3 (`load()` calls them with no args); `synthesize` yields `bytes` everywhere.
4. **Review Focus:** all five lines have owning tests (Tasks 1–2).
5. **Proportion:** plan states decisions and pins values; bodies live in implementation. No transcript code beyond test sketches.
