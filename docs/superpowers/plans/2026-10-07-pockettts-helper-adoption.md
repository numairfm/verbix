# PocketTTS Helper Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run TTS through a resident PocketTTS.cpp helper process (proven 370 MB peak, trusted Alba voice) driven from Python, replacing the torch backend entirely.

**Architecture:** New `voice/pocket_helper.py` owns a resident helper: spawn on first `load()`, health-poll until ready, synthesize per sentence via `POST /tts` on localhost with a bearer token, supervise with one restart on transient failure, terminate on `aclose()`. Same `TextToSpeech` surface as today's engine (lazy `load()`, `synthesize()` yielding 24 kHz f32 ~80 ms chunks, `aclose()`), so the pipeline, player, and barge-in paths are untouched. Helper binary is built from source per platform in CI; models, voice, and licenses download once into a cache dir with pinned byte-size checks. Designed fresh for this repo — no code ported from the old project, though its validated settings are reused where noted.

**Tech Stack:** PocketTTS.cpp (pinned commit — implementer records it), ONNX Runtime (fetched by the helper's own CMake), stdlib only on the Python side (`subprocess`, `urllib`, `http` — numpy already a main dep for chunk math).

**Spec:** This plan IS the spec — decided in audit discussion 2026-10-07. Evidence: helper measured 370 MB peak / RTF 0.40x on this machine with the trusted voice; torch measured 1067 MB peak; sherpa configurations disqualified on voice. STT stays faster-whisper (separate migration). Licensing follow-up (attribution, voice-likeness clearance) tracked separately, not in this plan.

## Global Constraints

- Python >= 3.13; chunk format stays 24 kHz f32 mono (`voice/player.py`, `voice/output.py` unchanged).
- `TextToSpeech.synthesize(text: str) -> AsyncIterator[bytes]` (`voice/base.py:21`) is the only consumer contract.
- No network, binary, or weights in tests — ever. Engine tests run against an in-thread fake HTTP server speaking the helper's `POST /tts` protocol.
- `ruff check` clean on touched files; full suite green at the end of Task 3.
- Nothing copied from the old project: fresh module, fresh tests, fresh build script. Validated values (flags, sizes) may be reused; code may not.

## Review Focus

- Empty/whitespace text yields zero chunks and never touches the helper — test pinned in Task 1.
- Helper dies mid-turn: one supervised restart, then a loud error naming the chunk — test pinned in Task 1.
- First run with no network fails with a clear "download failed, retry with network" error, not a traceback — test pinned in Task 2.
- Partial/corrupt download is detected by byte-size check and re-fetched, never loaded — test pinned in Task 2.
- Two `load()` calls start exactly one helper (concurrent-safe); `aclose()` always terminates it, no orphans — test pinned in Task 1.

---

### Task 1: Helper engine module with fake-server tests

**Files:**
- Create: `src/verbix/voice/pocket_helper.py`
- Test: `tests/test_pocket_helper.py`

**Interfaces:**
- Consumes: a helper speaking `POST /tts` with JSON `{"text": str, "voice": str}` returning raw f32le PCM bytes (200) or JSON error (non-200); `GET /health` returning 200 when warm. (Verified against the helper's `--help` and observed behavior 2026-10-07.)
- Produces: `PocketHelperEngine(helper_bin: Path | None = None, models_dir: Path | None = None, voices_dir: Path | None = None, voice: str = "alba.wav", threads: int | None = None)` with `.is_loaded: bool`, `async load() -> None` (spawns `helper --server --port <ephemeral> --precision int8 --mimi-precision fp32 --temperature 0.3 --lsd-steps 1 --first-chunk 1 --max-chunk 1 --threads <min(4,cpu)> --models-dir … --voices-dir … --tokenizer …/tokenizer.model`, polls `/health` ≤10 s; idempotent under a lock), `async synthesize(text: str) -> AsyncIterator[bytes]` (empty text yields nothing; one POST per call; non-200 raises `RuntimeError` naming chunk + status; bytes yielded verbatim), `async aclose() -> None` (terminates child, clears state). `VERBIX_TTS_DIR` env overrides the binary/models/voices root for dev.

- [ ] **Step 1: Write the failing tests** in `tests/test_pocket_helper.py` (in-thread `http.server` fake: `/health` → 200, `/tts` → canned f32 bytes or configured status; spawn is injected as a stub so no real process starts):

```python
def test_synthesize_posts_text_and_yields_pcm_verbatim(): ...
    # fake records JSON body; engine yields its exact bytes
def test_empty_text_yields_nothing_and_never_spawns(): ...
def test_non200_raises_naming_chunk_and_status(): ...
def test_dead_helper_restarts_once_then_errors(): ...
    # first POST hangs/kills connection, second succeeds after restart stub
def test_double_load_starts_one_helper_and_aclose_kills_it(): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_pocket_helper.py -q`
Expected: FAIL with import error on the missing module.

- [ ] **Step 3: Implement `PocketHelperEngine` in `src/verbix/voice/pocket_helper.py`**

Stdlib only (`subprocess`, `urllib.request`, `asyncio.to_thread` for blocking IO). Ephemeral port via a bound-then-closed socket. Bearer token generated per spawn, sent as header. No retries except the single supervised restart on transport-level failure.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_pocket_helper.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/pocket_helper.py tests/test_pocket_helper.py
git commit -m "feat: add PocketTTS helper engine with supervised HTTP backend"
```

### Task 2: Assets, build script, CI assembly

**Files:**
- Create: `src/verbix/voice/tts_assets.py`
- Create: `scripts/build_tts_helper.sh`
- Create: `.github/workflows/tts-helper.yml` (or extend existing CI — implementer checks first)
- Test: `tests/test_tts_assets.py`

**Interfaces:**
- Consumes: nothing from Task 1 (pure paths + builds).
- Produces: `ensure_tts_runtime(root: Path | None = None) -> dict` returning `{"bin": Path, "models": Path, "voices": Path}` with everything verified present. Resolution order (fast dev loop first): usable binary already cached → download prebuilt bundle for this platform → compile pinned source locally via `scripts/build_tts_helper.sh` → `RuntimeError` naming what failed. Customers only ever hit the first two; local compile exists so development and unsupported platforms never wait on CI. Pinned byte sizes (re-verify at implementation; recorded 2026-10-07): `flow_lm_flow_int8.onnx` 9962530, `flow_lm_main_int8.onnx` 76341079, `mimi_decoder.onnx` 41471926, `mimi_encoder.onnx` 39768446, `text_conditioner.onnx` 16388344, `tokenizer.model` 59339, `bos_before_voice.npy` 4224, `alba.wav` 958542 (canonical Alba reference `alba-mackenna/casual.wav` from `kyutai/tts-voices`, not a custom variant). Model/voice sources: canonical Kyutai + exporter releases (exact URLs recorded by implementer in the manifest; no third-party mirrors as primary). Default root `$VERBIX_TTS_DIR` or `~/.cache/verbix/tts`.

- [ ] **Step 1: Write the failing tests** in `tests/test_tts_assets.py` (all fixtures local; downloads faked via `file://` URLs):

```python
def test_missing_or_wrong_size_file_refetches(tmp_path): ...
def test_no_network_fails_clearly(tmp_path, monkeypatch): ...
    # URLError -> RuntimeError mentioning "download failed" + file name
def test_layout_contains_bin_models_voices(tmp_path): ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tts_assets.py -q`
Expected: FAIL with import error on the missing module.

- [ ] **Step 3: Implement `ensure_tts_runtime` + build script + CI**

`build_tts_helper.sh <source-commit> <out-dir>`: clone pinned commit, cmake release build, install the binary as `bin/verbix-tts` (+ `lib/`, fetched models/voices/licenses) into the layout. The engine resolves `bin/verbix-tts` first with a `bin/pocket-tts` fallback, so Task Manager shows `verbix-tts`. CI matrix (linux, macos, windows) runs it and uploads the layout as the per-platform TTS bundle. Record source commit, model URLs, byte sizes, and license references in `models/MANIFEST.json` written into the layout.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tts_assets.py tests/test_pocket_helper.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/voice/tts_assets.py tests/test_tts_assets.py scripts/build_tts_helper.sh .github/workflows/tts-helper.yml
git commit -m "feat: TTS asset fetch, helper build script, per-platform CI"
```

### Task 3: Wire cli, drop torch, suite green

**Files:**
- Modify: `src/verbix/cli.py` (`_build_voice_pipeline`)
- Modify: `pyproject.toml` (`[project.optional-dependencies] tts`)
- Delete: `src/verbix/voice/pocket_tts.py`
- Test: full suite (existing voice tests use mocks and must stay green)

**Interfaces:**
- Consumes: `PocketHelperEngine` from Task 1 (constructor: `PocketHelperEngine()` — everything resolves at `load()`).
- Produces: identical voice-mode behavior with torch uninstallable.

- [ ] **Step 1: Swap the builder** — in `_build_voice_pipeline`, replace the `PocketTTSEngine` import/construction with `PocketHelperEngine` (same lazy-import-inside-function pattern). `AudioPlayer(tts=tts, sink=sink)` untouched.

- [ ] **Step 2: Drop the torch dep and backend** — remove `pocket-tts` and `torch` (and the pytorch-cpu index if then unused — implementer checks `uv.lock` for other consumers first) from the `tts` extra; if nothing remains, delete the extra entirely. Delete `src/verbix/voice/pocket_tts.py`. Confirm: `grep -rn "PocketTTSEngine\|pocket_tts\|import torch" src/ tests/` returns nothing.

- [ ] **Step 3: Run the full suite plus lint**

Run: `.venv/bin/python -m pytest -q` (expected: all pass), `.venv/bin/ruff check src tests` (expected: no new errors vs. baseline).

- [ ] **Step 4: Commit**

```bash
git add src/verbix/cli.py pyproject.toml uv.lock src/verbix/voice/pocket_tts.py
git commit -m "feat: default TTS to resident helper, drop torch backend"
```

### Task 4: Bench vs. baselines, attribution, docs

**Files:**
- Modify: `README.md` (voice section: helper + runtime layout + cache locations)
- Create: `THIRD-PARTY-NOTICES.md` (exact filename at implementer discretion, recorded in commit)

**Interfaces:**
- Consumes: finished Tasks 1–3 with a CI-built (or locally built) helper + fetched assets.
- Produces: recorded numbers + attribution + green suite on the clean env.

- [ ] **Step 1: Fresh env + full suite** — `uv sync --extra tts` (or without the extra if deleted), then `.venv/bin/python -m pytest -q` (expected: all pass; torch must be absent: `.venv/bin/python -c "import torch"` fails with ImportError).

- [ ] **Step 2: Bench the migration** — fixed benchmark sentence 3x through the real engine, recording helper peak RSS (via `/usr/bin/time -v`), wall time, RTF. Gates (2026-10-07 baselines, same machine): helper peak <= 450 MB (was 370 with the reference setup; torch was 1067), RTF <= 0.5x (was 0.40). Voice parity: user listens against the reference recording before sign-off. On miss: stop, report numbers, do not tune blindly.

- [ ] **Step 3: Attribution** — `THIRD-PARTY-NOTICES.md` rows: PocketTTS.cpp (repo + pinned commit + its license), Kyutai PocketTTS weights (CC-BY-4.0, link, "int8/fp32 ONNX export" modification note), ONNX Runtime (MIT), Alba voice reference (source URL + likeness-clearance status from the commercial track). One row per heavy asset (add whisper tiny.en while there: Apache-2.0).

- [ ] **Step 4: Docs + commit** — README voice section names the helper, the runtime layout, and cache locations.

```bash
git add README.md THIRD-PARTY-NOTICES.md
git commit -m "docs: helper TTS bench, attribution, voice docs"
```

## Self-Review

1. **Spec coverage:** 370 MB target → Task 4 gates; trusted voice → same bundle + voice file + flags in Tasks 1–2; protocol unchanged → Task 1 surface mirrors today's engine; no torch → Task 3 grep + ImportError check; attribution → Task 4. STT untouched. Covered.
2. **Step scan:** test steps assert exact values; code steps name exact signatures/files/flags; verifications name commands + expected output. Fresh-build requirement explicit (no ports).
3. **Type consistency:** `PocketHelperEngine(...)` shape identical Tasks 1–3; `ensure_tts_runtime` consumed by `load()` defaults; `synthesize` yields `bytes` everywhere.
4. **Review Focus:** all five lines have owning tests (Tasks 1–2).
5. **Proportion:** decisions and pins stated; bodies in implementation. No transcript code beyond test sketches.
