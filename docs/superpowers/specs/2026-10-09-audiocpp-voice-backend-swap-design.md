# M1 Spec — audio.cpp voice backend swap (VAD + STT + TTS, PTT unchanged)

Milestone 1 of 3. M2 adds the VAD-gated hands-free loop. M3 adds voice
interruption with external AEC. Each milestone ships on its own.

Revision 5: addresses critic rounds 1–4. Open question count is zero.

## 1. Intent

Replace Verbix's scattered Python voice backends (faster-whisper STT,
PocketTTS.cpp helper, Silero-through-faster-whisper) with one native
runtime, audio.cpp, covering VAD + ASR + TTS through a resident local
server. What the user operates is interaction-identical to today —
same toggles, same floor states, same barge-in — except voice quality
and latency characteristics (bounded in §5). This also removes
faster-whisper and its model-runtime chain from the install.

Locked decisions: half-duplex interaction (M2), server subprocess
integration, full swap including Supertonic 3, three milestones, no
heuristic echo hacks, cross-platform in M1 (Linux/Windows/macOS).

Success: `uv run verbix --voice` works on all three OSes with no Python
audio ML in-process, ASR at parity with `tiny.en` (§5), Supertonic voice
accepted, suite + `ruff` green, license gates closed (§7).

## 2. Goals / non-goals

Goals: resident `audiocpp_server` supervised by Python; VAD/STT/TTS HTTP
clients behind the existing `voice/base.py` protocols; sounddevice mic
capture backend for Windows/macOS; Supertonic 3 as the TTS voice;
one-command provisioning per OS; closed license gates with
`THIRD-PARTY-NOTICES.md` updated; old backends deleted in the same
change.

Non-goals: VAD-gated auto-submit (M2), voice interruption / AEC (M3),
runtime/floor/announcer changes, ASR supremacy (parity is the bar).

## 3. Architecture

`voice/` keeps its one-concern-per-file layout:

- `voice/audiocpp_server.py` (new): owns the process only — spawn with
  the provisioned `server.json`, `GET /health` wait, crash respawn
  (respawn reuses `server.json`; a restarted process owns no prior
  sessions, so later requests open fresh ones — there is no
  re-registration step, the config is the registration), `aclose()`.
  Binds `127.0.0.1`. No auth: the server documents none.
- `voice/audiocpp_vad.py` (new): one-shot no-speech gate only —
  `POST /v1/tasks/run` with the configured Silero model entry id
  (`silero-vad`, family `silero_vad`, written by the provisioner) and
  `{"request": {"audio_base64": "<wav>", "options": {"threshold": 0.5}}}`
  (`threshold` nests under `options` — a top-level key is silently
  dropped by the server; remaining tunables stay at documented
  defaults), returning speech/no-speech from the result segments. No
  streaming session in M1; M2 adds it.
- `voice/audiocpp_stt.py` (new): one-shot `transcribe(buffered_pcm)`
  via `POST /v1/audio/transcriptions` (multipart upload), with
  `language` pinned to `"en"` (today's forced-`en` behavior preserved).
  No `open_session`: M1 drops live incremental STT (see §4).
- `voice/audiocpp_tts.py` (new): `synthesize(text)` as an async
  generator function, matching the convention `PocketHelperEngine`
  and the mocks already follow (call without await, iterate chunks).
  Voice control is Supertonic's documented request option
  `voice_id: "M1"` (same control as CLI `--voice-id`; the exact JSON
  field on `POST /v1/audio/speech` is confirmed live in task 1 —
  top-level `"voice"` is NOT used: it routes through
  voice-library/clone fallback machinery and silently depends on no
  `M1.wav` existing in `voice_dir`, so the provisioner both omits the
  field and asserts `voice_dir` holds no `M1.wav`). Voice `M1` is
  provisional pending the provenance gate (§7). Latency metric is
  request-to-complete-yield per sentence (one-shot offline synthesis).
  Output assumed 24 kHz f32 mono (task 1 confirms the native rate; on
  mismatch the client converts — numpy is already a dependency — or
  the sink rate is reconfigured, named fallback).
- `voice/wavutil.py` (new, one job): stdlib-`wave` PCM↔WAV packaging
  (16 kHz mono s16le capture bytes → WAV for `audio_base64` and
  multipart uploads; WAV strip for TTS responses). Both server audio
  inputs require WAV; capture returns raw s16le — this helper closes
  that gap in one place.
- `voice/capture.py` (edited): adds a sounddevice input backend.
  Backend order: `pw-record` → `arecord` → sounddevice (lazy import,
  same pattern as `callback_sink.py:55`). Required capture format is
  16 kHz mono s16le: the backend requests it from the device, and on
  mismatch resamples via numpy (existing pattern, dependency already
  present) — never silently accepts another rate. Hatch open logs one
  line (backend + rate + channels) as a §3 requirement, so the §6
  checklist's "backend named in log" is assertable. sounddevice is an
  OPTIONAL extra (`pyproject.toml:26-29`, not a base dep): the
  provisioner installs `verbix[audio]`, and voice startup raises with
  the provision command when no backend is available — the
  silent-`b""` path becomes unreachable by gate, not by luck.
  PortAudio rule, per OS: Linux needs the system package (provisioner
  checks import, hints `dnf` on Fedora / `apt` on Debian-Ubuntu on
  failure); Windows/macOS ride the wheel-bundled library (import test
  decides, same gate).
- Playback rule (no `sinks.py` edit): voice startup requires a working
  sink — callback sink via the installed extra, or a present
  subprocess backend — else it raises with guidance. This makes the
  no-backend discard path unreachable in voice mode instead of
  silently no-op'ing on Win/Mac.
- `voice/pipeline.py` (edited minimally): the `open_session` live
  branch (`pipeline.py:85-88`, finalize-preference `106-108`) is
  deleted; stop always does one-shot transcribe. Pre-submit failures
  (VAD/STT gate inside the toggle path) return
  `{"action": "error", "error": <message>}` from toggle (never raise —
  toggle's contract is dicts) with the floor reset to `IDLE`; the CLI
  voice loop checks `"error"` explicitly (prints the message,
  continues, spawns no turn — today an unknown action would fall
  through silently); the IPC toggle branch maps it to `status="error",
  error=<message>, data={"action": "error"}` with the request `id`
  echoed per the `IPCResponse` shape. Post-submit failures (TTS
  synthesis / playback inside the background turn, after toggle
  already returned `processing`) keep today's surfacing: the turn task
  fails, the existing done-callback logs, the floor returns to `IDLE`
  via the existing `finally`, and the CLI prints one failure line at
  that callback — no new IPC surface in M1 (daemon clients observe no
  completion, exactly as they do for failed turns today). Toggles,
  barge-in, floor transitions, and empty-transcript idle are otherwise
  unchanged.
- `voice/base.py` (edited): protocols untouched; mocks extended.
- `voice/__init__.py` (edited): the `__getattr__` lazy re-exports
  switch to the new clients; old names raise `AttributeError` as
  unknown names do today.
- `src/verbix/cli.py` (edited): the voice-mode wiring block
  constructs supervisor + new clients, adds the `"error"` branch to
  the voice loop, extends the turn-failure callback with the printed
  line; the `ensure_tts_runtime` call becomes a voice-runtime
  provisioning check with setup guidance.
- `pyproject.toml` + `uv.lock` (edited): faster-whisper removed from
  base dependencies (its absence breaks the old imports — intended,
  they are deleted in the same change); lockfile regenerated. No new
  dependencies: httpx2 (HTTP client), numpy (conversion/resample),
  wave (stdlib), sounddevice (pre-existing optional extra).
- `LICENSES/` (new, committed dir): license copies (§7).
  `DISTRIBUTION-TERMS.md` (new, repo root): user-facing terms (§7).
  `THIRD-PARTY-NOTICES.md` (edited): entries + retained notices (§7).
- Test dispositions (suite must stay green): `test_pocket_helper.py`
  → rewritten as the new-clients suite (its fake-HTTP-server pattern
  is reused); `test_tts_assets.py` → rewritten against
  `provision_voice.py`; `test_stt_nospeech.py` → rewritten against the
  new STT; `test_incremental_stt.py` → DELETED (tests the dropped
  session machinery; M2 reintroduces session tests);
  `test_voice_loop_barge.py` → re-imported to the new modules,
  behavior assertions unchanged; `test_capture_warmup.py` →
  rewritten (drops the `_live_stt` fixture, keeps on-chunk-clearing,
  frame-bytes, and warm-rate-limit assertions).
- Deleted in the same change: `voice/stt.py`, `voice/pocket_helper.py`,
  `voice/tts_assets.py`, `scripts/setup_tts_runtime.sh`,
  `scripts/build_tts_helper.sh`, `scripts/fetch_tts_assets.py` (all
  PocketTTS-specific). `scripts/benchmark_toggle_voice.py` is already
  stale, is not collected by the suite, and is explicitly excluded
  (not blessed, not fixed here).
- Untouched: `voice/output.py`, `voice/player.py`, `voice/sinks.py`,
  `voice/callback_sink.py`, `voice/segments.py` (imported by
  `agent/agents/default.py:44`, stays), `runtime/`, `core/`, `agent/`.

audio.cpp surface relied on (verified in-repo): warm model+session
reuse per id; `GET /health`, `GET /v1/models`, `POST /v1/audio/speech`
(wav default), `POST /v1/audio/transcriptions` (multipart), generic
`POST /v1/tasks/run` (`model` + `request` with `audio`/`audio_base64`
WAV, options nested under `options`); config-driven `server.json` with
cpu/cuda/vulkan/metal; Silero `vad` offline+streaming with
`--segments-out` contract and documented request tunables; Supertonic
31 languages, offline+streaming GGUF, 10 preset voices controlled by
`voice_id`. Implementation task 1 confirms four value-level details
against a live local server and records them here: VAD segment result
shape, Supertonic native rate/channels, WAV header size, and the exact
JSON field carrying `voice_id` on `POST /v1/audio/speech`. Routes are
pinned from docs; only values are confirmed live.

## 4. Data flow (M1)

Toggle → capture records → stop → raw bytes → `wavutil` package → VAD
gate (one HTTP call) → silence: `IDLE`, no model call → speech:
one-shot transcribe → `runtime.submit` → agent turn → TTS per-sentence
synthesize → playback drains → `IDLE`. Deliberate change vs today:
finalize pays full-buffer ASR instead of tail-only, because the live
session is gone; §5 budgets it.

## 5. Model selection and validation (concrete)

- VAD: Silero, bundled (≈1.2 MB on disk, MIT), entry id `silero-vad`,
  threshold 0.5, tunables default. Fixed.
- TTS: Supertonic 3, package `supertonic_3_orig` (default); fallback
  package `supertonic_3_f16` only if CPU synthesis latency misses the
  bar below. `supertonic_3_q8_0` is explicitly excluded (upstream flags
  Q8 blockers unresolved for this family). Voice `M1` provisional.
- ASR shortlist with pre-cleared licenses: `fun_asr_nano`
  (Apache-2.0), `qwen3_asr` (Apache-2.0), `nemotron_asr` (OpenMDW-1.1:
  permission granted, retain agreement copy + origin notices on
  distribution, patent-retaliation termination, no output restrictions,
  clear third-party rights yourself — obligations recorded in
  `THIRD-PARTY-NOTICES.md`), `parakeet_tdt` nvidia official release
  (CC-BY-4.0) ONLY — the `oruk/orukeet` variant is a community port
  under CC-BY-SA-4.0 share-alike and is explicitly excluded (copyleft
  trap for closed-source). ASR stays `UNRESOLVED` until a package id
  is pinned.
- Validation corpus (fixed in this spec, not deferred): these 10
  English sentences, recorded once on the dev machine and committed
  under `tests/fixtures/voice/` as 16 kHz mono WAV —
  1. "The quick brown fox jumps over the lazy dog."
  2. "Pack my box with five dozen liquor jugs."
  3. "Schedule the deployment for Thursday morning."
  4. "The microphone level seems a little low today."
  5. "Cancel the background task and summarize the results."
  6. "What time is it in Tokyo right now?"
  7. "Sphinx of black quartz, judge my vow."
  8. "Push the hotfix after the tests turn green."
  9. "How does the voice pipeline handle interruptions?"
  10. "The five boxing wizards jump quickly."
  (No committed corpus exists under `tests/fixtures/voice/` today;
  `scratch/test_tts.wav` is an uncommitted manual STT fixture and
  `tts-runtime/voices/alba.wav` is a voice-clone clip — neither is a
  transcription corpus.) WER harness is a stdlib-`difflib` word-split
  script under `scripts/` (validation-only, not product): word-level
  tokenization, median WER and max stop-to-transcript over K=3 repeats.
  Machine string + server version + backend recorded with every
  result.
- Order matters: baselines (PocketTTS latency, `tiny.en` WER/latency)
  are measured BEFORE the §3 deletions, on the same corpus and
  machine. Bars: ASR WER no worse than `tiny.en` on identical audio;
  transcription real-time factor ≤ 0.5 on CPU; stop-to-transcript max
  ≤ 3 s for ≤ 15 s utterances over the repeats; TTS request-to-complete
  no worse than 1.25× today's PocketTTS on the same sentences.
  Silence/noise must produce empty output (the reason today's gate
  exists).

## 6. Cross-platform shipping

Prebuilt matrix (verified): Windows x64 CPU/Vulkan/CUDA (CUDA runtime
in a separate archive beside the binaries), Ubuntu x64 CPU/Vulkan,
macOS arm64/x64 Metal (documented Metal script build). Version pinned:
release TAG `v0.9.1`; per-OS artifact SHA256 pinned statically in the
provisioner manifest (updated by the engineer on version bump — no
runtime-written lockfile, no dirty tree) and verified on every
download. Provisioner is `scripts/provision_voice.py` — Python stdlib
only, flags `--backend cpu|cuda|vulkan|metal` (default auto) and
`--force-rebuild` — installing `verbix[audio]`, fetching binaries +
pinned GGUFs via the Python v2 model manager, writing `server.json`,
then gating exit 0 on health + VAD/ASR/TTS round-trips. Per-OS rules:
Windows — CPU flavor default, CUDA flavor only with the cudart archive
present; macOS — Metal, backend `metal`; Linux — `cuda` only with a
local toolkit, else `cpu`, Vulkan via the flag. Fedora rule: try the
Ubuntu binary, smoke check (`--list-devices`, health, round-trips); on
failure, CPU source build of target `audiocpp_server` via the Linux
build script, log kept, failure raising loudly with the log tail and
the supported-distro list. Layout keeps today's root convention under
`VERBIX_VOICE_DIR` (new; legacy `VERBIX_TTS_DIR`/PocketTTS contents are
incompatible and raise with the re-provision command — fail loudly).
Windows/macOS capture + playback are validated by the engineer on
hardware with this checklist: provision exit 0, record 5 s (backend
named in log), hear the TTS sentence, toggle barge-in once. No OS-only
behavior without a same-behavior fallback.

## 7. License gates (2 blockers named; sign-off blocks release)

Framework audio.cpp is Apache-2.0 (ShugoAI LLC) — clear. Silero VAD is
MIT — clear. Supertonic 3 is BigScience OpenRAIL-M (Supertone/
supertonic-3 `LICENSE`, `main` as read 2026-10-09; audio.cpp
`docs/model_licenses.md` @ `bd88e6ea` 2026-10-08 marks commercial use
allowed with restrictions passed on; OpenMDW-1.1 text read at
openmdw.ai/license/1-1/ 2026-10-09). The FULL restriction set binds
Verbix: (a) obey all applicable law; (b) no harming/exploiting minors;
(c) no verifiably false information to harm others; (d) no harmful
PII; (e) machine-generated content must be expressly disclaimed; (f)
no defamation/disparagement/harassment; (g) no impersonation
(deepfakes) without consent — voice cloning implicates this; (h) no
fully automated decisions adverse to legal rights or binding
obligations; (i) no discrimination on behavior/predicted traits; (j)
no exploiting group vulnerabilities causing harm; (k) no
discrimination on protected characteristics; (l) no medical advice or
results interpretation; (m) no justice/law-enforcement/immigration
prediction uses. (Lettered items summarize license Attachment A; the
license text governs.) Deliverables in this change: license copy
shipped under `LICENSES/` (committed dir; provisioner asserts
presence; it ships with the app tree — bundle method named in the
plan); `THIRD-PARTY-NOTICES.md` entry retaining all notices; a
user-facing `DISTRIBUTION-TERMS.md` (repo root) carrying the license
§4a restrictions as enforceable terms with the license §4b license
pointer, the license §6 output-accountability note, and the license
§7 latest-version duty (server version pinned per spec §6,
re-validated on bump). Legal sign-off blocks release; anything unclear
stays `UNRESOLVED` and blocks cutover. Blocker 1: ASR package pin
(§5). Blocker 2: preset voice `M1` provisional pending a
provenance/consent gate, same duty as the Alba voice-likeness
clearance in `THIRD-PARTY-NOTICES.md:19-22` (model licenses do not
cover personality rights).

## 8. Error handling

Loud failures only: missing binary/weights/unhealthy server raise with
the exact provision command — never silent fallback. Server crash
mid-turn: supervisor respawns from `server.json`, in-flight turn
errors via the §3 split contract. HTTP via httpx2 (already a base
dep) with per-request timeouts; timeouts enter the transport bucket.
Bounds: health wait 10 s total; VAD gate 15 s; STT `max(30 s, 4× audio
duration)`; TTS 60 s per sentence. Mapping, three buckets: transport
errors and timeouts (no HTTP status) → setup guidance; 4xx →
model-missing 404 → setup guidance, all other 4xx (validation, busy)
→ error contract with NO respawn (the server is alive; respawning
fixes nothing); 5xx → one supervised respawn, then the error
contract. Empty transcript → idle without a model call.

## 9. Testing

Fake-server tests (no weights): VAD gate blocks silent audio and
passes speech (`tasks/run` shape asserted byte-level,
`options.threshold` placement asserted); pre-submit failure returns
the §8 dict at both call sites with floor `IDLE` and no turn spawned;
background failure logs via the turn callback with floor `IDLE`;
wavutil round-trips PCM↔WAV and strips TTS responses; supervisor
restarts a dead server; missing binary raises with the provision
command; pipeline never touches `open_session`; new clients follow the
async-generator convention; `voice_id` field shape asserted per task-1
confirmation; committed corpus fixtures exist. Pinned, not worded:
request shapes, segment/transition behavior — never model-generated
wording. Manual validation per §5 on the dev machine plus the §6
hardware checklist. Gates: full suite + `ruff check` green before any
completion claim.

## 10. Cutover

Direct cut, no compat flag: §3 file table executed in full (including
test dispositions and `pyproject.toml` + lockfile), same change,
baselines measured first (§5 order), suite green via mocks + fake
server. `README.md` module map updated; this spec's file list is the
authority until then. `docs/archive/` stays history.

Spec grade pending engineer review. On approval at 8+, invoke
writing-plans for the implementation plan.
