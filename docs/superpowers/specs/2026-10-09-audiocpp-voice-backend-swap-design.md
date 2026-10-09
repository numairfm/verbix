# M1 Spec — audio.cpp voice backend swap (VAD + STT + TTS, PTT unchanged)

Milestone 1 of 3. M2 adds the VAD-gated hands-free loop. M3 adds voice
interruption with external AEC. Each milestone ships on its own.

Revision 2: addresses critic rounds (accuracy, license, platform).
Open question count is zero; every former question is decided below.

## 1. Intent

Replace Verbix's scattered Python voice backends (faster-whisper STT,
PocketTTS.cpp helper, Silero-through-faster-whisper) with one native
runtime, audio.cpp, covering VAD + ASR + TTS through a resident local
server. Interaction stays push-to-talk, exactly as the user experiences
it today, except voice quality and latency characteristics (bounded in
§5). No behavior the user operates changes: same hotkey/toggle,
same floor states, same barge-in.

Locked decisions: half-duplex interaction (M2), server subprocess
integration (not C ABI, not CLI-per-request), full swap including
Supertonic 3, three milestones, no heuristic echo hacks,
cross-platform in M1 (Linux/Windows/macOS).

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

- `voice/audiocpp_server.py` (new): owns the process only — spawn,
  `GET /health` wait, crash restart with model re-registration (the
  server keeps loaded models/sessions in memory per process, so a
  restart re-registers entries and later requests open fresh sessions;
  nothing survives a process death by design), `aclose()`. Binds
  `127.0.0.1`. No token auth: the server documents no auth mechanism,
  and localhost binding is its documented posture.
- `voice/audiocpp_vad.py` (new): one-shot no-speech gate only —
  `POST /v1/tasks/run` with the configured Silero model id and
  `{"request": {"audio_base64": "<wav>"}}`, returns speech/no-speech
  from the result segments (same segment contract the CLI exposes via
  `--segments-out`). No streaming session in M1; M2 adds it. Gate
  failure (server down mid-session) is fail-closed: the turn errors to
  the existing deterministic voice error path. Fail-open is wrong here
  because a dead server also means STT/TTS are dead — silence would be
  a lie, while today's local gate fails open only because the decoder
  still works without it.
- `voice/audiocpp_stt.py` (new): one-shot `transcribe(buffered_pcm)`
  via `POST /v1/audio/transcriptions` (multipart upload, OpenAI shape).
  It does NOT implement `open_session`: M1 drops live incremental STT
  during recording (see §4), so there is no live route to consume.
- `voice/audiocpp_tts.py` (new): `synthesize(text)` as an async
  generator function (`async def` + `yield`, the convention
  `PocketHelperEngine.synthesize` and both mocks already follow) via
  `POST /v1/audio/speech` with the Supertonic model id and preset
  voice id. Server returns `audio/wav` by default; the client strips
  the WAV header and yields 24 kHz f32 mono PCM (numpy conversion, both
  already dependencies), matching what `AudioPlayer`/sinks consume
  today (`sinks.py:21-23`, `player.py:63-64` defaults). Voice request
  sends the preset id only (server precedence level 2: preset wins;
  no `voice_ref`, no `voice_dir`). Preset is `M1`, provisional pending
  the provenance gate (§7).
- `voice/capture.py` (edited, not untouched): adds a sounddevice input
  backend alongside `pw-record`/`arecord` (`capture.py:27-32` today is
  Linux-only: with neither binary, recording silently yields `b""` and
  the turn idles). Backend order: `pw-record` → `arecord` →
  sounddevice (lazy import, same pattern as
  `callback_sink.py:55`). sounddevice is already a dependency
  (`pyproject.toml:28`). No new dependency.
- `voice/pipeline.py` (edited minimally): the `open_session` live
  branch (`pipeline.py:85-88`, finalize-preference `106-108`) is
  deleted; stop always does one-shot transcribe. Everything else —
  toggle flow, barge-in, floor transitions, empty-transcript idle —
  is byte-for-byte behavior-identical.
- `voice/base.py` (edited): protocols untouched; mocks extended to
  cover the new clients' shapes. (Earlier "untouched" wording was
  wrong; this is the correction.)
- `voice/__init__.py` (edited): lazy re-exports
  (`__init__.py:22-34`) switch from `PocketHelperEngine` /
  `FasterWhisperSTT` to the new clients; old names raise
  `AttributeError` as unknown names do today.
- `src/verbix/cli.py` (edited): voice-mode wiring (`cli.py:306-312`)
  constructs the supervisor + new clients; `ensure_tts_runtime` call
  replaced by voice-runtime provisioning check with setup guidance.
- Deleted in the same change: `voice/stt.py`, `voice/pocket_helper.py`,
  `voice/tts_assets.py`, `scripts/setup_tts_runtime.sh`,
  `scripts/build_tts_helper.sh`, `scripts/fetch_tts_assets.py` (all
  PocketTTS-specific; the last two serve only the deleted setup path).
- Untouched: `voice/output.py`, `voice/player.py`, `voice/sinks.py`,
  `voice/callback_sink.py`, `voice/segments.py` (imported by
  `agent/agents/default.py:44`, stays), `runtime/`, `core/`, `agent/`,
  `scripts/benchmark_toggle_voice.py`, `scripts/spike_xdg_shortcuts.py`.

audio.cpp surface relied on (all verified in-repo, not assumed):
server keeps one loaded model + session per id across requests and
documents `GET /health`, `GET /v1/models`, `POST /v1/audio/speech`
(wav default, JSON-base64 and streaming options), `POST
/v1/audio/transcriptions` (multipart/JSON-path, plus `/details` and
`/live` variants), generic `POST /v1/tasks/run` (`model` + `request`
object, `audio`/`audio_base64` WAV input), config-driven `server.json`
with `backend` cpu/cuda/vulkan/metal, Silero `vad` offline+streaming
with `--segments-out` segment contract, Supertonic 31 languages with
offline+streaming GGUF and 10 preset voices (`M1`-`M5`/`F1`-`F5`).
Implementation task 1 confirms three details against a live local
server and records them here: VAD segment result shape over HTTP,
Supertonic output rate/channels, WAV header size. Routes are pinned
from docs; only value-level details are confirmed live.

## 4. Data flow (M1)

Toggle → capture records → stop → full-buffer bytes → VAD gate (one
HTTP call) → silence: `IDLE`, no model call → speech: one-shot
transcribe → `runtime.submit` → agent turn → TTS per-sentence
synthesize → playback drains → `IDLE`. Deliberate change vs today:
finalize pays full-buffer ASR instead of tail-only, because the live
session is gone; §5 budgets it. Everything the user operates —
toggles, barge-in, floor states, `delivered_text` trimming — is
unchanged.

## 5. Model selection and validation (concrete)

- VAD: Silero, bundled `assets/framework/models` (1.2 MB, MIT). Fixed.
- TTS: Supertonic 3, default `orig` package; `q8_0` only if CPU
  first-chunk latency misses the bar below. Voice `M1` provisional.
- ASR shortlist with pre-cleared licenses: `fun_asr_nano`
  (Apache-2.0), `qwen3_asr` (Apache-2.0), `nemotron_asr` (OpenMDW-1.1:
  permission granted, retain agreement copy + origin notices on
  distribution, patent-retaliation termination, no output restrictions,
  clear third-party rights yourself — compatible, obligations recorded
  in `THIRD-PARTY-NOTICES.md`), `parakeet_tdt` nvidia variant
  (CC-BY-4.0) ONLY — the `oruk/orukeet` CC-BY-SA-4.0 share-alike
  variant is explicitly excluded (copyleft trap for closed-source).
  `parakeet_tdt` is a community port: weaker support ownership, weighed
  against it. ASR stays `UNRESOLVED` until a package id is pinned.
- Validation corpus: 10 fixed English sentences (listed in the
  implementation plan), recorded once on the dev machine, committed
  under `tests/fixtures/voice/` (16 kHz mono WAV, small). The repo
  currently contains zero WAVs, so the corpus is created, not found.
- Bars (measured on the dev machine, server version + backend recorded
  with results): ASR WER on the corpus no worse than `tiny.en` on the
  identical audio; transcription real-time factor ≤ 0.5 on CPU;
  stop-to-transcript p95 ≤ 3 s for ≤ 15 s utterances; TTS first-chunk
  latency no worse than 1.25× today's PocketTTS on the same sentences.
  Silence/noise response must be empty output (no hallucinated
  punctuation — the reason today's gate exists).

## 6. Cross-platform shipping

Prebuilt matrix (verified): Windows x64 CPU/Vulkan/CUDA (CUDA runtime
in a separate archive that must sit next to the binaries), Ubuntu x64
CPU/Vulkan, macOS arm64/x64 Metal (build via the documented Metal
script). Provisioner is `scripts/provision_voice.py` — Python stdlib
only, so it runs wherever `uv` runs; no `.sh`-on-Windows problem.
Model installs go through the Python v2 model manager (no native
manager build needed). Version strategy decided: pin release TAG, and
the provisioner verifies per-OS artifact SHA256 at download (hashes
recorded in the script).

Per-OS rules: Windows — CPU flavor default, CUDA flavor only with the
cudart archive present, backend `cpu`/`cuda` to match; macOS — Metal
build, backend `metal`; Linux — backend `cuda` only with a local
toolkit, else `cpu` (Vulkan opt-in flag). Fedora rule: try the Ubuntu
binary, smoke-check (`--list-devices`, `/health`, one VAD/ASR/TTS
round-trip); on failure, CPU source build (GCC 13+). Gate: provisioner
exits 0 only after health + all three round-trips pass.

Layout keeps today's root convention with a new voice runtime dir;
`VERBIX_VOICE_DIR` overrides, installed fallback as today. Legacy
`VERBIX_TTS_DIR` / PocketTTS runtime contents are incompatible: if
detected, voice startup raises naming the re-provision command (fail
loudly, no silent reuse). No OS-only behavior without a same-behavior
fallback. Windows/macOS capture paths are validated by the engineer on
hardware; Linux paths run in CI via fakes.

## 7. License gates (all closed in this change)

Framework audio.cpp is Apache-2.0 (ShugoAI LLC) — clear. Silero VAD is
MIT — clear. Supertonic 3 is BigScience OpenRAIL-M (license text read
on its Hugging Face page; audio.cpp marks commercial use allowed with
restrictions passed on). The FULL restriction set binds Verbix, not a
trio: (a) obey all applicable law; (b) no harming/exploiting minors;
(c) no verifiably false information to harm others; (d) no harmful
PII; (e) machine-generated content must be expressly disclaimed;
(f) no defamation/disparagement/harassment; (g) no impersonation
(deepfakes) without consent — voice cloning implicates this; (h) no
fully automated decisions adverse to legal rights or binding
obligations; (i) no discrimination on behavior/predicted traits;
(j) no exploiting group vulnerabilities causing harm; (k) no
discrimination on protected characteristics; (l) no medical advice or
results interpretation; (m) no justice/law-enforcement/immigration
prediction uses. Deliverables in this change: license copy shipped
under `LICENSES/`; `THIRD-PARTY-NOTICES.md` entry (hyphens — the
underscored name used in revision 1 was wrong) retaining all notices;
a user-facing distribution terms document carrying the §4a
restrictions as enforceable terms with the §4b license pointer, the §6
output-accountability note, and the §7 latest-version duty (server
version pinned per §6, re-validated on bump). Legal sign-off blocks
release; anything unclear stays `UNRESOLVED` and blocks cutover.
Preset voice `M1` is provisional pending a provenance/consent gate,
same duty as the Alba voice-likeness clearance tracked in
`THIRD-PARTY-NOTICES.md:19-22` (personality rights are not covered by
model licenses).

## 8. Error handling

Loud failures only: missing binary/weights/unhealthy server raise with
the exact provision command — never silent fallback. Server crash
mid-turn: supervisor respawns, re-registers model entries (a restarted
process owns no prior sessions), in-flight turn errors to the existing
deterministic voice error path. Bounds: health wait 10 s total;
per-request timeouts — VAD gate 15 s, STT `max(30 s, 4× audio
duration)`, TTS 60 s per sentence. Mapping: connection refused /
missing model entry → setup guidance; 5xx → one supervised restart,
then the error phrase; empty transcript → idle without a model call.

## 9. Testing

Fake-server tests (no weights): VAD gate blocks silent audio and
passes speech; gate failure errors (fail-closed); STT posts multipart
and returns text; TTS strips WAV to 24 kHz f32 chunks; supervisor
restarts a dead server and re-registers; missing binary raises with
the provision command; pipeline never touches `open_session`; new
clients follow the async-generator `synthesize` convention. Pinned, not
worded: request shapes, segment/transition behavior — never
model-generated wording. Manual validation per §5 on the dev machine
plus engineer hardware checks for Windows/macOS capture. Gates: full
suite + `ruff check` green before any completion claim.

## 10. Cutover

Direct cut, no compat flag: new clients in, §3 deletions executed,
same change, suite green via mocks + fake server. `README.md` module
map updated (stale on `runtime/` paths today); this spec's file list is
the authority until then. `docs/archive/` stays history.

Spec grade pending engineer review. On approval at 8+, invoke
writing-plans for the implementation plan.
