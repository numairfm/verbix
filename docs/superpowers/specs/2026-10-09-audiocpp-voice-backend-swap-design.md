# M1 Spec — audio.cpp voice backend swap (VAD + STT + TTS, PTT unchanged)

Milestone 1 of 3. M2 adds the VAD-gated hands-free loop. M3 adds voice
interruption with external AEC. Each milestone ships on its own.

## 1. Intent

Replace Verbix's scattered Python voice backends (faster-whisper STT,
PocketTTS.cpp helper, Silero-through-faster-whisper) with one native
runtime, audio.cpp, covering VAD + ASR + TTS through a resident local
server. Interaction stays push-to-talk, exactly as today — no behavior
change the user can hear except voice quality.

Locked decisions from brainstorming: half-duplex interaction (M2), server
subprocess integration (not C ABI, not CLI-per-request), full swap
including Supertonic 3 (not PocketTTS kept), three milestones, no
heuristic echo hacks, cross-platform from day one.

Success: `uv run verbix --voice` works on Linux/Windows/macOS with no
Python audio ML in-process, ASR quality at parity with `tiny.en`,
Supertonic voice acceptable, suite + `ruff` green, license review done.

## 2. Goals / non-goals

Goals: resident `audiocpp_server` supervised by Python; VAD/STT/TTS HTTP
clients behind the existing `voice/base.py` protocols; Supertonic 3 as
the TTS voice; one-command provisioning per OS; per-change license
review with `THIRD_PARTY_NOTICES.md` updated; old backends deleted in
the same change.

Non-goals: VAD-gated auto-submit (M2), voice interruption / AEC (M3),
microphone capture changes (`capture.py` stays), runtime/floor/announcer
changes, ASR model perfection (parity is the bar, upgrades later).

## 3. Architecture

`voice/` mirrors its current file-for-file layout, one concern per file:

- `voice/audiocpp_server.py` (new): owns the process only — spawn,
  `GET /health` wait, token auth, crash restart, `aclose()`. Same shape
  as `voice/pocket_helper.py:31-132`.
- `voice/audiocpp_vad.py` (new): streaming Silero session client. M1
  uses it only as the no-speech gate (today's `stt.py:174-184` behavior);
  the session stays warm for M2.
- `voice/audiocpp_stt.py` (new): implements `SpeechToText.transcribe`
  (`voice/base.py:9-14`) via the server transcription route.
- `voice/audiocpp_tts.py` (new): implements `TextToSpeech.synthesize`
  (`voice/base.py:17-23`) via `POST /v1/audio/speech`, Supertonic,
  preset voice fixed in config.
- Deleted in the same change: `voice/stt.py` (faster-whisper),
  `voice/pocket_helper.py`, `voice/tts_assets.py`,
  `scripts/setup_tts_runtime.sh` (superseded, see §6).
- Untouched: `voice/base.py` protocols + mocks, `voice/capture.py`,
  `voice/output.py`, `voice/player.py`, `voice/sinks.py`,
  `voice/pipeline.py` (PTT flow as-is, only the injected backends change),
  `runtime/`, `core/`, `agent/`.

audio.cpp surface this relies on (verified in-repo): server keeps one
loaded model + one session per model id across requests
(`audio.cpp/app/server/README.md:3`); `GET /health`, `GET /v1/models`,
`POST /v1/audio/speech` (wav default, JSON base64 option, streaming);
config-driven `server.json` with `backend` cpu/cuda/vulkan/metal;
Silero VAD MIT, 1.2MB, bundled (`assets/framework/models`), streaming;
Supertonic 31 languages, offline + streaming GGUF, 10 preset voices.
The exact VAD/ASR request route shapes are confirmed against a live
local server in the first implementation task; this spec is amended if
they differ.

## 4. Data flow (M1, unchanged PTT)

Toggle → `capture.py` records → stop → VAD no-speech gate via
`audiocpp_vad` → speech present → `audiocpp_stt.transcribe` once →
`runtime.submit` → agent turn → `audiocpp_tts.synthesize` sentence
stream → `AudioPlayer` → drain → `IDLE`. Empty transcription returns to
`IDLE` with no model call, as today (`voice/pipeline.py:112-118`).
Barge-in, floor states, and `delivered_text` trimming are untouched.

## 5. Model selection (validated, not guessed)

- VAD: Silero, bundled, no download. Fixed.
- TTS: Supertonic 3, default `orig` package; fall back to `q8_0` only if
  CPU latency on the dev machine misses the parity bar (first-chunk
  latency vs PocketTTS today, judged by ear + stopwatch, not by
  audio.cpp's CUDA benchmarks).
- ASR: shortlist from the supported table — `qwen3_asr`,
  `nemotron_asr` (streaming), `fun_asr_nano`, `parakeet_tdt`.
  Validation: English transcription spot-check against `tiny.en` on the
  repo's own voice samples plus live mic, and CPU real-time factor on
  the dev machine. Winner pinned by package id in the setup script.
  English parity is the M1 bar; wider languages follow the winner.

## 6. Cross-platform shipping

audio.cpp publishes prebuilt binaries for Windows x64
(CPU/Vulkan/CUDA, CUDA runtime in a separate archive), Ubuntu x64
(CPU/Vulkan), and macOS arm64/x64 (Metal), plus build scripts for
source builds (GCC 13+, VS 2022, Xcode). Dev machine is Fedora KDE:
first try the Ubuntu binary, else CPU source build.

`scripts/setup_voice_runtime.sh` (new) provisions everything in one
command per OS: fetch or build `audiocpp_server`, install pinned GGUF
packages via the model manager, verify `GET /health` plus one VAD /
ASR / TTS round-trip. No manual binary copying, no weights committed
to git. Same-directory layout convention as today (`tts-runtime/`
pattern, installed fallback) so existing overrides keep working in
spirit. No OS-only behavior without a same-behavior fallback.

## 7. License review (in this change, not after)

Framework audio.cpp is Apache-2.0 (repo `LICENSE` header, ShugoAI LLC)
— fine. Weights are separate (`README.md:85`): Silero VAD is MIT —
fine. Supertonic 3 is BigScience OpenRAIL-M (license text read on its
Hugging Face page; audio.cpp `docs/model_licenses.md` marks commercial
use allowed with restrictions that must be passed on). That means:
ship the license copy, pass the use restrictions through as enforceable
terms for downstream users, and note the three that bite a voice
assistant — machine-generated speech must be disclosed as such, no
impersonation without consent (voice cloning), no medical advice. The
chosen ASR family's weight license gets the same review before it is
pinned. Anything unclear is `UNRESOLVED` and blocks the cutover, never
assumed allowed. `THIRD_PARTY_NOTICES.md` updated in the same change.

## 8. Error handling

Loud failures only, matching existing patterns (`tools/sandbox.py`,
`voice/pocket_helper.py:76-80`): missing binary, missing weights, or
unhealthy server raise with the exact setup command — never silent
fallback to another backend. Server crash mid-turn: supervisor
restarts, sessions re-created, in-flight turn errors to the existing
deterministic voice error path (no new prose). HTTP timeouts bounded;
no unbounded retries. Empty/no-speech stays silent without a model
call.

## 9. Testing

- New fake-server tests (no weights): VAD gate blocks silent audio,
  STT client posts bytes and returns text, TTS client streams chunks,
  supervisor restarts a dead server, missing binary raises with setup
  guidance. Mock protocols in `voice/base.py` extended, never weakened.
- Pinned, not worded: observations, request shapes, state transitions —
  never model-generated wording.
- Manual validation on the dev machine: PTT turn end to end, ASR parity
  spot-check (§5), first-chunk TTS latency, server restart mid-turn.
- Gates: full suite + `ruff check` green before any completion claim.

## 10. Cutover

Direct cut, no compat flag: new clients in, old backends deleted, same
change, suite green via mocks + fake server. `README.md` module map
updated (it is currently stale on `runtime/` paths); this spec's file
list is the authority until then. `docs/archive/` stays history.

## 11. Open questions

1. Which ASR package wins validation (§5)?
2. Supertonic `orig` vs `q8_0` on dev-machine CPU?
3. Ubuntu `audiocpp_server` binary on Fedora as-is, or source build?
4. Server version pin strategy (release tag vs build hash)?

Spec grade pending engineer review. On approval with grade 8+, invoke
writing-plans for the implementation plan.
