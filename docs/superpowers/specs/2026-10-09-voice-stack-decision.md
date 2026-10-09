# Voice stack decision — compiled for review (2026-10-09)

Everything verified this session. No speculation. Decisions needed from the
engineer at the bottom.

---

## 1. Where the code lives right now

| Thing | State |
|---|---|
| `main` (repo root) | Old stack, working: faster-whisper STT, bundled Silero VAD, PocketTTS.cpp helper |
| `.worktrees/m1-audiocpp-swap` | audio.cpp version, 18 commits, 322 tests green, **not merged, not deleted** |
| Both | Preserved. Nothing lost either way. |

The DC-bias bug exists on main (measured, see §4). The audio.cpp branch is a
reference you can merge later or ignore.

---

## 2. What we learned about the audio.cpp path

It works. All five performance bars passed (WER 0.069 vs tiny.en's 0.272,
transcription RTF 0.27, stop-to-transcript max 1.38 s, TTS latency 1.23×
PocketTTS). Costs:

- **1.5 GB disk** (ASR 997 MB, Supertonic 434 MB, binary 95 MB)
- **1.9 GB RAM resident** with all three models loaded (TTS-only config ≈ 450 MB)
- A second resident process, currently named `audiocpp_server`
- A 540-line provisioner doing per-platform binary + SHA pinning (hit an
  AVX-512-vs-portable split on this box already)
- A 0.x upstream; model licenses screened one by one

**What it genuinely bought:** swappable TTS weights (one config line, 11
engines), and better ASR. You said you only need one fixed TTS, so the
swappability is worth ~0 to you.

**Verdict: not worth merging today.** Keep it as the reference branch.

---

## 3. What we learned about the sherpa-onnx path (new)

One pip package covers all three jobs, pure Python, same resident-helper
pattern you already run. Verified:

| Job | Model | Size |
|---|---|---|
| TTS | `sherpa-onnx-supertonic-3-tts-int8-2026-05-11` | ~102 MB |
| ASR | `sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8` | ~200 MB (estimate) |
| VAD | Silero, bundled in-package | tiny |

Supertonic INT8 is a real published quant: 398 MB → 102 MB weights,
peak RSS ~1,026 MB → 327 MB, 31 languages, G2P-free (no espeak dependency
when bundling). Small community quant (46 downloads, one author) — quality
needs your ears, not the card's word.

Sherpa-onnx added Supertonic support in release v1.12.29.

**Files the swap touches on main:** `voice/stt.py`, `voice/pocket_helper.py`,
`voice/tts_assets.py`, plus the DC fix in `voice/capture.py`. Four files.
Flow identical.

---

## 4. The DC bias (real, but its effect on main is UNPROVEN)

Your mic driver adds a constant **+5,762** to every sample (measured on the
raw device path, stable across runs, 17.6% of full scale). That number is
not in dispute.

What **is** disputed, by you, correctly: whether it breaks anything on main.
My evidence — "the VAD gate drops every live utterance, press PTT, speak,
get nothing" — was measured against **the audio.cpp branch's Silero**, not
yours. They are different implementations with different thresholds and
preprocessing. Your lived experience is that transcription worked fine on
main, which means main's gate (faster-whisper's bundled Silero) either
doesn't care about the pedestal, or handles it.

So the honest status:

- The bias is real and measurable.
- It demonstrably breaks the **branch's** VAD.
- Its effect on **main** is untested and doubted.
- The fix (~31 lines in `capture.py`, subtract each chunk's mean + a 60 Hz
  high-pass) is harmless on clean audio by construction, so porting it to
  main is cheap defensive hygiene — **not** a proven bug fix. Call it that
  in any commit message.

The `git worktree add .worktrees/old main` check would settle it in two
minutes if you ever want certainty. Your report that it just worked is
currently the best evidence available, and it points the other way.

---

## 5. The Supertonic license gate (only real blocker)

Supertonic 3 is **BigScience OpenRAIL-M**, not CC-BY. Commercial use is
allowed; the use restrictions must ship as enforceable terms to your users.
The three that bite a voice assistant: machine-generated speech must be
disclosed as machine-generated, no non-consensual impersonation (voice
cloning), no medical advice. The INT8 quant carries the same terms.

Blocking until you sign off: `DISTRIBUTION-TERMS.md` + notices + the
disclosure wording. Drafted already on the branch (`LICENSES/`,
`DISTRIBUTION-TERMS.md`) — reusable as-is.

ASR licensing is a non-issue: you're not bundling Canary.

---

## 6. What this means for your roadmap

| Milestone | Feasible on main today? |
|---|---|
| Hands-free VAD auto-submit (M2) | Yes — `voice/stt.py`'s `STTSession` already does 250 ms-pause endpointing. Wiring work, ~2-3 h, one file, DC fix first |
| Hotkey barge-in | Already works |
| Talk-over interruption (M3) | Needs an echo canceller either way (SpeexDSP/WebRTC). Backend-independent |

---

## 7. What I need from you

1. ** sherpa-onnx swap: build it, spec it first, or park it?** My
   recommendation: build it — four files, ~2 hours, and it makes every later
   voice decision cheaper.
2. **The INT8 Supertonic quality spike: run it?** 20 minutes — synthesize the
   same sentence at fp32 and INT8, measure, write WAVs to `/tmp/opencode/` for
   your ears. Do this *before* committing to INT8 in the app.
3. **Merge or keep the audio.cpp branch as reference?** My recommendation:
   keep. Merging buys you nothing you asked for today.

Optional, 2 minutes, tells us whether the DC bug predates the migration:

```
git worktree add .worktrees/old main
cd .worktrees/old && uv sync
uv run verbix --voice        # speak a sentence, watch what comes back
```
