# Hands-Free Parity Plan (canonical — supersedes all chat milestone drafts)

> **For agentic workers:** implement slice by slice, in order, one worker at a time
> (all slices share `src/verbix/voice/pipeline.py`).
> Global constraints: NO git commits. TDD (failing test first). Full suite +
> `ruff check` green before any completion claim. PTT paths stay green throughout.
> Simplify, no shortcuts: no new decoder, no backend swap, no OS AEC, no new
> config knobs beyond what's listed, no output repeat caps.

Goal: continuous hands-free hearing matches press-to-talk word for word.
Dirty (speaker-playing) audio may trigger interrupts; it must never be decoded.

## Slice 1 — Fresh session at interrupt + 300ms look-back
- [x] Failing test: barge-in interrupt, then finalize yields only post-interrupt
      speech plus buffered onset; echo-era segments excluded.
- [x] In `_interrupt_active_speech` (`pipeline.py:163-180`): cancel the live
      session, open a fresh one with the same endpoint hook.
- [x] Rolling 300ms mic ring (≈9600 bytes @16kHz s16) fed from the capture
      callback regardless of gate; seed the fresh session with it on interrupt.
- [x] Cancel (not abandon) the replaced session so decode tasks don't leak
      (`pipeline.py:307` overwrites today — proven leak).
- [x] Run voice tests + `ruff check`.

## Slice 2 — Complete reference feed, no mid-echo resets
- [x] Route gap silences (`output.py:449`) and earcon cues (`:405`, `:420`,
      `:289-292`) through `on_audio` like all other yielded audio.
- [x] Delete the reference wipe on interrupt (`pipeline.py:171-172`) and on
      idle (`:250-251`); reference must survive while its echo is airborne.
- [x] Pass sink audio format explicitly into the reference feed (bound at
      construction); byte-sniff heuristic kept as fallback for direct
      callers (`aec.py:53-62`).
- [x] Run voice tests + `ruff check`.

## Slice 3 — Bounded chunks, quicker endpoint
- [x] Set `max_speech_duration` explicitly on the segmenter
      (`sherpa_vad.py:90-96`; field exists upstream, never assigned — proven).
      Done: `MAX_SPEECH_S = 20.0` (upstream default, verified live).
- [x] Retune trailing-silence endpoint 1.0s → 0.6s; no test pinned 1.0s.
      Live listening verification still open.
- [x] Delete dead code per evidence (only `player.is_playing` removed —
      dead producer proven; `pipeline.py:312,320,343-345`, `:182-199`).
      Outcome: only `player.is_playing` removed (dead producer proven).
      `turn_interrupted` KEPT deliberately: True-branch `pass` skips the
      clear+reset the else branch runs — always resetting wipes barge-in
      speech, never resetting leaks echo. Gate KEPT: load-bearing across
      the endpoint→finalize window. Neither had test cover for removal.
- [x] Run voice tests + `ruff check`.

## Slice 4 — Parity gate
- [x] Scripted echo test in an existing test file: known playback + known
      speech; continuous transcript must equal PTT transcript word for word.
- [x] Clean up Ctrl+C audio-thread shutdown noise (in user log).
- [x] Full suite + `ruff check` green. 309 passed at slice time.

## Slice 5 — Draft shelf (separate, last)
- [x] Delete auto re-speak on empty transcripts (`pipeline.py:286-296`;
      fires on silence — proven).
- [x] Keep the heard-pile snapshot for history repair (required — proven).
- [x] Stash unplayed text in a runtime side note (NOT in history: assistant
      role is spoken-text-only per `AGENTS.md:48-50` — proven).
- [x] Bare "continue"/"go on" (case-insensitive exact match, no fuzzy
      intent) hands the stashed text to the agent explicitly.
- [x] Run full suite + `ruff check` green.
