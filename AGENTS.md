# AGENTS.md — how to work with the engineer

The user is the engineer. The agent is not. Decisions run by the user in a
way both sides understand completely, in and out: the solution plus its
proof (failing test first, passing suite, file:line pins, grades). The user
stays involved in the project; the agent shows its work so the user can
correct it.

## Code shape

- Modular: one thing does one thing, ideally one thing per file. A tool bug
  is fixed in that tool's file plus its test. Nothing more.
- Simple: the minimal code that drives the system. No scaffolding for later.
- Tidy always. Tidy helps the humans, and it helps the agent more than it
  looks: small focused files mean reliable edits and fast reviews.
- Dynamic tools: write `tools/<name>.py`, add the name to the agent yaml
  `tools:` list, and it works. Remove it and it stops. Unknown names fail
  loudly, never silently.

## Project direction

- `agent/` is our own agent (`verbix/agent/` package). Its config-list tool assignment is
  the mechanism we keep; the pollution around it goes.
- Three agents, one engine: foreground (voice), worker, announcer.
  Foreground and announcer work hand in hand as one entity: same priority,
  same floor rights, same chat hook. The announcer is the foreground
  without tools.

## Product constraints

- What Verbix is: voice-operated desktop assistant. Talk, keep talking while
  work happens in the background, hear each result in the same voice.
  Main flow: toggle, speak, transcribe, agent works, generate speech,
  hear answer.
- Seams: `agent/` does work, `voice/` owns the floor, `ipc/` owns the socket,
  `core/` owns the loop, `provider/` owns all model wire behavior (HTTP, auth,
  SSE/JSON, retries). Provider wire never lives in the agent loop.
  `README.md` is the map. `docs/archive/` is history. Code wins on conflict.
- Shipping: one command sets up the app (`uv sync`,
  `uv run scripts/fetch_voice_models.py`). Helpers and weights fetch automatically.
  Never require manual copy into app folders.
- Commercial: per-change license review for any new dependency, model, voice,
  font, icon, or audio. `THIRD_PARTY_NOTICES.md` is canonical and ships with
  the app. Unclear licensing is `UNRESOLVED`, never assumed allowed.
- Cross-platform: behavior is identical on macOS, Windows, Linux. No OS-only
  behavior without a same-behavior fallback. Keep platform code behind a narrow
  seam.
- Model-visible text: only user messages and text Verbix actually spoke carry
  prose. Tool observations return terse status or raw evidence. The `assistant`
  role is reserved for spoken text. Product-owned task wording is fixed strings
  in code, never model narration. Tests pin observations and state transitions,
  never model wording.

## Workflow

- Plain formatting, no jargon, English only.
- No implementation without an approved plan or slice. Brainstorm first,
  spec and plan for architectural work, graded 8+ before building.
- TDD: failing test first, then minimal code, then green suite.
- Full test suite plus `ruff check` green before any completion claim.
  Bare-pipe `| tail` is forbidden; use `set -o pipefail`.
- Closed-source commercial track.
