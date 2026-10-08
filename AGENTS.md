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

- `minisweagent/` is being reshaped into our own agent (verbix agent
  branding, `verbix/agent/` package). Its config-list tool assignment is
  the mechanism we keep; the pollution around it goes.
- Three agents, one engine: foreground (voice), worker, announcer.
  Foreground and announcer work hand in hand as one entity: same priority,
  same floor rights, same chat hook. The announcer is the foreground
  without tools.

## Workflow

- Plain formatting, no jargon, English only.
- No implementation without an approved plan or slice. Brainstorm first,
  spec and plan for architectural work, graded 8+ before building.
- TDD: failing test first, then minimal code, then green suite.
- Full test suite plus `ruff check` green before any completion claim.
  Bare-pipe `| tail` is forbidden; use `set -o pipefail`.
- Closed-source commercial track.
