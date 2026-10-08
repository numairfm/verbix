# Tool Output Rework Design

**Goal:** Tool results carry photos as photos, caps touch text only, overflow
spills to disk instead of being deleted, limits live in config. Replicates
OpenCode's proven pattern, ported to plain Python.

**Status:** Spec for review before plan. No code yet. Round 2 (round 1
graded 5/10; every finding below is addressed inline).

## Why now (pinned to code)

- `src/verbix/tools/read.py:65-83` (`_read_image`, `base64.b64encode` at
  `:79`, marker string at `:80-83`) returns a photo as base64 inside one
  flat string. Measured live 2026-10-08 via agent-path replay: the
  2,504,992-byte workspace photo reached the model as 10,032 elided chars
  of base64 header, zero image blocks.
- `src/verbix/agent/agents/default.py:191-204` caps most observations at
  10,000 chars by default (`spawn.py:15`, `status.py:15`, `cancel.py:15`
  set 1500; the 10,000 default is stamped at `tools/__init__.py:76`). The
  number is inherited from the old observation template (5,000 head +
  5,000 tail), not chosen. It cannot tell pixels from text.
- The image-lifting code (`actions_toolcall.py:108-120`, shape built at
  `:115-117`) sits on the sync path workers never take: `step_stream`
  builds tool messages from raw strings at `default.py:460-465` with no
  expansion, while the sync formatter is reachable only via
  `litellm_model.py:330-343` from `default.py:609-610`. Workers take
  `step_stream` (`runtime/worker.py:90-95`).
- Downscale constants below (1568 px, JPEG 82) are chosen, not OpenCode's
  (their resize path was not read). Justification is measured: text-embedded
  base64 runs ~3.3M chars (~800K text tokens — impossible on any request
  budget), while provider-metered image parts go through (a ~110 KB photo
  probe succeeded on the ~15k metered key). Downscale is a cost control
  with a measured reason, not a dodge around a text cap.
- Pillow is not a project dependency (no match in `pyproject.toml`), so no
  tool may downscale today.
- Live symptom (user-reported voice session 2026-10-08, photo independently
  confirmed as clouds): the model guessed bee, spider, flower across turns.

## OpenCode pattern (read from their source, not memory)

- Results are typed text/file parts
  (`packages/core/src/tool/tool.ts:38`); reads return photos as file parts
  (`packages/core/src/tool/read.ts:50`).
- `bound()` splits media from text, previews text only, media passes
  untouched; full text spills to disk with its path in the marker, 7-day
  retention (`packages/core/src/tool-output-store.ts:138-174`, defaults
  `MAX_LINES=2000`, `MAX_BYTES=50*1024` at `:13-14`).
- Limits are config (`tool_output.max_lines/max_bytes`), defaults 2,000
  lines / 51,200 bytes
  (`packages/core/src/v1/config/config.ts:136-148`).

## Design

### 1. Typed attachments on ToolResult (`tools/base.py:11-14`)

- Add `FileAttachment`: frozen dataclass, `data: bytes`, `mime: str`,
  `name: str`.
- Add `attachments: tuple[FileAttachment, ...] = ()` to `ToolResult`.
  `observation` stays the text channel. Every existing
  `ToolResult("...")` construction keeps working untouched.

Look right so far?

### 2. Caps see text only; overflow spills (in `tools/shared.py`, no new file)

- New `bind_output(text, *, max_lines=2000, max_bytes=51200,
  spill_dir: Path | None) -> str`: port of their preview — head/tail lines
  first, then byte split. Under limits returns text unchanged. Over limits:
  full text goes to `<workspace>/.verbix/logs/tool_<uuid4hex12>`
  (inside the workspace so the `read` containment rule lets workers
  re-open it), marker `... output truncated; full content saved to
  <workspace-relative path> ...` takes their wording. Each over-limit
  re-read writes a new file; the 7-day prune below bounds growth.
- Bind runs in the registry `_make_handler` (`tools/__init__.py:72-77`,
  the one place with the session `ToolContext`), not in `_execute_tool`
  (which has no workspace). `.verbix/` is the machine-owned namespace in
  the workspace: `logs/` holds spilled outputs today, `transcripts/` or
  siblings can join later without disturbing it. `build_registry` gains
  `output_limits: tuple[int, int] | None = None` (stdlib typing, defaults
  to the module constants); the two call sites pass the yaml override
  through: `workers.py:72` from `agent_cfg_dict` (loaded at
  `workers.py:50`), `foreground.py:342` from `agent_cfg_dict` (loaded at
  `foreground.py:311`). New optional `AgentConfig.tool_output` dict carries
  it into config.
- Kill site (exact): `default.py:203-205` must become
  `ToolResult(obs, is_error=..., is_terminal=...,
  attachments=res.attachments)` (same locals, plus preservation). A
  literal implementation that reconstructs without `attachments=`
  reproduces the incident with a new marker string.
- Deletions (no two cap systems coexist): `cap_observation`
  (`default.py:88-93`), `observation_cap` on `spawn.py:15`,
  `status.py:15`, `cancel.py:15`, `bash.py:12`, and the stamp at
  `tools/__init__.py:76`.
- Tool message (`default.py:460-465`) becomes one text part plus one
  `image_url` part per attachment
  (`{"type": "image_url", "image_url": {"url": "data:<mime>;base64,..."}}`),
  the shape at `actions_toolcall.py:115-117`.
- List-content fallout (list tool messages are new; three str-only
  consumers choke on them): widen `CanonicalMessage.content`
  (`core/models.py:29`, today `content: str = ""`) to
  `str | list[dict[str, Any]]`;
  the adapter (`foreground.py:195-203`, `content=m.get("content") or ""`)
  and the history proxy (`foreground.py:90-120`, same pattern) pass list
  content through, eliding image parts to `[image elided]`; same guard at
  the status reader (`worker.py:136-137`, `content.strip()`). Otherwise
  the first `status` after a photo task crashes the turn.
- Spill prune hooks into `spawn_worker` (`workers.py:34`): files older
  than 7 days removed before the worker starts.
- Compaction (`agent/utils/compaction.py`): alongside the `isinstance str`
  elisions at `:90` and `:105`, list-content tool messages get image parts
  replaced with `{"type": "text", "text": "[image elided]"}` keeping text
  parts — otherwise a ~100 KB base64 part permanently occupies budget and
  `estimate_tokens` (`:21-28`) starves everything around it.

Look right so far?

### 3. Photos ride as attachments (`tools/read.py`)

- Image branch (`read.py:65-83`) downscales with Pillow (added to project
  dependencies): longest edge 1568 px, RGBA flattened onto white, first
  frame only for GIF, JPEG quality 82, metadata stripped. The 10 MB
  pre-downscale refuse (`MAX_IMAGE_BYTES`) is deleted with it — size is
  handled by shrinking, and Pillow's own decompression-bomb guard turns
  absurd dimensions into an ordinary error result. Returns
  `ToolResult(observation=f"[image attached: {name}]",
  attachments=(FileAttachment(data, mime, name),))`. Marker strings stop
  being produced.
- Legacy sync path (`default.py:521-610`, reachable in production only via
  the never-firing `worker.py:102` fallback; otherwise tests): left
  untouched. Photos through it degrade to the `[image attached]` text
  without pixels — documented, not silent. The marker regex machinery stays
  for anything still emitting markers.
- `read` description gains one line: line args are ignored for images and
  directories. Steadies the argument variance seen live.

Look right so far?

### 4. Data flow

Tool `execute` -> `ToolResult(text + attachments)` -> handler binds and
spills text only -> tool message is text part plus image parts -> provider.
A 200 KB build log becomes a preview plus a re-readable path; a 2.5 MB
photo becomes a ~100 KB image part the model actually sees.

### 5. Error handling

- Oversized images are downscaled, never refused (their ingest refuses past
  20 MB; ours shrinks instead — friendlier on a small key, same spirit).
  Pillow's decompression-bomb guard is the backstop for absurd dimensions.
- Spill write failure degrades to capped text with no path.
- Non-image tools are behavior-identical except overflow now spills instead
  of deleting. Each re-read of still-over-limit output writes one new spill
  file; the prune bounds it.

### 6. Testing

- Photo pin, end to end (not `read` in isolation): worker `read` of the
  2.5 MB photo through `_execute_tool` into the `step_stream` tool message
  yields exactly one attachment under ~150 KB and one `image_url` part —
  asserting at `default.py:203-205` and `:460-465`, the two kill sites.
- Spill pin: 200 KB `bash` output returns a preview plus a path the worker
  can `read` back verbatim.
- List-content pin: a photo tool message passes through the adapter,
  history proxy, and `status` without crashing (image parts elided to
  text where those consumers need strings).
- Config pin: yaml `tool_output` override changes the preview boundary.
- Full suite plus `ruff` green; Pillow present in dependencies.

## Grading criteria (this spec is graded on this, 10 points)

1. Truthfulness (3): every code claim pinned to a file:line above; no
   invented paths. Fail if any pin is wrong.
2. Contract completeness (3): all four behaviors present — typed media,
   text-only caps, spill plus re-read, config limits — with the kill-site
   rewrite exact and no cap able to touch pixels. Fail otherwise.
3. Minimal diff (2): fewest files (`shared.py`, no new module), dead cap
   code deleted not coexisting, existing `ToolResult("...")` calls
   untouched. Fail if the plan could delete a file and still work.
4. Human review (2): every behavior locatable by filename; workspace,
   limits, prune, and yaml binding each name their file. Fail if logic
   hides where filenames don't suggest.

Pass is 8 or more. Max 5 grading rounds (currently round 2).
