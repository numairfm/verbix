# Tool + Runtime Reorg Design

**Goal:** One tool lives in one file. Adding a tool name to an agent yaml makes it work. A tool bug is fixed in one file.

**Status:** Approved design A. Spec for review before plan.

## Why now (pinned to code)

- Tool code is split today: `src/verbix/tools/filesystem.py` holds read+write+edit, `src/verbix/runtime/tools.py:12-146` holds spawn+status+cancel+finish, `src/verbix/tools/bash.py` holds bash alone.
- Wiring ignores the yaml list in two places:
  - `build_worker_registry()` in `src/verbix/runtime/tools.py:150-173` hardcodes 4 tools, never reads `worker.yaml:25-30`. Adding `spawn` to `worker.yaml` tools silently does nothing.
  - Foreground registry in `src/verbix/runtime/runtime.py:424-450` hardcodes 3 tools. `ToolRegistry.get_definitions()` in `src/verbix/minisweagent/tools/registry.py:32-36` silently skips names missing from the registry, so a yaml typo also silently does nothing.
  - `bash` is listed in `worker.yaml:29` but never registered. It works only through the env fallback in `src/verbix/minisweagent/agents/default.py:225-252`. Second path for one tool.
- Engine is polluted with tool names:
  - `FOREGROUND_TOOL_NAMES` in `default.py:96-97` plus `_observation_cap` in `default.py:208-209`.
  - `finish` special case in `default.py:456-479` checks `name == "finish"`.
  - `BASH_TOOL` fallback in `default.py:198-204`.
- Result: a tool fix today touches tool file + `runtime/tools.py` + `default.py` + yaml + tests. Five files for one problem.

## Dream config (the contract)

1. Write `src/verbix/tools/<name>.py` implementing `Tool` in `src/verbix/tools/base.py:33-45` (`name`, `description`, `parameters_schema`, `execute(args, context)`).
2. Add `<name>` to the agent yaml `tools:` list (`voice.yaml:31-34`, `worker.yaml:25-30`, `announcer.yaml:14`).
3. Maybe adjust that agent prompt. Nothing else. It works.
4. Remove a name from the list and it stops being offered. No code change.
5. Unknown name in yaml fails loudly at startup, never silently missing.
6. One tool bug is fixed in `src/verbix/tools/<name>.py` plus its test. No engine edit.

## Design

### 1. Tools: one file per tool

- `src/verbix/tools/read.py` — `FileReadTool` (dir list, text slice, image open).
- `src/verbix/tools/write.py` — `FileWriteTool`.
- `src/verbix/tools/edit.py` — `FileEditTool`.
- `src/verbix/tools/bash.py` — keep, now a real registered tool (sandbox stays in `sandbox.py`).
- `src/verbix/tools/spawn.py`, `status.py`, `cancel.py`, `finish.py` — moved out of `runtime/tools.py`. Runtime-needing tools take `runtime` in the constructor (same as today `SpawnTool(runtime)` in `runtime/tools.py:18`).
- `src/verbix/tools/shared.py` — the only shared helper. Holds `resolve_contained_path` (today `filesystem.py:20-30`) and the `file_versions` stale guard helpers. Not a tool, never offered to the model. This answers shared.py plainly: the 10-line path-inside-workspace check plus the read-first guard, used by read/write/edit so it is fixed once.
- `src/verbix/tools/__init__.py` — holds `TOOL_CATALOG: dict[name, factory]`. No new file for the catalog (ponytail: fewest files). `build_registry(names, runtime, workspace)` lives here: looks up each yaml name in the catalog, raises `KeyError` naming the bad key plus the catalog keys on unknown, registers with `ToolRegistry.register`. Each registry owns ONE `ToolContext(workspace)` shared by its handlers (per-agent session, so the read-first stale guard in `base.py:18-30` actually accumulates) — this fixes a live bug proven 2026-10-08: `build_worker_registry` registers bare `tool.execute` which requires `(args, context)`, but `DefaultAgent._execute_tool` (`default.py:216`) calls `aexecute(name, args)` with no context, so every worker file-tool call fails with `missing 1 required positional argument: 'context'`. Registry handlers keep the full `ToolResult` (never flattened to `str` in the foreground handler at `runtime.py:440-446` — that flattening is deleted) so `is_terminal`/`is_error` survive to `default.py`. The registry branch of `_execute_tool` unwraps `ToolResult` directly (no `json.dumps` on the dataclass).
- Delete `src/verbix/tools/filesystem.py` and `src/verbix/runtime/tools.py` (finish moved, builder moved to `__init__.py`).

Look right so far?

### 2. Engine: default.py knows no tool names

- Delete `FOREGROUND_TOOL_NAMES` (`default.py:96-97`) and `_observation_cap` (`default.py:208-209`). Each tool file carries its own `observation_cap: int` (foreground tools set 1500, rest default 10000). `cap_observation` (`default.py:88-93`) stays generic, takes the cap from the tool.
- Delete the `name == "finish"` block (`default.py:456-479`). `finish.py` returns `ToolResult(is_terminal=True)` using the existing unused flag in `base.py:11-14`. `default.py` ends the turn on `is_terminal`, for any tool, no name check. The registry handler path preserves the `ToolResult` object end to end for this to work.
- Delete the `BASH_TOOL` fallback (`default.py:198-204`) and the env-execute second path (`default.py:224-252`). Bash is a registered tool like the rest. `env` stays on the agent only for `LocalEnvironment` process-group teardown, not for tool routing. Also delete the third bash default in `litellm_model.py:120` (`kwargs_dict.pop("tools", [BASH_TOOL])` becomes `pop("tools", None)`): no tools when the agent offers none.
- `get_tool_definitions` keeps its filtering role (`default.py:181-206`) but `ToolRegistry.get_definitions` (`registry.py:32-36`) changes from silent skip (`if name in self._tools`) to raising `KeyError` on any missing name. Fail-fast everywhere, including direct `DefaultAgent(config.tools=[...])` construction in tests — no silent typos on any path.

Look right so far?

### 3. Runtime: split by agent, thin coordinator

- `src/verbix/runtime/runtime.py` stays as the public facade (imports, `Runtime` class name, `spawn`/`status`/`cancel` methods keep signatures) so callers and tests do not churn. It becomes thin: construction + floor + ledger + broadcast.
- New `src/verbix/runtime/foreground.py` — builds the voice agent from `voice.yaml` (registry via `build_registry`, prompt grounding as today `runtime.py:521-538`).
- New `src/verbix/runtime/workers.py` — `spawn()` flow from `runtime.py:627-688` (config load, `LocalEnvironment`, `build_registry(worker names)`, `WorkerTask.start`).
- New `src/verbix/runtime/announcer.py` — announcer build from `announcer.yaml:1-17` plus `_speak_batch` from `runtime.py:847-887` and the drain from `718-740`. Keeps isolation (`messages = []`) and `tool_choice: none`.
- Layer rule for reviews: speech breaks belong to voice/output, loop breaks to runtime coordinator, announcer-calls-tools to `runtime/announcer.py` or `config/announcer.yaml`, tool misbehaves to `tools/<name>.py`. File path names the layer.

Look right so far?

### 4. Data flow

User text -> foreground (`foreground.py`) -> maybe `spawn` -> worker (`workers.py` + `tools/<name>.py` on `LocalEnvironment`) -> `finish` (`is_terminal`) -> ledger -> announcer (`announcer.py`) speaks result. Tool list for each agent comes only from its yaml, resolved through `TOOL_CATALOG` once at build time.

### 5. Error handling

- Unknown tool name in any yaml raises at agent build with the bad name and the catalog keys. Startup fails fast.
- Tool `execute` returns `ToolResult(observation, is_error)` as today (`base.py:11-14`). No change.
- Sandbox missing `bwrap` still fails closed (`sandbox.py:32-36`).
- Turn-boundary repair and retry behavior unchanged.

### 6. Testing

- Keep every existing tool test passing, moved beside new paths (read/write/edit containment + stale guard, bash sandbox, spawn/status/cancel, announcer isolation + `tool_choice none`).
- New pins: yaml-add-works (add `spawn` to a test worker registry via `build_registry(["spawn"], runtime)` and it executes; no engine edit), unknown-name-errors (bad yaml name raises naming the key), no-tool-names-in-engine (grep `spawn|finish|FOREGROUND` in `default.py` finds nothing outside comments), one-file-fix (touching `tools/read.py` needs no change in `default.py` or `runtime.py`).
- Full suite + `ruff` green before any merge claim.

## Grading criteria (this spec is graded on this, 10 points)

1. Truthfulness (3): every code claim pinned to a file:line above; no invented paths or behavior. Fail if any pin is wrong.
2. One-file rule (3): a tool bug touches exactly its `tools/<name>.py` + test; engine and coordinator untouched. Fail if any tool name remains in `default.py`.
3. Dynamic config (2): add-name-to-yaml works, remove works, unknown name errors loudly. Fail if any silent skip remains.
4. Human review (2): reviewer can find foreground vs worker vs announcer setup by filename; each file states its one job at the top. Fail if runtime setup stays in one 900-line file.

Pass is 8 or more. Max 5 grading rounds.
