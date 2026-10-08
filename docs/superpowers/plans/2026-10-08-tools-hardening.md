# Tools Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Worker tools match battle-tested shapes: pictures fit the budget, search needs no shell, every tool states its own limits.

**Architecture:** Single-file tools in `src/verbix/tools/` behind the existing `Tool` protocol, registered in `build_worker_registry`; `ripgrep` binary via `rg` subprocess for search; Pillow for image resize; caps enforced where the payload is built.

**Tech Stack:** Python ≥3.13, stdlib + `Pillow` + `ripgrep` (both via `uv add`), `num2words` untouched.

**Spec:** In-chat approved scope (Slice A: 1280px/JPEG80/~2MB/loud errors; Slice B: grep+glob on `rg`, no fallback; Slice C: bash timeout/caps, edit guards, read caps/notices, prompt touch-ups). Key decisions: `uv add` for dependencies; no Python search fallback; no web/question/todo/skill tools; letter-slashes stay raw.

## Global Constraints

- Python >= 3.13.
- Dependencies via `uv add` only (`pillow`, `ripgrep`).
- Every behavior ships with exact input→output tests; TDD per task.
- `ruff check` clean on touched files; full suite green before each task commits.
- Displayed text stays raw; worker prompt keeps bash-as-escape-hatch language.
- Never broaden: no screenshot capture, no web tools, no phonemizer changes, no prompt sections beyond touch-ups.

## Review Focus

1. A 20MB photo must error loudly before decode, never enter history (Task 1).
2. A machine without the `rg` binary must get a clear error naming the missing piece, not a traceback (Task 2).
3. Binary-file grep hits must never dump raw bytes into observations (Task 2).
4. A command exceeding its timeout must report `timed out after Ns` with partial output, never hang the turn (Task 4).
5. An edit whose target occurs twice must refuse with counts, never pick one silently (Task 4).

---

### Task 1: Image downscale before send

**Files:**
- Modify: `src/verbix/tools/filesystem.py` (`_read_image`, `MAX_IMAGE_BYTES`)
- Modify: `pyproject.toml`, `uv.lock` (via `uv add pillow`)
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: existing `_read_image(target, path_str) -> ToolResult`, `IMAGE_MIMES`, tag format (unchanged).
- Produces: bounded image payloads (≤1280px longest side, JPEG, ≤2MB) that Task 6 verifies live.

- [ ] **Step 1: Write the failing tests**

```python
def test_image_errors_are_exact(tmp_path):
    # empty file errors exactly `error: empty image`; Pillow-missing errors exactly
    # `error: cannot process images: Pillow is not installed` (monkeypatched import).

def test_large_image_is_downscaled_to_budget(tmp_path):
    # 3000x2000 generated PNG must come back `data:image/jpeg;base64,` with decoded longest side == 1280
    # and total payload ≤ 2MB; empty file errors `error: empty image`; 21MB file errors before decode;
    # still-over-2MB errors naming actual vs limit
```

(Plus: 20MB+ file errors before decode; small PNG passes through byte-identical with `image/png` mime.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_tools.py -q -p no:cacheprovider -k "image_errors or downscaled or passthrough"`
Expected: FAIL (full-size passthrough; no budget).

- [ ] **Step 3: Run `uv add pillow`, implement bounded `_read_image` in `src/verbix/tools/filesystem.py`**

Reject empty files (`error: empty image`) and files over 20MB before decode; missing Pillow errors exactly `error: cannot process images: Pillow is not installed` (lazy import inside the image path, never at module top). Only when width > 1280 or height > 1280 or bytes > 2MB: downscale longest side to 1280px (bilinear) and re-encode JPEG quality 80; error loudly if still over 2MB or if Pillow is unavailable. Small images within all bounds pass through byte-identical (existing test pins the `image/png` prefix).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_tools.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/filesystem.py tests/test_tools.py pyproject.toml uv.lock
git commit -m "feat: bound image payloads before model send"
```

### Task 2: Grep tool on the ripgrep binary
**Files:**
- Create: `src/verbix/tools/search.py` (`GrepTool`)
- Modify: `src/verbix/runtime/tools.py` (`build_worker_registry`), `pyproject.toml`, `uv.lock` (via `uv add ripgrep`)
- Test: `tests/test_search_tools.py`

**Interfaces:**
- Consumes: `resolve_contained_path`, `ToolContext`/`ToolResult` from `verbix.tools.base`; `rg` binary from the `ripgrep` package (located via `shutil.which("rg")`).
- Produces: `GrepTool` (name `grep`) registered on the worker; Task 5 advertises it.

- [ ] **Step 1: Write the failing tests**

```python
def test_grep_finds_grouped_hits(tmp_path):
    # two files with matches → exactly `a.txt:1: hit one\na.txt:3: hit two\nb.txt:1: hit one`
    # plus: over-limit run ends with `... truncated; refine pattern`; binary file never leaks raw bytes;
    # missing binary errors `error: ripgrep binary 'rg' not found (uv add ripgrep)`

def test_grep_missing_binary_errors_clearly(monkeypatch):
    # shutil.which → None → "error: ... ripgrep ..." naming uv add ripgrep
```

(Plus: hit cap with refine notice; binary-file hits never leak raw bytes.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search_tools.py -q -p no:cacheprovider`
Expected: FAIL with collection error (module missing).

- [ ] **Step 3: Run `uv add ripgrep`, implement `GrepTool` in `src/verbix/tools/search.py`**

Schema: `pattern` (required), `path` (default `.`), `include` (glob, optional), `limit` (default 50). Resolves `rg` via `shutil.which`, clear error when absent. Runs `rg --line-number --no-heading`, passing `--glob={include}` when set and `--max-count={limit}`, workspace-contained, grouped `file:line: text` output, `... truncated; refine pattern` notice past the cap, `--text` off so binaries never leak bytes.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search_tools.py tests/test_worker_tools.py -q -p no:cacheprovider`
Expected: PASS (worker contract test updated here to exactly {read, write, edit, grep, finish}; add `grep` to the `tools:` list in `src/verbix/minisweagent/config/worker.yaml` in this task).

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/search.py src/verbix/runtime/tools.py src/verbix/minisweagent/config/worker.yaml tests/test_search_tools.py tests/test_worker_tools.py pyproject.toml uv.lock
git commit -m "feat: grep tool backed by ripgrep"
```

### Task 3: Glob tool on the same binary

**Files:**
- Modify: `src/verbix/tools/search.py` (`GlobTool`), `src/verbix/runtime/tools.py`
- Test: `tests/test_search_tools.py`

**Interfaces:**
- Consumes: `rg` resolution helper from Task 2.
- Produces: `GlobTool` (name `glob`) registered on the worker.

- [ ] **Step 1: Write the failing tests**

```python
def test_glob_lists_matching_paths(tmp_path):
    # *.py pattern → one relative path per line, sorted
```

(Plus: no matches → "No files found"; missing binary → same clear error.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_search_tools.py -q -p no:cacheprovider -k glob`
Expected: FAIL (no such tool).

- [ ] **Step 3: Implement `GlobTool` in `src/verbix/tools/search.py`**

Schema: `pattern` (required), `path` (default `.`), `limit` (default 100). Runs `rg --files --glob`, workspace-contained, one relative path per line.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_search_tools.py tests/test_worker_tools.py -q -p no:cacheprovider`
Expected: PASS (worker contract test updated here to exactly {read, write, edit, grep, glob, finish}; add `glob` to the `tools:` list in `src/verbix/minisweagent/config/worker.yaml` in this task).

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/search.py src/verbix/runtime/tools.py src/verbix/minisweagent/config/worker.yaml tests/test_search_tools.py tests/test_worker_tools.py
git commit -m "feat: glob tool backed by ripgrep"
```

### Task 4: Bash timeout, output cap, edit guards, read caps

**Files:**
- Modify: `src/verbix/minisweagent/agents/default.py` (`_execute_tool` bash path), `src/verbix/minisweagent/models/utils/actions_toolcall.py` (`BASH_TOOL` schema), `src/verbix/tools/filesystem.py` (edit guards, read caps)
- Test: `tests/test_agent.py`, `tests/test_tools.py`

**Interfaces:**
- Consumes: `LocalEnvironment.aexecute(action, timeout=...)` (already accepts timeout).
- Produces: bounded shell/file behavior Task 6 verifies live.

- [ ] **Step 1: Write the failing tests**

```python
def test_bash_timeout_and_cap(...):
    # action {"command": "sleep 5", "timeout_seconds": 2} reaches env as timeout=2 (fake env records it);
    # env-shaped timeout dict (`extra.exception_type == "TimeoutExpired"`, partial "out") yields exactly
    # `timed out after 2s` with the partial output; default (absent) is 120, `timeout_seconds: 9999` clamps to 600;
    # 2MB output ends with exactly `[output truncated: showing last 1048576 chars]`

def test_edit_refuses_empty_identical_multiple(...):
    # "" old_str → `error: empty old_str`; same new_str → `error: new_str identical to old_str`;
    # 2 matches → `error: old_str occurs 2 times in 'f'`; 50KB+ file ends with `[truncated at 51200 bytes; re-read with start_line past {n}]`
```

(Plus: text reads over 50KB truncate with resume offset notice.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_agent.py tests/test_tools.py -q -p no:cacheprovider -k "timeout or cap or guard or resume"`
Expected: FAIL.

- [ ] **Step 3: Implement the timeout chain end to end**

`BASH_TOOL` schema gains `timeout_seconds` (number, default 120). `parse_toolcall_actions` passes `timeout_seconds` through on bash actions (absent → 120 downstream). `DefaultAgent._execute_tool` clamps to 600 and forwards as `timeout=` to `env.aexecute` and `env.execute` (both accept it today). `LocalEnvironment` catches expiry into a result dict (never raises), so `_execute_tool` maps the real signal: when result `extra.exception_type == "TimeoutExpired"`, the observation is exactly `timed out after {timeout:g}s` plus partial stdout when present. No `local.py` change. Bash output past 1MB truncates with exactly `[output truncated: showing last 1048576 chars]`. `FileEditTool` refuses empty `old_str` (`error: empty old_str`), identical `new_str` (`error: new_str identical to old_str`), and multi-match (`error: old_str occurs {n} times ... Provide unique context.` — already present, keep). `FileReadTool` caps text at 50KB: observation ends with exactly `[truncated at 51200 bytes; re-read with start_line past {last_shown}]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_agent.py tests/test_tools.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/minisweagent/agents/default.py src/verbix/minisweagent/models/utils/actions_toolcall.py src/verbix/tools/filesystem.py tests/test_agent.py tests/test_tools.py
git commit -m "feat: bounded bash, guarded edit, capped reads"
```

### Task 5: Prompt touch-ups and tool descriptions

**Files:**
- Modify: `src/verbix/tools/filesystem.py` (descriptions), `src/verbix/tools/search.py` (descriptions), `src/verbix/minisweagent/config/worker.yaml` (Discover line)
- Test: `tests/test_worker_tools.py` (registry names grow to read/write/edit/grep/glob/finish)

**Interfaces:**
- Consumes: tool names from Tasks 2–3.
- Produces: advertised capabilities the live check in Task 6 relies on.

- [ ] **Step 1: Write the prompt test, run to verify it fails**

```python
def test_worker_prompt_names_search_tools():
    # worker.yaml Discover/rules text names read, grep, and glob; every tool
    # description in the worker registry contains the word "bash" (the boundary).
```

Run: `.venv/bin/python -m pytest tests/test_worker_tools.py -q -p no:cacheprovider -k prompt`
Expected: FAIL (Discover line and descriptions predate grep/glob).

- [ ] **Step 2: Rewrite each tool description as one scope sentence plus its bash boundary** (e.g. read: files/dirs/images via offset; use bash only for recursive search before glob existed — now: glob covers search); update `worker.yaml` Discover line to name read/grep/glob.

- [ ] **Step 3: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_worker_tools.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/verbix/tools/filesystem.py src/verbix/tools/search.py src/verbix/minisweagent/config/worker.yaml tests/test_worker_tools.py
git commit -m "docs: tool boundaries and worker discovery prompt"
```

### Task 6: Full verification and live photo check

**Files:** none (verification only).

- [ ] **Step 1: Run the full suite and lint**

Run: `.venv/bin/python -m pytest -q -p no:cacheprovider` (expected: all pass) and `.venv/bin/ruff check` on touched files only (expected: clean; pre-existing baseline findings untouched).

- [ ] **Step 2: Live photo check**

In a voice run with `--workspace` pointed at a folder containing one photo, `read` that photo and ask what it shows (or: `uv run verbix --show-worker-turns`, spawn `read the image in the workspace`). Expected: worker `read` on the file is followed by a true visual description with no fallback tools and no surrender message.

- [ ] **Step 3: Record the outcome**

No commit (no files change). Report the suite count and the live result.

## Self-Review

1. **Spec coverage:** image bounds→T1, grep→T2, glob→T3, bash/edit/read bounds→T4, prompts→T5, proof→T6. Screenshot capture, web tools, phonemizers: correctly absent (non-goals).
2. **Step scan:** each test step carries exact assertions; code steps name exact schemas/functions/values; verification steps name commands + expected output.
3. **Type consistency:** `GrepTool`/`GlobTool` names match registry/test references; `build_worker_registry` grows read/write/edit/grep/glob/finish; `ToolResult`/`ToolContext` reused, not redefined.
4. **Review Focus:** five lines each map to owning tasks (T1, T2, T4, T4, T4).
5. **Proportion:** decisions + tests + signatures; bodies left to implementer.
