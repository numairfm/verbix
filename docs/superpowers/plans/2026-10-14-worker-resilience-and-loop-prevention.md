# Worker Resilience, Search Validation & Loop Prevention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent worker runaway loops and improve error recovery by implementing search contract validation (failing loudly on bot challenges), preserving model prose during format errors, adding a 3-strike tool call circuit breaker, and harmonizing worker prompt invariants.

**Architecture:**
- `src/verbix/tools/web_search.py`: validates DDG HTML response contract. If 0 results are found and search markup is missing (e.g. CAPTCHA/bot block), returns `is_error=True`.
- `src/verbix/agent/models/utils/actions_toolcall.py`: on format error (no tool calls), bundles both the assistant message and the user reminder into `FormatError` so model thoughts are preserved in context.
- `src/verbix/agent/config/worker.yaml`: updates Invariant 4 to permit `finish` on blockers, clarifies format error guidance, and sets `step_limit: 30`.
- `src/verbix/agent/worker.py`: adds a 3-strike consecutive duplicate tool call circuit breaker that intercepts identical repeated calls and forces a change or completion.

**Tech Stack:** Python 3.13+, standard library (`json`, `html.parser`, `urllib.error`).

**Spec:** `docs/superpowers/specs/2026-10-14-worker-resilience-and-loop-prevention-design.md`

## Global Constraints
- Zero new third-party dependencies added to `pyproject.toml`.
- Modular per `AGENTS.md`: changes are scoped to their respective files with dedicated unit tests.
- 100% offline unit tests: external network calls are strictly mocked.
- Full test suite and `ruff check` pass cleanly.

## Review Focus
- Contract validation: does a valid query returning genuinely 0 results on a real search page still return `No results found for query` with `is_error=False`, while a CAPTCHA/bot block returns `is_error=True`?
- Context preservation: does `worker.messages` retain the assistant's attempted prose when a format error occurs?
- Circuit breaker: does changing even one argument reset the repeat counter so legitimate sequential calls (like reading different files) are never blocked?
- Step ceiling: is `step_limit: 30` properly enforced when workers take many steps?

---

### Task 1: Search Contract Validation in `src/verbix/tools/web_search.py`

**Files:**
- Modify: `src/verbix/tools/web_search.py:80-165`
- Test: `tests/test_tools_web_search.py`

**Interfaces:**
- Consumes: `DuckDuckGoLiteParser`, `SearchResult`, `ToolResult` from `verbix.tools.web_search`.
- Produces: `execute()` returning `ToolResult(..., is_error=True)` when DuckDuckGo returns an unparseable/blocked response.

- [ ] **Step 1: Write failing tests in `tests/test_tools_web_search.py`**

Add tests:
- `test_ddg_bot_challenge_returns_error`: mocks DDG returning HTML containing an anomaly modal or non-search layout; asserts `result.is_error == True` and message mentions provider blocked/unrecognized response.
- `test_ddg_genuine_zero_results_returns_no_results`: mocks DDG returning a valid search page with explicit zero-results indicator (e.g. `<div class="no-results">No results.</div>`); asserts `result.is_error == False` and observation is `No results found for query: '...'`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools_web_search.py -k "ddg_bot_challenge" -v`
Expected: FAIL.

- [ ] **Step 3: Implement contract validation in `src/verbix/tools/web_search.py`**

In `DuckDuckGoLiteParser`:
- Track `self.has_search_form: bool = False` and `self.has_zero_results_marker: bool = False`.
- In `handle_starttag` / `handle_data`: detect if the search form exists (e.g. form with action `/lite/` or input `q`) or text contains `"No results."`.
In `_search_duckduckgo`:
- If `len(parser.results) == 0 and not parser.has_zero_results_marker and not parser.has_search_form`:
  Raise `urllib.error.URLError("DuckDuckGo returned an unrecognized or bot-challenged response. For reliable search, configure BRAVE_API_KEY, TAVILY_API_KEY, or EXA_API_KEY.")`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_tools_web_search.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/tools/web_search.py tests/test_tools_web_search.py
git commit -m "fix(web_search): fail honestly on unrecognized or blocked DDG responses"
```

---

### Task 2: Preserve Model Context on FormatError

**Files:**
- Modify: `src/verbix/agent/models/utils/actions_toolcall.py:45-65`
- Modify: `src/verbix/agent/worker.py:170-195`
- Test: `tests/test_worker_agent.py`

**Interfaces:**
- Consumes: `FormatError` from `verbix.agent.exceptions`.
- Produces: `WorkerAgent.messages` containing both the `assistant` message and the user format error reminder when the model produces pure text.

- [ ] **Step 1: Write failing test in `tests/test_worker_agent.py`**

Add test:
- `test_worker_preserves_assistant_text_on_format_error`: configure mock model to return text `"I could not find the file."` without tool calls on first step, then `finish` on second step. Assert `worker.messages` contains `{"role": "assistant", "content": "I could not find the file."}` followed by `{"role": "user", ...}`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker_agent.py -k "test_worker_preserves_assistant_text" -v`
Expected: FAIL.

- [ ] **Step 3: Implement context preservation in `actions_toolcall.py` and `worker.py`**

In `actions_toolcall.py`:
- When `not tool_calls`:
  Extract raw assistant text from response.
  Raise `FormatError` with:
  ```python
  [
      {"role": "assistant", "content": raw_text},
      {
          "role": "user",
          "content": Template(format_error_template).render(
              error="No tool calls found in the response. Every response MUST include at least one tool call.",
              actions=[],
              has_tool_calls=False,
              **template_kwargs,
          ),
          "extra": {"interrupt_type": "FormatError"},
      },
  ]
  ```
In `src/verbix/agent/worker.py`:
- In `except FormatError as e`:
  `if e.messages: self.messages.extend(e.messages)` properly preserves both the assistant message and user error.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_worker_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/verbix/agent/models/utils/actions_toolcall.py src/verbix/agent/worker.py tests/test_worker_agent.py
git commit -m "fix(worker): preserve assistant text in history on format error"
```

---

### Task 3: Tool Loop Circuit Breaker & Prompt Invariant Harmonization

**Files:**
- Modify: `src/verbix/agent/worker.py:85-115,225-275`
- Modify: `src/verbix/agent/config/worker.yaml:1-50,110-120`
- Test: `tests/test_worker_agent.py`

**Interfaces:**
- Consumes: `WorkerConfig`, `WorkerAgent` in `src/verbix/agent/worker.py`.
- Produces: 3-strike circuit breaker on identical consecutive tool calls and default `step_limit: 30`.

- [ ] **Step 1: Write failing tests in `tests/test_worker_agent.py`**

Add tests:
- `test_worker_circuit_breaker_intercepts_repeated_calls`: mock model to call `web_search({"query": "news"})` repeatedly. Verify that on the 3rd consecutive call with identical args, a circuit breaker error is returned to the model (`"Error: Tool 'web_search' was called 3 times in a row with identical arguments"`).
- `test_worker_circuit_breaker_resets_on_different_arguments`: verify calling `read({"path": "a.txt"})` then `read({"path": "b.txt"})` does not trigger the circuit breaker.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_worker_agent.py -k "circuit_breaker" -v`
Expected: FAIL.

- [ ] **Step 3: Implement circuit breaker and update `worker.yaml`**

In `src/verbix/agent/worker.py`:
- In `__init__`: initialize `self._last_call: tuple[str, str] = ("", "")` and `self._consecutive_repeats: int = 0`.
- In `step()` before `_execute_tool`:
  ```python
  sig = (name, json.dumps(args, sort_keys=True))
  if sig == self._last_call and name != "finish":
      self._consecutive_repeats += 1
      if self._consecutive_repeats >= 3:
          result = ToolResult(
              f"Error: Tool '{name}' was called 3 times consecutively with identical arguments without progress. "
              "You must try a different approach or call 'finish'.",
              is_error=True,
          )
          # append tool result to messages and return
  else:
      self._last_call = sig
      self._consecutive_repeats = 1
  ```
In `src/verbix/agent/config/worker.yaml`:
- Line 5: update persistence invariant to `persist until the objective is resolved or definitively determined to be unachievable.`
- Lines 15-20: harmonize Invariant 4 to permit `finish` on blockers while keeping anti-slacking constraint.
- Line 44: set `step_limit: 30`.
- Line 116: update format error guidance to include calling `finish` on blockers.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_worker_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Run full test suite and linter**

Run: `pytest`
Run: `ruff check`
Expected: all tests pass, ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/verbix/agent/worker.py src/verbix/agent/config/worker.yaml tests/test_worker_agent.py
git commit -m "feat(worker): add 3-strike circuit breaker and harmonize prompt invariants"
```

---

### Task 4: Final Whole-Branch Review & Verification

**Files:**
- All modified files across Tasks 1-3.

- [ ] **Step 1: Run full test suite and lint**
Run: `pytest -q`
Run: `ruff check`

- [ ] **Step 2: Subagent Review Pass**
Dispatch a reviewer subagent to inspect the entire diff, verifying test coverage, circuit breaker behavior, and prompt cleanliness.

- [ ] **Step 3: Report git diff and summary with LOC and file sizes**
Provide the final diff, file sizes, and LOC summary.
