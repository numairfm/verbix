# Worker Resilience, Search Validation & Loop Prevention Design Spec

## 1. Overview
Hardens Verbix worker autonomy and error recovery against infinite tool loops, silent search engine blocks (CAPTCHAs), discarded model thoughts on format errors, and self-contradicting prompt invariants.

## 2. Requirements & Constraints
- **Zero New Dependencies**: Uses Python standard library only.
- **Contract-Based Search Validation**: `web_search.py` must validate the structural integrity of search responses. If zero results are parsed and the page lacks search result markup, it must fail honestly (`is_error=True`) instead of claiming "No results found".
- **Preserve Model Context on Format Error**: When a model produces text without a tool call, its `assistant` message must NOT be discarded. It must be kept in history so the model retains its own reasoning on the subsequent turn.
- **Universal, Non-Overfitted Prompt Invariants**:
  - Reconcile Invariant 4 to permit `finish` when an objective is blocked or resources do not exist, while keeping the anti-slacking rule ("do not call finish prematurely if tools can still make progress").
  - Update format error template to offer `finish` as the exit door for blockers.
  - Set a default `step_limit: 30` in `worker.yaml`.
- **Tool Loop Circuit Breaker**: If a worker executes the exact same tool with identical arguments 3 times consecutively, intercept and return an error forcing a change in approach or completion.

## 3. Interfaces & Changes

### A. `src/verbix/tools/web_search.py`
In `_search_duckduckgo`:
- Inspect `DuckDuckGoLiteParser`: track whether search result markup was encountered (e.g. `result-link`, `result-snippet`, or search result tables).
- If `len(parser.results) == 0` and no search results container/marker was found in the HTML:
  Raise `urllib.error.URLError("DuckDuckGo returned an unrecognized or bot-challenged response. For reliable search, configure BRAVE_API_KEY, TAVILY_API_KEY, or EXA_API_KEY.")`.

### B. `src/verbix/agent/models/utils/actions_toolcall.py` & `src/verbix/agent/worker.py`
When `parse_tool_calls` encounters a response with no tool calls:
- Raise `FormatError` containing BOTH:
  1. An `assistant` message with the model's text (`{"role": "assistant", "content": raw_text}`).
  2. A `user` error message reminding the model: `"You responded with text instead of a tool call. If your task is complete or you are blocked, call 'finish' with your conclusion. Otherwise, call a tool to proceed."`
- `WorkerAgent.step()` preserves both messages in `self.messages`.

### C. `src/verbix/agent/config/worker.yaml`
- Line 5: `persist until the objective is resolved or definitively determined to be unachievable.`
- Lines 15-20:
  ```yaml
  4. Complete: When the task is complete, or when an unrecoverable failure or missing resource prevents completion, call the 'finish' tool:
     - Lead with the conclusive finding or direct answer. If an operation failed, a resource was missing, or information was unavailable, state that fact clearly as the primary conclusion.
     - Follow with supporting detail (actions attempted, errors encountered, verification evidence).
     - Do not call 'finish' prematurely if available tools can still make progress.
  ```
- Format error template: `"When your task is complete, or if you are blocked and cannot proceed, call the 'finish' tool with your conclusion:"`
- Change `step_limit: 0` to `step_limit: 30`.

### D. `src/verbix/agent/worker.py`
In `WorkerAgent`:
- Add tracking for `_last_tool_call: tuple[str, str]` and `_consecutive_tool_repeats: int`.
- In `step()` before `_execute_tool`:
  Compare `(name, json.dumps(args, sort_keys=True))`.
  If identical 3 times in a row:
  Return `ToolResult(f"Error: Tool '{name}' was called 3 times consecutively with identical arguments without progress. You must try a different approach or call 'finish'.", is_error=True)`.
  If different: reset `_consecutive_tool_repeats = 0`.
