# Explicit Finish Tool Specification for Autonomous Workers

## Problem Statement
Verbix's background worker previously inherited Mini-SWE-Agent's benchmark exit protocol: requiring the LLM to execute a shell command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT: <summary>` in bash.

In SWE-bench, the harness only evaluated `git diff` after agent termination; the text output of the submission command was never read. In Verbix, however, the background worker must report its findings back to the user via the voice announcer.

Forcing models (especially fast, lightweight models like `gemini-2.5-flash-lite`) to terminate via bash stdout parsing causes:
1. **Empty Output Glitches ("Task two done. No words made"):** Models often execute `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` without text, leading to `submission = ""` and synthetic fallback text.
2. **Cheated Execution:** Models learn that echoing the completion token immediately ends the task with success, causing them to skip real actions (e.g. skipping `sleep 15` and finishing in 1 second).
3. **Shell Escaping Issues:** Complex summaries with quotes, brackets, or newlines get corrupted when passed through `bash echo`.

## Proposed Solution: Approach B (Dedicated `finish` Tool)
Equip background workers with two first-class tools:
1. **`bash(command: str)`**: Pure shell execution. Returns exit code, stdout, and stderr without string interception.
2. **`finish(summary: str)`**: Dedicated task completion tool. Requires a non-empty `summary` describing the actions taken and verified results.

### Architectural Flow
```
[User Request] -> Foreground Companion spawns background task
                     │
                     ▼
[Worker Loop in DefaultAgent]
  Turn 1: Model calls bash("find src -type f | xargs wc -l")
  Turn 1 Observation: returns stdout "5936 src/..."
  Turn 2: Model verifies observation and calls finish(summary="Counted 5,936 lines across all Python files.")
  Turn 2 Finish Handling:
    - Yields ToolCallFinished(tool_name="finish", observation="Task marked complete.")
    - Yields TurnComplete(final_text=summary)
    - Concludes background loop cleanly
                     │
                     ▼
[WorkerTask] captures final_text as self._result
                     │
                     ▼
[Voice Announcer] speaks natural summary to user (in caveman or configured persona)
```

## Key Invariants and Guarantees
1. **No Premature Exit:** Conversational text or intermediate thinking emitted by the model does not terminate the loop. The task only terminates when `finish()` is explicitly invoked (or limits reached).
2. **Information-Dense Inter-Agent Communication:**
   - **High-Bandwidth Handoff:** In modern multi-agent architectures (like Codex, OpenCode, Claude Code), subagent delegation and debriefing must be information-dense and technically complete.
   - **Objective Prompting (`spawn`):** The foreground companion delegates with a complete specification: intent, context/scope, and expected verification criteria (e.g. "Measure execution time, count passed/failed tests, verify no regressions").
   - **Summary Debriefing (`finish`):** The worker's `summary` argument must provide a dense factual debrief: exact numbers, durations, exit codes, modified paths, and concrete evidence. Vague one-liners without data are prohibited.
   - **Announcer Synthesis:** The Announcer is the lossy compression and persona translation layer. It consumes this rich debrief and distills it into conversational dialogue matching the active persona without dropping critical metrics.
3. **Schema-Enforced Summary:** The `finish` tool schema enforces that `summary` is a required non-empty string.
4. **Zero Shell Token Interception:** `LocalEnvironment` performs pure command execution; all stdout parsing for benchmark tokens is removed.
5. **Fallback:** If `finish` is called with empty whitespace or if a task completes without a result, `WorkerTask` falls back to the stdout of the last tool executed rather than defaulting to `"Task finished without output."`.

## Affected Files and Exact Scope

### 1. `src/verbix/runtime/tools.py`
- Add `FinishTool` definition with OpenAI function schema:
  - `name`: `"finish"`
  - `description`: `"Call this tool when the task objective has been fully executed and verified. Provide an information-dense factual summary of actions taken, concrete numbers/metrics, and verified results."`
  - `parameters`: `{"type": "object", "properties": {"summary": {"type": "string", "description": "Information-dense factual debrief: specific commands run, exact metrics/counts, exit codes, and verification evidence."}}, "required": ["summary"]}`

### 2. `src/verbix/minisweagent/config/worker.yaml`
- Configure tools:
  ```yaml
  tools:
    - bash
    - finish
  ```
- Update execution instructions:
  - Discover, Execute, and Verify using `bash`.
  - When verified, call `finish(summary="...")` with the final report.
  - Never call `finish` before executing the necessary actions.
- Remove all references to `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and update `format_error_template`.

### 3. `src/verbix/minisweagent/environments/local.py`
- Remove `_check_finished`, `_check_command_finished`, and benchmark token interception. `LocalEnvironment.execute` strictly runs bash and returns standard output/returncode/exception dictionaries.

### 4. `src/verbix/minisweagent/agents/default.py`
- In `step_stream()`: When a tool call is named `"finish"`, extract `summary = args.get("summary", "")`, yield `ToolCallFinished`, emit `TurnComplete(final_text=summary)`, and terminate the turn loop.
- In synchronous `step()`: Remove `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` check.
- Retain `Submitted` exception handling for backwards compatibility with any external callers.

### 5. `src/verbix/runtime/runtime.py`
- In `Runtime.spawn()`: Register `FinishTool` into the worker's `ToolRegistry`.
- Synchronize `DEFAULT_WORKER_PROMPT`.

### 6. `src/verbix/runtime/worker.py`
- In `WorkerTask._run()`: Capture `TurnComplete.final_text`. If empty, inspect session history and use the output of the last bash command.

### 7. Tests
- `tests/test_prompt_invariants.py`: Verify `worker.yaml` has `tools: [bash, finish]` and no benchmark submission tokens.
- `tests/test_miniswe_environment.py`: Verify clean shell execution without string interception.
- `tests/test_miniswe_agent_turn.py`: Verify worker turn flow with `finish` tool execution and clean result propagation.
- `tests/test_runtime_e2e.py`: Verify worker task executes commands and sets result from `finish(summary=...)`.
