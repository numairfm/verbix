from __future__ import annotations

from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
    TaskState,
    WorkerResult,
)


class ConversationHistory:
    """In-memory append-only sequence of canonical messages with turn-boundary validation,

    runtime-authored completion injection, and barge-in truncation.
    """

    def __init__(self) -> None:
        self._messages: list[CanonicalMessage] = []

    @property
    def messages(self) -> list[CanonicalMessage]:
        return list(self._messages)

    def append(self, message: CanonicalMessage) -> None:
        self._messages.append(message)

    def append_user(self, content: str) -> None:
        self.append(CanonicalMessage(role="user", content=content))

    def append_assistant(
        self,
        content: str,
        thought: str | None = None,
        tool_calls: list[CanonicalToolCall] | None = None,
    ) -> None:
        self.append(
            CanonicalMessage(
                role="assistant",
                content=content,
                thought=thought,
                tool_calls=tool_calls,
            )
        )

    def inject_task_completion(
        self, task_id: str, title: str, result: WorkerResult
    ) -> None:
        """Injects an immutable, runtime-authored task completion record into history.

        Uses role='user' with runtime notice attribution so it works across providers
        and allows a wake turn to respond naturally.
        """
        outcome = "Succeeded" if result.success else "Failed"
        summary = result.speech_summary.rstrip(".")
        details = result.details.strip()
        if len(details) > 600:
            details_str = (
                f" Details: {details[:600]}... (call task_status for full details)"
            )
        elif details:
            details_str = f" Details: {details}"
        else:
            details_str = ""
        formatted = (
            "[Verbix runtime notice — not said by the user]\n"
            f"[Task completed: '{title}' ({task_id}) -> {outcome}: {summary}.{details_str}]"
        )
        self.append(CanonicalMessage(role="user", content=formatted))

    def close_aborted_turn(self, delivered: str) -> None:
        """Closes an aborted turn cleanly so history remains structurally valid.

        1. If an assistant message with tool_calls has unanswered IDs, append
           synthetic tool results so the tool calls are not left dangling.
        2. Else, if the last message is an assistant message without tool_calls:
           if delivered is non-empty, update its content to delivered;
           if delivered is empty, pop that assistant message.
        3. Else, if delivered is non-empty, append an assistant message with delivered text.
        """
        if not self._messages:
            if delivered:
                self.append_assistant(content=delivered)
            return

        last = self._messages[-1]
        if last.role == "assistant" and last.tool_calls:
            for tc in last.tool_calls:
                self.append(
                    CanonicalMessage(
                        role="tool",
                        content="error: not executed (turn aborted)",
                        tool_call_id=tc.id,
                    )
                )
            return

        # Check if an earlier assistant message has unanswered tool calls
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.role == "assistant" and msg.tool_calls:
                expected_ids = [tc.id for tc in msg.tool_calls]
                provided_ids = {
                    m.tool_call_id
                    for m in self._messages[i + 1 :]
                    if m.role == "tool" and m.tool_call_id is not None
                }
                missing_ids = [cid for cid in expected_ids if cid not in provided_ids]
                for cid in missing_ids:
                    self.append(
                        CanonicalMessage(
                            role="tool",
                            content="error: not executed (turn aborted)",
                            tool_call_id=cid,
                        )
                    )
                return
            if msg.role == "user":
                break

        if last.role == "assistant" and not last.tool_calls:
            if delivered:
                last.content = delivered
            else:
                self._messages.pop()
        elif delivered:
            self.append_assistant(content=delivered)

    def render_turn_prompt(
        self, running_tasks: list[TaskState] | None = None
    ) -> list[CanonicalMessage]:
        """Renders messages for the current turn prompt.

        If running tasks exist, prepends an ephemeral running-task line to the active user message.
        Invariants:
        1. Never includes completed/finished tasks.
        2. Never mutates or persists the running-task annotation to underlying history.
        """
        msgs = [m.model_copy() for m in self._messages]
        if not running_tasks or not msgs:
            return msgs

        active = [t for t in running_tasks if t.is_running]
        if not active:
            return msgs

        lines = [
            f"[Running: {t.id} '{t.title}' ({t.step_count} tool calls)]" for t in active
        ]
        prefix = "\n".join(lines) + "\n\n"

        # Attach prefix to the last user message
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                msgs[i].content = prefix + msgs[i].content
                break

        return msgs
