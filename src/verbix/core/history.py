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

    def append_runtime_turn(self, user_input: str, runtime_reply: str) -> None:
        """Appends both user input and runtime reply together, preserving dialogue parity

        when the runtime answers a turn deterministically without calling the model.
        """
        self.append_user(user_input)
        self.append_assistant(content=runtime_reply)

    def inject_task_completion(
        self, task_id: str, title: str, result: WorkerResult
    ) -> None:
        """Injects an immutable, runtime-authored task completion record into history.

        Zero LLM calls are made to format this message. Uses role='system' so it is
        never confused with human dialogue or assistant speech.
        """
        outcome = "Succeeded" if result.success else "Failed"
        formatted = f"[Task completed: '{title}' ({task_id}) -> {outcome}: {result.speech_summary}]"
        self.append(CanonicalMessage(role="system", content=formatted))

    def prune_foreground_barge_in(self, played_text: str) -> None:
        """Prunes strictly the last assistant message to match the text physically played

        before the user barged in. Background task records are never pruned.
        """
        if not self._messages:
            raise ValueError("History is empty, cannot prune barge-in")

        last = self._messages[-1]
        if last.role != "assistant":
            raise ValueError(f"Cannot prune non-assistant message (role={last.role})")

        last.content = played_text

    def is_turn_boundary(self) -> bool:
        """Reports whether history is currently at a settled turn boundary.

        Returns False if a user message is waiting for a reply, or if an assistant
        tool call is still waiting for its matching tool result.
        """
        if not self._messages:
            return True

        last = self._messages[-1]
        if last.role == "user":
            return False

        if last.role == "assistant":
            # If the assistant made tool calls, it is waiting for tool observations
            return not bool(last.tool_calls)

        if last.role == "tool":
            # Find the preceding assistant message with tool calls
            for msg in reversed(self._messages[:-1]):
                if msg.role == "assistant" and msg.tool_calls:
                    expected_ids = {tc.id for tc in msg.tool_calls}
                    # Gather provided tool results following that assistant message
                    provided_ids = {
                        m.tool_call_id
                        for m in self._messages
                        if m.role == "tool" and m.tool_call_id is not None
                    }
                    return expected_ids.issubset(provided_ids)
                if msg.role == "user":
                    break
            return True

        return True

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

        lines = [f"[Running: {t.id} '{t.title}' (step {t.step_count})]" for t in active]
        prefix = "\n".join(lines) + "\n\n"

        # Attach prefix to the last user message
        for i in range(len(msgs) - 1, -1, -1):
            if msgs[i].role == "user":
                msgs[i].content = prefix + msgs[i].content
                break

        return msgs
