from __future__ import annotations

from verbix.core.models import (
    CanonicalMessage,
    CanonicalToolCall,
)


class ConversationHistory:
    """In-memory append-only sequence of canonical messages with turn-boundary validation

    and barge-in truncation.
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
