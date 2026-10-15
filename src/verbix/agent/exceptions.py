class FormatError(Exception):
    """Raised when the model's output is not in the expected tool-call format."""

    def __init__(self, *messages: dict) -> None:
        self.messages = list(messages)
        super().__init__(str(messages[-1].get("content", "")) if messages else "")
