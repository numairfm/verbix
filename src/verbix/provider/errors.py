from __future__ import annotations


class ProviderError(Exception):
    """Base error for all model provider failures."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class AuthenticationError(ProviderError):
    """HTTP 401: Invalid API key or unauthorized."""


class RateLimitError(ProviderError):
    """HTTP 429: Rate limit or quota exceeded."""


class InvalidRequestError(ProviderError):
    """HTTP 400: Malformed request payload, invalid parameters, or context limit."""


class ServerError(ProviderError):
    """HTTP 5xx: Provider server error or outage."""
