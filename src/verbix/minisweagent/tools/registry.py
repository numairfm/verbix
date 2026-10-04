from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any


class ToolRegistry:
    """Registry for agent tools, schemas, and execution handlers."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}

    def register(
        self, name: str, definition: dict[str, Any], handler: Callable[..., Any]
    ) -> None:
        """Register a tool definition and its handler."""
        self._tools[name] = definition
        self._handlers[name] = handler

    def get_definitions(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        """Return tool definitions for the requested names, or all if names is None."""
        if names is None:
            return list(self._tools.values())
        return [self._tools[name] for name in names if name in self._tools]

    def _call_handler(
        self, handler: Callable[..., Any], args: dict[str, Any], **kwargs: Any
    ) -> Any:
        if kwargs:
            try:
                return handler(args, **kwargs)
            except TypeError:
                return handler(args)
        return handler(args)

    def execute(self, name: str, args: dict[str, Any], **kwargs: Any) -> Any:
        """Execute a tool handler synchronously, or run async handler in event loop."""
        if name not in self._handlers:
            raise KeyError(f"Tool '{name}' is not registered.")
        handler = self._handlers[name]
        res = self._call_handler(handler, args, **kwargs)
        if inspect.iscoroutine(res):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(res)
            return res
        return res

    async def aexecute(self, name: str, args: dict[str, Any], **kwargs: Any) -> Any:
        """Execute a tool handler asynchronously (supports both sync and async handlers)."""
        if name not in self._handlers:
            raise KeyError(f"Tool '{name}' is not registered.")
        handler = self._handlers[name]
        res = self._call_handler(handler, args, **kwargs)
        if inspect.iscoroutine(res):
            return await res
        return res
