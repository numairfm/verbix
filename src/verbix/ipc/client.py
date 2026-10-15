from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from verbix.ipc.protocol import IPCEvent, IPCRequest, IPCResponse


class VerbixClient:
    """Async client for communicating with a running Verbix IPC server."""

    def __init__(
        self,
        socket_path: Path | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.socket_path = socket_path
        self.host = host
        self.port = port
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None

    async def connect(self) -> None:
        """Connects to the Verbix IPC server over UDS or TCP."""
        if self.socket_path:
            open_unix = getattr(asyncio, "open_unix_connection", None)
            if callable(open_unix):
                self._reader, self._writer = await open_unix(
                    path=str(self.socket_path)
                )
            else:
                endpoint = self.socket_path.read_text(encoding="utf-8").strip()
                host_str, port_str = endpoint.rsplit(":", 1)
                self._reader, self._writer = await asyncio.open_connection(
                    host=host_str, port=int(port_str)
                )
        elif self.host and self.port:
            self._reader, self._writer = await asyncio.open_connection(
                host=self.host, port=self.port
            )
        else:
            raise ValueError("Must specify either socket_path or host/port.")

    async def close(self) -> None:
        """Closes the connection to the server."""
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, RuntimeError):
                pass
            self._writer = None
            self._reader = None

    async def ping(self) -> bool:
        """Sends ping and waits for pong."""
        req = IPCRequest(id="ping-1", action="ping")
        resp = await self._send_request(req)
        return resp.data.get("pong", False) is True

    async def prompt(self, text: str) -> AsyncIterator[IPCEvent]:
        """Sends a prompt and yields streaming IPCEvents until turn_complete."""
        if not self._writer or not self._reader:
            raise RuntimeError("Client not connected.")

        req = IPCRequest(
            id=f"prompt-{asyncio.get_event_loop().time()}", action="prompt", prompt=text
        )
        req_line = json.dumps(req.model_dump(exclude_none=True)) + "\n"
        self._writer.write(req_line.encode("utf-8"))
        await self._writer.drain()

        while True:
            line = await self._reader.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue

            data = json.loads(line_str)
            # Check if this is the final IPCResponse for the prompt request
            if "status" in data and data.get("id") == req.id:
                break

            yield IPCEvent.model_validate(data)

    async def get_status(self, task_id: str | None = None) -> str:
        """Queries task status."""
        req = IPCRequest(id="status-1", action="status", task_id=task_id)
        resp = await self._send_request(req)
        return str(resp.data.get("summary", ""))

    async def cancel_task(self, task_id: str) -> bool:
        """Cancels a task."""
        req = IPCRequest(id="cancel-1", action="cancel", task_id=task_id)
        resp = await self._send_request(req)
        return bool(resp.data.get("cancelled", False))

    async def toggle(self) -> dict[str, Any]:
        """Toggles voice capture/playback."""
        req = IPCRequest(id="toggle-1", action="toggle")
        resp = await self._send_request(req)
        return resp.data

    async def _send_request(self, req: IPCRequest) -> IPCResponse:
        """Sends a single request and awaits the matching response."""
        if not self._writer or not self._reader:
            raise RuntimeError("Client not connected.")

        req_line = json.dumps(req.model_dump(exclude_none=True)) + "\n"
        self._writer.write(req_line.encode("utf-8"))
        await self._writer.drain()

        while True:
            line = await self._reader.readline()
            if not line:
                raise ConnectionError("Server closed connection unexpectedly.")
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            data = json.loads(line_str)
            if "status" in data and data.get("id") == req.id:
                return IPCResponse.model_validate(data)
