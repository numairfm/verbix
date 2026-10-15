from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verbix.core.events import (
    AgentEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.ipc.protocol import IPCEvent, IPCRequest, IPCResponse

if TYPE_CHECKING:
    from verbix.voice.pipeline import VoicePipeline

logger = logging.getLogger(__name__)


def event_to_payload(event: AgentEvent) -> tuple[str, dict[str, Any]]:
    """Converts a domain AgentEvent into an IPC event type and payload dictionary."""
    if isinstance(event, TextDelta):
        return "text_delta", {"text": event.text}
    if isinstance(event, ThoughtDelta):
        return "thought_delta", {"thought": event.thought}
    if isinstance(event, ToolCallStarted):
        return "tool_call_started", {
            "call_id": event.call_id,
            "tool_name": event.tool_name,
            "arguments": event.arguments,
        }
    if isinstance(event, ToolCallFinished):
        return "tool_call_finished", {
            "call_id": event.call_id,
            "observation": event.observation,
            "is_error": event.is_error,
        }
    if isinstance(event, TurnComplete):
        return "turn_complete", {
            "final_text": event.final_text,
            "exit_reason": event.exit_reason,
            "is_error": event.is_error,
        }
    return "unknown", {}


class BroadcastWakeOutput:
    """Delivers wake-turn events to all connected IPC clients."""

    def __init__(self, server: VerbixServer) -> None:
        self.server = server
        self._delivered: list[str] = []

    async def emit(self, event: AgentEvent) -> None:
        if isinstance(event, TextDelta):
            self._delivered.append(event.text)
        ev_type, payload = event_to_payload(event)
        ipc_event = IPCEvent(type=ev_type, payload=payload)
        msg_bytes = ipc_event.to_jsonl()
        for writer in list(self.server._clients):
            await self.server._send(writer, msg_bytes)

    def delivered_text(self) -> str:
        return "".join(self._delivered)

    def reset_delivered(self) -> None:
        self._delivered.clear()


class VerbixServer:
    """Headless Unix Domain Socket / TCP server for communicating with Verbix runtime."""

    def __init__(
        self,
        runtime: Any,
        pipeline: VoicePipeline | None = None,
        socket_path: Path | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self.runtime = runtime
        self.pipeline = pipeline
        self.socket_path = socket_path
        self.host = host
        self.port = port

        self._server: asyncio.Server | None = None
        self._clients: set[asyncio.StreamWriter] = set()
        self._writer_locks: dict[asyncio.StreamWriter, asyncio.Lock] = {}
        self._client_tasks: set[asyncio.Task[Any]] = set()
        self._prompt_tasks: set[asyncio.Task[Any]] = set()
        self._event_forwarder_task: asyncio.Task[None] | None = None

        # Configure wake and foreground voice turn output fan-out to IPC clients
        bcast = BroadcastWakeOutput(self)
        turn_out = getattr(self.pipeline, "turn_output", None) if self.pipeline else None
        if turn_out is not None:
            turn_out.on_event = bcast.emit
        if self.runtime.wake_output is None:
            self.runtime.wake_output = bcast
        elif (
            self.runtime.wake_output is not turn_out
            and not isinstance(self.runtime.wake_output, BroadcastWakeOutput)
        ):
            from verbix.runtime.announcer import WakeOutputFanOut

            self.runtime.wake_output = WakeOutputFanOut(
                self.runtime.wake_output, bcast.emit
            )

    async def _send(self, writer: asyncio.StreamWriter, msg_bytes: bytes) -> bool:
        """Sends data to a client writer with per-writer synchronization."""
        lock = self._writer_locks.get(writer)
        if lock is None:
            return False
        try:
            async with lock:
                writer.write(msg_bytes)
                await writer.drain()
                return True
        except (OSError, RuntimeError):
            self._remove_client(writer)
            return False

    def _remove_client(self, writer: asyncio.StreamWriter) -> None:
        self._clients.discard(writer)
        self._writer_locks.pop(writer, None)

    async def start(self) -> None:
        """Starts listening on Unix domain socket or TCP host/port."""
        if self.socket_path:
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self.socket_path.exists():
                self.socket_path.unlink()

            start_unix = getattr(asyncio, "start_unix_server", None)
            if callable(start_unix):
                self._server = await start_unix(
                    self._handle_client,
                    path=str(self.socket_path),
                )
                logger.info("Verbix IPC server listening on UDS: %s", self.socket_path)
            else:
                self._server = await asyncio.start_server(
                    self._handle_client,
                    host="127.0.0.1",
                    port=0,
                )
                bound_port = self._server.sockets[0].getsockname()[1]
                self.socket_path.write_text(
                    f"127.0.0.1:{bound_port}\n", encoding="utf-8"
                )
                logger.info(
                    "Verbix IPC server listening on loopback TCP 127.0.0.1:%d (%s)",
                    bound_port,
                    self.socket_path,
                )
        elif self.host and self.port:
            self._server = await asyncio.start_server(
                self._handle_client,
                host=self.host,
                port=self.port,
            )
            logger.info(
                "Verbix IPC server listening on TCP %s:%d", self.host, self.port
            )
        else:
            raise ValueError("Must specify either socket_path or host/port.")

        self._event_forwarder_task = asyncio.create_task(self._forward_runtime_events())

    async def stop(self) -> None:
        """Stops the server and cleans up resources."""
        if self._event_forwarder_task and not self._event_forwarder_task.done():
            self._event_forwarder_task.cancel()
            try:
                await self._event_forwarder_task
            except asyncio.CancelledError:
                pass
            self._event_forwarder_task = None

        if self._prompt_tasks:
            for task in list(self._prompt_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*list(self._prompt_tasks), return_exceptions=True)
            self._prompt_tasks.clear()

        if self._client_tasks:
            for task in list(self._client_tasks):
                if not task.done():
                    task.cancel()
            await asyncio.gather(*list(self._client_tasks), return_exceptions=True)
            self._client_tasks.clear()

        for writer in list(self._clients):
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, RuntimeError):
                pass
        self._clients.clear()
        self._writer_locks.clear()

        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

        if self.socket_path and self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

    async def _forward_runtime_events(self) -> None:
        """Listens to runtime events and broadcasts them to all connected clients."""
        q = self.runtime.subscribe()
        try:
            while True:
                event_data = await q.get()
                ipc_event = IPCEvent(type="runtime_event", payload=event_data)
                msg_bytes = ipc_event.to_jsonl()
                for writer in list(self._clients):
                    await self._send(writer, msg_bytes)
        except asyncio.CancelledError:
            pass
        finally:
            self.runtime.unsubscribe(q)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handles a single client connection over the socket."""
        self._clients.add(writer)
        self._writer_locks[writer] = asyncio.Lock()
        task = asyncio.current_task()
        if task:
            self._client_tasks.add(task)
            task.add_done_callback(self._client_tasks.discard)
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                line_str = line.decode("utf-8").strip()
                if not line_str:
                    continue

                try:
                    raw_json = json.loads(line_str)
                    request = IPCRequest.model_validate(raw_json)
                except (json.JSONDecodeError, ValueError) as e:
                    err_resp = IPCResponse(
                        status="error", error=f"Invalid request JSON: {e}"
                    )
                    await self._send(writer, err_resp.to_jsonl())
                    continue

                await self._dispatch_request(request, writer)

        except (OSError, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            self._remove_client(writer)
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, RuntimeError):
                pass

    async def _dispatch_request(
        self, request: IPCRequest, writer: asyncio.StreamWriter
    ) -> None:
        """Processes an incoming IPC request."""
        action = request.action.lower()

        if action == "ping":
            resp = IPCResponse(id=request.id, status="ok", data={"pong": True})
            await self._send(writer, resp.to_jsonl())

        elif action == "prompt":
            if not request.prompt:
                resp = IPCResponse(
                    id=request.id, status="error", error="Missing 'prompt' field"
                )
                await self._send(writer, resp.to_jsonl())
                return

            class IPCTurnOutput:
                def __init__(self, server: VerbixServer, req_id: str, w: asyncio.StreamWriter) -> None:
                    self.server = server
                    self.req_id = req_id
                    self.w = w
                    self._delivered: list[str] = []

                async def emit(self, event: AgentEvent) -> None:
                    if isinstance(event, TextDelta):
                        self._delivered.append(event.text)
                    ev_type, payload = event_to_payload(event)
                    msg_bytes = IPCEvent(
                        type=ev_type, request_id=self.req_id, payload=payload
                    ).to_jsonl()
                    await self.server._send(self.w, msg_bytes)

                def delivered_text(self) -> str:
                    return "".join(self._delivered)

                def reset_delivered(self) -> None:
                    self._delivered.clear()

            async def _run_prompt() -> None:
                ipc_out = IPCTurnOutput(self, request.id, writer)
                try:
                    await self.runtime.submit(request.prompt or "", ipc_out)
                    await self._send(writer, IPCResponse(id=request.id, status="ok").to_jsonl())
                except asyncio.CancelledError:
                    await self._send(
                        writer,
                        IPCResponse(
                            id=request.id, status="error", error="prompt cancelled"
                        ).to_jsonl(),
                    )
                    raise
                except Exception as e:  # noqa: BLE001
                    err_resp = IPCResponse(id=request.id, status="error", error=str(e))
                    await self._send(writer, err_resp.to_jsonl())

            task = asyncio.create_task(_run_prompt())
            self._prompt_tasks.add(task)
            task.add_done_callback(self._prompt_tasks.discard)

        elif action == "status":
            summary = self.runtime.get_task_status_summary(request.task_id)
            resp = IPCResponse(id=request.id, status="ok", data={"summary": summary})
            await self._send(writer, resp.to_jsonl())

        elif action == "cancel":
            if not request.task_id:
                resp = IPCResponse(
                    id=request.id, status="error", error="Missing 'task_id'"
                )
            else:
                ok = await self.runtime.cancel_task(request.task_id)
                resp = IPCResponse(
                    id=request.id,
                    status="ok" if ok else "error",
                    data={"cancelled": ok},
                )
            await self._send(writer, resp.to_jsonl())

        elif action == "toggle":
            data: dict[str, Any] = {}
            if self.pipeline:
                data = await self.pipeline.toggle()
            resp = IPCResponse(id=request.id, status="ok", data=data)
            await self._send(writer, resp.to_jsonl())

        else:
            resp = IPCResponse(
                id=request.id, status="error", error=f"Unknown action: '{action}'"
            )
            await self._send(writer, resp.to_jsonl())
