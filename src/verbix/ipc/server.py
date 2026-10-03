from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verbix.core.events import (
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.ipc.protocol import IPCEvent, IPCRequest, IPCResponse

if TYPE_CHECKING:
    from verbix.runtime.tasks import TaskRuntime
    from verbix.voice.pipeline import VoicePipeline

logger = logging.getLogger(__name__)


class VerbixServer:
    """Headless Unix Domain Socket / TCP server for communicating with Verbix runtime."""

    def __init__(
        self,
        runtime: TaskRuntime,
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
        self._event_forwarder_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Starts listening on Unix domain socket or TCP host/port."""
        if self.socket_path:
            self.socket_path.parent.mkdir(parents=True, exist_ok=True)
            if self.socket_path.exists():
                self.socket_path.unlink()

            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.socket_path),
            )
            logger.info("Verbix IPC server listening on UDS: %s", self.socket_path)
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

        # Start forwarding runtime events to subscribed clients
        self._event_forwarder_task = asyncio.create_task(self._forward_runtime_events())

    async def stop(self) -> None:
        """Stops the server and cleans up resources."""
        if self._event_forwarder_task and not self._event_forwarder_task.done():
            self._event_forwarder_task.cancel()

        for writer in list(self._clients):
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, RuntimeError):
                pass
        self._clients.clear()

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
                    try:
                        writer.write(msg_bytes)
                        await writer.drain()
                    except (OSError, RuntimeError):
                        self._clients.discard(writer)
        except asyncio.CancelledError:
            pass
        finally:
            self.runtime.unsubscribe(q)

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handles a single client connection over the socket."""
        self._clients.add(writer)
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
                    writer.write(err_resp.to_jsonl())
                    await writer.drain()
                    continue

                await self._dispatch_request(request, writer)

        except (OSError, RuntimeError):
            pass
        finally:
            self._clients.discard(writer)
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
            writer.write(resp.to_jsonl())
            await writer.drain()

        elif action == "prompt":
            if not request.prompt:
                resp = IPCResponse(
                    id=request.id, status="error", error="Missing 'prompt' field"
                )
                writer.write(resp.to_jsonl())
                await writer.drain()
                return

            try:
                async for event in self.runtime.step_receptionist(
                    user_input=request.prompt
                ):
                    ev_type = "unknown"
                    payload: dict[str, Any] = {}

                    if isinstance(event, TextDelta):
                        ev_type = "text_delta"
                        payload = {"text": event.text}
                    elif isinstance(event, ThoughtDelta):
                        ev_type = "thought_delta"
                        payload = {"thought": event.thought}
                    elif isinstance(event, ToolCallStarted):
                        ev_type = "tool_call_started"
                        payload = {
                            "call_id": event.call_id,
                            "tool_name": event.tool_name,
                            "arguments": event.arguments,
                        }
                    elif isinstance(event, ToolCallFinished):
                        ev_type = "tool_call_finished"
                        payload = {
                            "call_id": event.call_id,
                            "observation": event.observation,
                            "is_error": event.is_error,
                        }
                    elif isinstance(event, TurnComplete):
                        ev_type = "turn_complete"
                        payload = {"final_text": event.final_text}

                    writer.write(
                        IPCEvent(
                            type=ev_type, request_id=request.id, payload=payload
                        ).to_jsonl()
                    )
                    await writer.drain()

                # Send success response indicating turn ended
                writer.write(IPCResponse(id=request.id, status="ok").to_jsonl())
                await writer.drain()
            except (RuntimeError, OSError, ValueError) as e:
                err_resp = IPCResponse(id=request.id, status="error", error=str(e))
                writer.write(err_resp.to_jsonl())
                await writer.drain()

        elif action == "status":
            summary = self.runtime.get_task_status_summary(request.task_id)
            resp = IPCResponse(id=request.id, status="ok", data={"summary": summary})
            writer.write(resp.to_jsonl())
            await writer.drain()

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
            writer.write(resp.to_jsonl())
            await writer.drain()

        elif action == "steer":
            if not request.task_id or not request.instruction:
                resp = IPCResponse(
                    id=request.id,
                    status="error",
                    error="Missing 'task_id' or 'instruction'",
                )
            else:
                ok = await self.runtime.steer_task(request.task_id, request.instruction)
                resp = IPCResponse(
                    id=request.id, status="ok" if ok else "error", data={"steered": ok}
                )
            writer.write(resp.to_jsonl())
            await writer.drain()

        elif action == "ptt_down":
            if self.pipeline:
                await self.pipeline.on_ptt_down()
            resp = IPCResponse(id=request.id, status="ok")
            writer.write(resp.to_jsonl())
            await writer.drain()

        elif action == "ptt_up":
            transcription = None
            if self.pipeline:
                transcription = await self.pipeline.on_ptt_up()
            resp = IPCResponse(
                id=request.id, status="ok", data={"transcription": transcription}
            )
            writer.write(resp.to_jsonl())
            await writer.drain()

        else:
            resp = IPCResponse(
                id=request.id, status="error", error=f"Unknown action: '{action}'"
            )
            writer.write(resp.to_jsonl())
            await writer.drain()
