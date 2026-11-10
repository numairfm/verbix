#!/usr/bin/env python3
"""Headless fake voice service for shell integration tests.

Speaks the existing Verbix JSONL protocol on a Unix socket. It is intentionally
small: ping, toggle with floor events, status, and clean shutdown.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

RUNNING = True


def _stop(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


def send(conn: socket.socket, message: dict) -> None:
    conn.sendall((json.dumps(message) + "\n").encode("utf-8"))


def event(kind: str, payload: dict) -> dict:
    return {"type": kind, "payload": payload}


def runtime_event(kind: str, payload: dict) -> dict:
    return event("runtime_event", {"type": kind, "payload": payload})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--start-delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.start_delay:
        time.sleep(args.start_delay)

    Path(args.pid_file).write_text(str(os.getpid()), encoding="utf-8")
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    path = Path(args.socket)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.unlink()
    except FileNotFoundError:
        pass

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(4)
    server.settimeout(0.1)
    conn: socket.socket | None = None

    try:
        while RUNNING:
            if conn is None:
                try:
                    conn, _ = server.accept()
                    conn.settimeout(0.1)
                except TimeoutError:
                    continue
            try:
                data = conn.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                conn = None
                continue
            if not data:
                conn.close()
                conn = None
                continue
            for line in data.splitlines():
                if not line.strip():
                    continue
                request = json.loads(line)
                action = request.get("action", "")
                request_id = request.get("id")
                if action == "ping":
                    send(conn, {"id": request_id, "status": "ok", "data": {"pong": True}})
                elif action == "status":
                    send(conn, {"id": request_id, "status": "ok", "data": {"summary": "idle"}})
                elif action == "toggle":
                    send(conn, runtime_event("floor_state", {"state": "listening"}))
                    send(
                        conn,
                        {
                            "id": request_id,
                            "status": "ok",
                            "data": {"action": "recording"},
                        },
                    )
                    time.sleep(0.05)
                    send(
                        conn,
                        runtime_event(
                            "user_transcript",
                            {"text": "what time is it in tokyo", "duration": 1.4},
                        ),
                    )
                    send(conn, event("text_delta", {"text": "It is "}))
                    send(conn, event("text_delta", {"text": "6:42 PM in Tokyo."}))
                    send(
                        conn,
                        event(
                            "turn_complete",
                            {
                                "final_text": "It is 6:42 PM in Tokyo.",
                                "exit_reason": "completed",
                                "is_error": False,
                            },
                        ),
                    )
                    send(
                        conn,
                        runtime_event(
                            "task_registered",
                            {
                                "task_id": "task-1",
                                "objective": "summarize this repo",
                            },
                        ),
                    )
                    send(
                        conn,
                        runtime_event(
                            "task_completed",
                            {
                                "task_id": "task-1",
                                "result": "The repo is a voice agent with workers.",
                            },
                        ),
                    )
                    send(conn, event("text_delta", {"text": "Task one is complete."}))
                    send(
                        conn,
                        event(
                            "turn_complete",
                            {
                                "final_text": "Task one is complete.",
                                "exit_reason": "completed",
                                "is_error": False,
                            },
                        ),
                    )
                    send(
                        conn,
                        runtime_event(
                            "task_registered",
                            {
                                "task_id": "task-2",
                                "objective": "read the logs",
                            },
                        ),
                    )
                    send(
                        conn,
                        runtime_event(
                            "task_failed",
                            {
                                "task_id": "task-2",
                                "result": "log file missing",
                            },
                        ),
                    )
                    send(
                        conn,
                        runtime_event(
                            "task_registered",
                            {
                                "task_id": "task-3",
                                "objective": "index the workspace",
                            },
                        ),
                    )
                    send(
                        conn,
                        runtime_event("task_cancelled", {"task_id": "task-3"}),
                    )
                    time.sleep(0.05)
                    send(conn, runtime_event("floor_state", {"state": "speaking"}))
                    time.sleep(0.05)
                    send(conn, runtime_event("floor_state", {"state": "idle"}))
                else:
                    send(
                        conn,
                        {
                            "id": request_id,
                            "status": "error",
                            "error": f"unknown action: {action}",
                        },
                    )
    finally:
        if conn is not None:
            conn.close()
        server.close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
