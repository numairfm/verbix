from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from verbix.core.agent import AgentSession
from verbix.core.events import (
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.provider.client import ProviderClient
from verbix.tools.bash import BashTool
from verbix.tools.filesystem import FileEditTool, FileReadTool, FileWriteTool


def load_env(env_path: Path | None = None) -> None:
    """Loads key-value pairs from .env into os.environ if not already set."""
    candidates = []
    if env_path:
        candidates.append(env_path)
    candidates.append(Path.cwd() / ".env")
    candidates.append(Path.home() / ".config" / "verbix" / ".env")

    for p in candidates:
        if p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val
                break
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="verbix",
        description="Verbix CLI agent interface",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Optional single prompt to execute. If omitted, launches interactive chat.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "VERBIX_MODEL",
            os.environ.get("OPENROUTER_MODEL", "google/gemini-2.5-flash"),
        ),
        help="Model identifier (default: google/gemini-2.5-flash)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get(
            "VERBIX_BASE_URL",
            os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        ),
        help="API base URL (default: https://openrouter.ai/api/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Provider API key (defaults to OPENROUTER_API_KEY or VERBIX_API_KEY)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root path (default: current working directory)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("VERBIX_MAX_TOKENS", "2048")),
        help="Maximum completion tokens per turn (default: 2048)",
    )
    return parser


async def run_turn(session: AgentSession, user_input: str) -> str:
    """Runs a single conversation turn, streaming output directly to stdout."""
    in_thought = False
    final_output = ""

    try:
        async for event in session.step(user_input=user_input):
            if isinstance(event, ThoughtDelta):
                if not in_thought:
                    sys.stdout.write("\n[thought] ")
                    in_thought = True
                sys.stdout.write(event.thought)
                sys.stdout.flush()
            elif isinstance(event, TextDelta):
                if in_thought:
                    sys.stdout.write("\n")
                    in_thought = False
                sys.stdout.write(event.text)
                sys.stdout.flush()
            elif isinstance(event, ToolCallStarted):
                if in_thought:
                    sys.stdout.write("\n")
                    in_thought = False
                sys.stdout.write(f"\n[tool: {event.tool_name}({event.arguments})]\n")
                sys.stdout.flush()
            elif isinstance(event, ToolCallFinished):
                status = "error" if event.is_error else "ok"
                preview = event.observation.strip()
                if len(preview) > 120:
                    preview = preview[:120] + "..."
                sys.stdout.write(f"[{event.call_id} -> {status}: {preview}]\n")
                sys.stdout.flush()
            elif isinstance(event, TurnComplete):
                if in_thought:
                    sys.stdout.write("\n")
                    in_thought = False
                final_output = event.final_text
    except asyncio.CancelledError:
        sys.stdout.write("\n[Interrupted]\n")
        sys.stdout.flush()
        raise

    sys.stdout.write("\n")
    sys.stdout.flush()
    return final_output


async def async_main(args: argparse.Namespace) -> int:
    api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("VERBIX_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        sys.stderr.write(
            "Error: No API key found. Set OPENROUTER_API_KEY in .env or environment, or use --api-key.\n"
        )
        return 1

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        sys.stderr.write(f"Error: Workspace directory '{workspace}' does not exist.\n")
        return 1

    provider = ProviderClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
    )

    tools = [
        FileReadTool(),
        FileWriteTool(),
        FileEditTool(),
        BashTool(),
    ]

    session = AgentSession(
        session_id="cli-session",
        system_prompt=(
            "You are Verbix, an expert voice and coding assistant. "
            "You have access to filesystem and bash tools to inspect, edit, and run code. "
            "Be direct, accurate, and concise."
        ),
        provider=provider,
        workspace=workspace,
        tools=tools,
    )

    if args.prompt:
        try:
            await run_turn(session, args.prompt)
            return 0
        except asyncio.CancelledError:
            return 130

    # Interactive REPL mode
    sys.stdout.write(f"Verbix CLI (model: {args.model}, workspace: {workspace})\n")
    sys.stdout.write("Type 'exit' or 'quit' to end session.\n\n")
    sys.stdout.flush()

    while True:
        try:
            sys.stdout.write("verbix> ")
            sys.stdout.flush()
            loop = asyncio.get_running_loop()
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            prompt = line.strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                break

            await run_turn(session, prompt)
        except (KeyboardInterrupt, asyncio.CancelledError):
            sys.stdout.write("\n")
            continue
        except EOFError:
            break

    return 0


def main(argv: list[str] | None = None) -> None:
    load_env()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        exit_code = asyncio.run(async_main(args))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
