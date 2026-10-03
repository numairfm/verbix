from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path

from verbix.core.agent import AgentSession
from verbix.core.events import (
    AgentEvent,
    TextDelta,
    ThoughtDelta,
    ToolCallFinished,
    ToolCallStarted,
    TurnComplete,
)
from verbix.ipc.client import VerbixClient
from verbix.ipc.server import VerbixServer
from verbix.provider.client import ProviderClient
from verbix.runtime.tasks import TaskRuntime
from verbix.tools.bash import BashTool
from verbix.tools.filesystem import FileEditTool, FileReadTool, FileWriteTool
from verbix.voice.capture import PTTCapture
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer
from verbix.voice.pocket_tts import PocketTTSEngine
from verbix.voice.sinks import SubprocessAudioSink
from verbix.voice.stt import FasterWhisperSTT


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
            os.environ.get("OPENROUTER_MODEL", "google/gemini-3.5-flash-lite"),
        ),
        help="Model identifier (default: google/gemini-3.5-flash-lite)",
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
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Run direct foreground coding agent instead of background receptionist runtime.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run as a headless background IPC daemon listening on Unix Domain Socket.",
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        default=None,
        help="Path for Unix Domain Socket (default: ~/.local/share/verbix/verbix.sock).",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Run in voice mode (Toggle -> Live STT -> Receptionist -> PocketTTS audio playback).",
    )
    parser.add_argument(
        "--toggle",
        action="store_true",
        help="Send voice toggle event to running Verbix daemon over IPC socket.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm pending action on running Verbix daemon.",
    )
    parser.add_argument(
        "--deny",
        action="store_true",
        help="Deny pending action on running Verbix daemon.",
    )
    return parser


async def run_stream(stream: AsyncIterator[AgentEvent]) -> str:
    """Consumes an AgentEvent stream and renders thought, text, and tools to stdout."""
    in_thought = False
    final_output = ""

    try:
        async for event in stream:
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


async def run_turn(session: AgentSession, user_input: str) -> str:
    """Runs a single conversation turn on an AgentSession, streaming output to stdout."""
    return await run_stream(session.step(user_input=user_input))


async def async_main(args: argparse.Namespace) -> int:
    is_toggle = args.toggle or (args.prompt and args.prompt.strip().lower() == "toggle")
    is_confirm = args.confirm or (
        args.prompt and args.prompt.strip().lower() == "confirm"
    )
    is_deny = args.deny or (args.prompt and args.prompt.strip().lower() == "deny")

    if is_toggle or is_confirm or is_deny:
        socket_path = args.socket_path or (
            Path.home() / ".local" / "share" / "verbix" / "verbix.sock"
        )
        client = VerbixClient(socket_path=socket_path)
        try:
            await client.connect()
            if is_toggle:
                res = await client.toggle()
                action = res.get("action", "unknown")
                sys.stdout.write(f"Voice toggle: {action}\n")
            elif is_confirm:
                ok = await client.confirm()
                sys.stdout.write(
                    f"Action confirmation: {'confirmed' if ok else 'failed'}\n"
                )
            elif is_deny:
                ok = await client.deny()
                sys.stdout.write(f"Action denial: {'denied' if ok else 'failed'}\n")
            await client.close()
            return 0
        except (OSError, RuntimeError) as e:
            sys.stderr.write(f"Error connecting to Verbix daemon: {e}\n")
            return 1

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

    if args.serve:
        socket_path = args.socket_path or (
            Path.home() / ".local" / "share" / "verbix" / "verbix.sock"
        )
        runtime = TaskRuntime(workspace=workspace, provider=provider)
        runtime.create_receptionist()

        pipeline: VoicePipeline | None = None
        tts: PocketTTSEngine | None = None
        if args.voice:
            stt = FasterWhisperSTT()
            tts = PocketTTSEngine()
            sink = SubprocessAudioSink(sample_rate=24000, audio_format="f32")
            player = AudioPlayer(tts=tts, sink=sink, runtime=runtime)
            capture = PTTCapture()
            pipeline = VoicePipeline(
                runtime=runtime, stt=stt, player=player, capture=capture
            )

        server = VerbixServer(
            runtime=runtime, pipeline=pipeline, socket_path=socket_path
        )
        await server.start()
        voice_state = "on" if pipeline else "off"
        sys.stdout.write(
            f"Verbix daemon listening on {socket_path} (voice={voice_state})\nPress Ctrl+C to terminate.\n"
        )
        sys.stdout.flush()
        try:
            while True:
                await asyncio.sleep(1.0)
        finally:
            await server.stop()
            if tts:
                await tts.aclose()
        return 0

    if args.voice:
        runtime = TaskRuntime(workspace=workspace, provider=provider)
        runtime.create_receptionist()
        stt = FasterWhisperSTT()
        tts = PocketTTSEngine()
        sink = SubprocessAudioSink(sample_rate=24000, audio_format="f32")
        player = AudioPlayer(tts=tts, sink=sink, runtime=runtime)
        capture = PTTCapture()
        pipeline = VoicePipeline(
            runtime=runtime, stt=stt, player=player, capture=capture
        )

        sys.stdout.write(
            f"Verbix Voice Mode (model: {args.model}, workspace: {workspace})\n"
        )
        sys.stdout.write(
            "Toggle Mode: Press [Enter] to start recording, speak, then press [Enter] to stop.\n"
            "If audio is playing, [Enter] halts playback and begins recording immediately.\n"
            "Or type a prompt directly. Type 'exit' or 'quit' to end session.\n\n"
        )
        sys.stdout.flush()

        loop = asyncio.get_running_loop()
        try:
            while True:
                # Check for pending confirmations
                if pipeline.is_confirmation_pending:
                    prompt_txt = (
                        pipeline.confirmation_prompt or "Action confirmation required"
                    )
                    sys.stdout.write(
                        f"\n[Confirm action]: {prompt_txt}\nPress 'y' to confirm, 'n' to deny: "
                    )
                    sys.stdout.flush()
                    line = await loop.run_in_executor(None, sys.stdin.readline)
                    ans = line.strip().lower()
                    if ans in ("y", "yes"):
                        pipeline.confirm()
                    else:
                        pipeline.deny()
                    continue

                sys.stdout.write("verbix [Enter=Toggle voice / type prompt]> ")
                sys.stdout.flush()
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                text = line.strip()
                if text.lower() in ("exit", "quit"):
                    break

                if not text:
                    res = await pipeline.toggle()
                    state = res.get("action")
                    if state in ("recording", "interrupted_and_recording"):
                        if state == "interrupted_and_recording":
                            sys.stdout.write("[Playback interrupted]\n")
                        sys.stdout.write(
                            "[Recording mic... Press Enter when done speaking]\n"
                        )
                        sys.stdout.flush()
                        await loop.run_in_executor(None, sys.stdin.readline)
                        sys.stdout.write("[Capture stopped. Earcon cue playing...]\n")
                        sys.stdout.flush()
                        stop_res = await pipeline.toggle()
                        transcribed = stop_res.get("transcription", "")
                        if transcribed:
                            sys.stdout.write(f"[User spoke]: {transcribed}\n")
                        else:
                            sys.stdout.write(
                                "[No speech detected or audio too short]\n"
                            )
                        sys.stdout.flush()
                else:
                    await pipeline.execute_voice_turn(text)
        except (KeyboardInterrupt, asyncio.CancelledError):
            sys.stdout.write("\n")
        finally:
            await tts.aclose()
        return 0

    if args.direct:
        tools = [
            FileReadTool(),
            FileWriteTool(),
            FileEditTool(),
            BashTool(),
        ]
        session = AgentSession(
            session_id="cli-direct",
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

        sys.stdout.write(
            f"Verbix Direct Agent (model: {args.model}, workspace: {workspace})\n"
        )
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

    # Default: TaskRuntime Receptionist orchestration
    runtime = TaskRuntime(workspace=workspace, provider=provider)
    runtime.create_receptionist()

    def print_announcements() -> None:
        while not runtime.announcement_q.empty():
            task_id, res = runtime.announcement_q.get_nowait()
            outcome = "succeeded" if res.success else "failed"
            sys.stdout.write(f"\n[Task {task_id} {outcome}]: {res.speech_summary}\n")
            sys.stdout.flush()

    if args.prompt:
        try:
            await run_stream(runtime.step_receptionist(args.prompt))
            while runtime.list_active_tasks():
                print_announcements()
                await asyncio.sleep(0.2)
            print_announcements()
            return 0
        except asyncio.CancelledError:
            return 130

    # Interactive REPL mode
    sys.stdout.write(
        f"Verbix Receptionist (model: {args.model}, workspace: {workspace})\n"
    )
    sys.stdout.write("Type 'exit' or 'quit' to end session.\n\n")
    sys.stdout.flush()

    while True:
        try:
            print_announcements()
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

            await run_stream(runtime.step_receptionist(prompt))
            print_announcements()
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
