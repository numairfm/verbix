from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

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
from verbix.runtime.runtime import Runtime
from verbix.voice.capture import PTTCapture
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer
from verbix.voice.pocket_tts import PocketTTSEngine
from verbix.voice.sinks import SubprocessAudioSink
from verbix.voice.stt import FasterWhisperSTT
 
logger = logging.getLogger(__name__)


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
        "--receptionist-model",
        type=str,
        default=None,
        help="Model for the receptionist agent (overrides --model).",
    )
    parser.add_argument(
        "--worker-model",
        type=str,
        default=None,
        help="Model for background autonomous workers (overrides --model).",
    )
    parser.add_argument(
        "--isolate-worktrees",
        action="store_true",
        help="Run background workers in isolated git worktrees.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Allow outbound network access in worker sandbox.",
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
    return parser


def build_runtime(args: argparse.Namespace) -> Runtime:
    """Builds and initializes the Runtime from parsed command-line arguments."""
    api_key = (
        args.api_key
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("VERBIX_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError(
            "No API key found. Set OPENROUTER_API_KEY in .env or environment, or use --api-key."
        )

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise ValueError(f"Workspace directory '{workspace}' does not exist.")

    provider = ProviderClient(
        base_url=args.base_url,
        model=args.model,
        api_key=api_key,
        max_tokens=args.max_tokens,
    )
    return Runtime(
        workspace=workspace,
        provider=provider,
    )


class CliPrinter:
    """Renders the conversation as one assistant voice.

    Assistant replies go to stdout under a ``verbix:`` prefix; thought and tool
    diagnostics go to stderr so stdout stays a clean transcript. Tracking the
    delivered text lets the runtime trim history on barge-in and decide whether
    a completion was actually voiced.
    """

    SPEAKER = "verbix: "

    def __init__(self) -> None:
        self.in_thought = False
        self._delivered: list[str] = []
        self._prefix_open = False
        self._wrote_text = False

    def _open_prefix(self) -> None:
        if not self._prefix_open:
            sys.stdout.write("\r\033[K" + self.SPEAKER)
            sys.stdout.flush()
            self._prefix_open = True

    async def emit(self, event: AgentEvent) -> None:
        if isinstance(event, ThoughtDelta):
            if not self.in_thought:
                sys.stderr.write("\n[thought] ")
                self.in_thought = True
            sys.stderr.write(event.thought)
            sys.stderr.flush()
        elif isinstance(event, TextDelta):
            if self.in_thought:
                sys.stderr.write("\n")
                self.in_thought = False
            self._open_prefix()
            sys.stdout.write(event.text)
            sys.stdout.flush()
            self._delivered.append(event.text)
            self._wrote_text = True
        elif isinstance(event, ToolCallStarted):
            if self.in_thought:
                sys.stderr.write("\n")
                self.in_thought = False
            if self._wrote_text:
                sys.stdout.write("\n")
                sys.stdout.flush()
                self._prefix_open = False
                self._wrote_text = False
            sys.stderr.write(f"[tool: {event.tool_name}({event.arguments})]\n")
            sys.stderr.flush()
        elif isinstance(event, ToolCallFinished):
            status = "error" if event.is_error else "ok"
            preview = event.observation.strip()
            if len(preview) > 120:
                preview = preview[:120] + "..."
            sys.stderr.write(f"[{event.call_id} -> {status}: {preview}]\n")
            sys.stderr.flush()
        elif isinstance(event, TurnComplete):
            if self.in_thought:
                sys.stderr.write("\n")
                self.in_thought = False
            final = event.final_text.strip()
            if final and final not in "".join(self._delivered):
                self._open_prefix()
                sys.stdout.write(final)
                sys.stdout.flush()
                self._delivered.append(final)
                self._wrote_text = True
            if self._wrote_text:
                sys.stdout.write("\n")
                sys.stdout.flush()
            self._prefix_open = False
            self._wrote_text = False
            self._delivered.clear()

    def delivered_text(self) -> str:
        return "".join(self._delivered)

    def reset_delivered(self) -> None:
        self.in_thought = False
        self._delivered.clear()
        self._prefix_open = False
        self._wrote_text = False


async def async_main(args: argparse.Namespace) -> int:
    is_toggle = args.toggle or (args.prompt and args.prompt.strip().lower() == "toggle")

    if is_toggle:
        socket_path = args.socket_path or (
            Path.home() / ".local" / "share" / "verbix" / "verbix.sock"
        )
        client = VerbixClient(socket_path=socket_path)
        try:
            await client.connect()
            res = await client.toggle()
            action = res.get("action", "unknown")
            sys.stdout.write(f"Voice toggle: {action}\n")
            await client.close()
            return 0
        except (OSError, RuntimeError) as e:
            sys.stderr.write(f"Error connecting to Verbix daemon: {e}\n")
            return 1

    try:
        runtime = build_runtime(args)
    except ValueError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1

    if args.serve:
        socket_path = args.socket_path or (
            Path.home() / ".local" / "share" / "verbix" / "verbix.sock"
        )
        pipeline: VoicePipeline | None = None
        tts: PocketTTSEngine | None = None
        if args.voice:
            stt = FasterWhisperSTT()
            tts = PocketTTSEngine()
            sink = SubprocessAudioSink(sample_rate=24000, audio_format="f32")
            player = AudioPlayer(tts=tts, sink=sink)
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
        stt = FasterWhisperSTT()
        tts = PocketTTSEngine()
        sink = SubprocessAudioSink(sample_rate=24000, audio_format="f32")
        player = AudioPlayer(tts=tts, sink=sink)
        capture = PTTCapture()
        printer = CliPrinter()
        pipeline = VoicePipeline(
            runtime=runtime, stt=stt, player=player, capture=capture, printer=printer
        )

        sys.stdout.write(
            f"verbix voice mode (model: {args.model}, workspace: {runtime.workspace})\n"
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
                sys.stdout.write("you> ")
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
                        active_task = getattr(pipeline, "_active_voice_task", None)
                        if active_task:
                            try:
                                await active_task
                            except asyncio.CancelledError:
                                pass
                            except Exception:
                                logger.exception("Active voice task failed")
                else:
                    await pipeline.execute_voice_turn(text)
        except (KeyboardInterrupt, asyncio.CancelledError):
            sys.stdout.write("\n")
        finally:
            await tts.aclose()
        return 0

    # Text CLI Mode: single foreground queue with immediate completion printing
    printer = CliPrinter()
    runtime.wake_output = printer

    def _cli_on_wake(speech: str) -> None:
        if not args.prompt:
            sys.stdout.write("you> ")
            sys.stdout.flush()

    runtime.on_wake = _cli_on_wake

    if args.prompt:
        try:
            await runtime.submit(args.prompt, printer)
            # Wait until all background tasks and wake turns have finished
            while runtime.list_active_tasks() or not runtime.is_idle:
                await asyncio.sleep(0.1)
            sys.stdout.write("\n")
            sys.stdout.flush()
            return 0
        except asyncio.CancelledError:
            return 130

    # Interactive REPL mode
    sys.stdout.write(
        f"verbix (model: {args.model}, workspace: {runtime.workspace})\n"
    )
    sys.stdout.write("Type 'exit' or 'quit' to end session.\n\n")
    sys.stdout.flush()

    while True:
        try:
            sys.stdout.write("you> ")
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

            printer.reset_delivered()
            await runtime.submit(prompt, printer)
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
