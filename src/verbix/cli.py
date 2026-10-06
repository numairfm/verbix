from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

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
from verbix.minisweagent.models.litellm_model import LitellmModel, LitellmModelConfig
from verbix.runtime.runtime import Runtime
from verbix.voice.callback_sink import select_sink
from verbix.voice.capture import PTTCapture
from verbix.voice.pipeline import VoicePipeline
from verbix.voice.player import AudioPlayer
from verbix.voice.pocket_tts import PocketTTSEngine
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
            os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-3.5-flash-lite"),
        ),
        help="Model identifier (default: openrouter/google/gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("VERBIX_BASE_URL", None),
        help="API base URL (optional, defaults to provider standard endpoint)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("VERBIX_API_KEY", None),
        help="Provider API key (optional, LiteLLM auto-detects from environment per provider)",
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
        "--audio-sink",
        choices=("auto", "callback", "subprocess"),
        default="auto",
        help="Audio playback backend (default: auto — callback if sounddevice is installed, else subprocess).",
    )
    parser.add_argument(
        "--toggle",
        action="store_true",
        help="Send voice toggle event to running Verbix daemon over IPC socket.",
    )
    parser.add_argument(
        "--show-thoughts",
        action="store_true",
        help="Print model reasoning and thought deltas to stderr.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom persona or instructions layered onto Verbix's core system prompt.",
    )
    return parser


def build_runtime(args: argparse.Namespace) -> Runtime:
    """Builds and initializes the Runtime from parsed command-line arguments."""
    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise ValueError(f"Workspace directory '{workspace}' does not exist.")

    model_kwargs: dict[str, Any] = {}
    if getattr(args, "max_tokens", None):
        model_kwargs["max_tokens"] = args.max_tokens
    if getattr(args, "base_url", None):
        model_kwargs["api_base"] = args.base_url
    if getattr(args, "api_key", None):
        model_kwargs["api_key"] = args.api_key

    receptionist_model_name = getattr(args, "receptionist_model", None) or args.model
    fg_config = LitellmModelConfig(
        model_name=receptionist_model_name,
        model_kwargs=model_kwargs,
    )
    foreground_model = LitellmModel(config_class=lambda **kw: fg_config)

    worker_model = None
    if getattr(args, "worker_model", None):
        worker_config = LitellmModelConfig(
            model_name=args.worker_model,
            model_kwargs=model_kwargs,
        )
        worker_model = LitellmModel(config_class=lambda **kw: worker_config)

    return Runtime(
        workspace=workspace,
        provider=foreground_model,
        worker_model=worker_model,
        system_prompt=getattr(args, "system_prompt", None),
    )


class CliPrinter:
    """Renders the conversation as one assistant voice.

    Assistant replies go to stdout under a ``verbix:`` prefix; thought and tool
    diagnostics go to stderr so stdout stays a clean transcript. Tracking the
    delivered text lets the runtime trim history on barge-in and decide whether
    a completion was actually voiced.
    """

    SPEAKER = "verbix: "

    def __init__(self, show_thoughts: bool = False) -> None:
        self.show_thoughts = show_thoughts
        self.in_thought = False
        self._delivered: list[str] = []
        self._prefix_open = False
        self._wrote_text = False

    def _open_prefix(self) -> None:
        if not self._prefix_open:
            now_str = datetime.now().astimezone().strftime("%H:%M:%S")
            sys.stdout.write(f"\r\033[K[{now_str}] {self.SPEAKER}")
            sys.stdout.flush()
            self._prefix_open = True

    async def emit(self, event: AgentEvent) -> None:
        if isinstance(event, ThoughtDelta):
            if self.show_thoughts:
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
        sink: Any = None
        if args.voice:
            stt = FasterWhisperSTT()
            tts = PocketTTSEngine()
            sink = select_sink(
                getattr(args, "audio_sink", "auto"),
                sample_rate=24000,
                audio_format="f32",
            )
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
            await runtime.aclose()
            if tts:
                await tts.aclose()
            getattr(sink, "close", lambda: None)()
        return 0

    if args.voice:
        stt = FasterWhisperSTT()
        tts = PocketTTSEngine()
        sink = select_sink(
            getattr(args, "audio_sink", "auto"), sample_rate=24000, audio_format="f32"
        )
        player = AudioPlayer(tts=tts, sink=sink)
        capture = PTTCapture()
        printer = CliPrinter(show_thoughts=getattr(args, "show_thoughts", False))
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
                now_str = datetime.now().astimezone().strftime("%H:%M:%S")
                sys.stdout.write(f"[{now_str}] you> ")
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
            await runtime.aclose()
            await tts.aclose()
            getattr(sink, "close", lambda: None)()
        return 0

    # Text CLI Mode: single foreground queue with immediate completion printing
    printer = CliPrinter(show_thoughts=getattr(args, "show_thoughts", False))
    runtime.wake_output = printer

    def _cli_on_wake(speech: str) -> None:
        if not args.prompt:
            now_str = datetime.now().astimezone().strftime("%H:%M:%S")
            sys.stdout.write(f"[{now_str}] you> ")
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
            await runtime.aclose()
            return 0
        except asyncio.CancelledError:
            await runtime.aclose()
            return 130

    # Interactive REPL mode
    sys.stdout.write(
        f"verbix (model: {args.model}, workspace: {runtime.workspace})\n"
    )
    sys.stdout.write("Type 'exit' or 'quit' to end session.\n\n")
    sys.stdout.flush()

    while True:
        try:
            now_str = datetime.now().astimezone().strftime("%H:%M:%S")
            sys.stdout.write(f"[{now_str}] you> ")
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

    await runtime.aclose()
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
