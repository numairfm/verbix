from __future__ import annotations

import argparse
import asyncio
import logging
import os
import select
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from verbix.agent.models.litellm_model import LitellmModel, LitellmModelConfig
from verbix.auth import resolve_provider
from verbix.config import ensure_global_scaffold, load_config
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
from verbix.runtime.runtime import Runtime

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
        "--provider",
        default=None,
        help="Provider name from ~/.verbix/auth.json (overrides config file and VERBIX_PROVIDER).",
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
        "--worker-max-tokens",
        type=int,
        default=int(os.environ["VERBIX_WORKER_MAX_TOKENS"])
        if os.environ.get("VERBIX_WORKER_MAX_TOKENS")
        else None,
        help="Maximum completion tokens per turn for background workers (default: None / uncapped).",
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
        help="Run in voice mode (Live STT -> Receptionist -> Supertonic TTS audio playback).",
    )
    parser.add_argument(
        "--voice-mode",
        choices=("continuous", "ptt"),
        default="continuous",
        help="Voice interaction mode (default: continuous — hands-free conversation; ptt — push-to-talk).",
    )
    parser.add_argument(
        "--audio-sink",
        choices=("auto", "callback", "subprocess"),
        default="auto",
        help="Audio playback backend (default: auto — callback if sounddevice is installed, else subprocess).",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=None,
        help="Path to GGUF speech-to-text model file (default: auto-discover).",
    )
    parser.add_argument(
        "--dump-mic-dir",
        type=Path,
        default=None,
        help="Directory for per-turn mic-audio WAV dumps (debugging STT glitches; off by default).",
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
        "--show-worker-turns",
        action="store_true",
        help="Print background worker turns tagged with their task ID.",
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
    provider_name = load_config(workspace, args).get("provider")
    entry = resolve_provider(provider_name) if provider_name else None
    if getattr(args, "base_url", None):
        model_kwargs["api_base"] = args.base_url
    elif entry and entry.get("base_url"):
        model_kwargs["api_base"] = entry["base_url"]
    if getattr(args, "api_key", None):
        model_kwargs["api_key"] = args.api_key
    elif entry and entry.get("key"):
        model_kwargs["api_key"] = entry["key"]

    default_model = os.environ.get(
        "VERBIX_MODEL",
        os.environ.get("OPENROUTER_MODEL", "openrouter/google/gemini-3.5-flash-lite"),
    )
    if entry and entry.get("model") and args.model == default_model:
        args.model = entry["model"]
    receptionist_model_name = getattr(args, "receptionist_model", None) or args.model
    fg_config = LitellmModelConfig(
        model_name=receptionist_model_name,
        model_kwargs=model_kwargs,
    )
    foreground_model = LitellmModel(config_class=lambda **kw: fg_config)

    worker_model_name = getattr(args, "worker_model", None) or args.model
    worker_model_kwargs = dict(model_kwargs)
    if getattr(args, "worker_max_tokens", None):
        worker_model_kwargs["max_tokens"] = args.worker_max_tokens
    else:
        worker_model_kwargs.pop("max_tokens", None)

    worker_config = LitellmModelConfig(
        model_name=worker_model_name,
        model_kwargs=worker_model_kwargs,
    )
    worker_model = LitellmModel(config_class=lambda **kw: worker_config)

    flag_prompt = getattr(args, "system_prompt", None)
    system_prompt = flag_prompt.strip() if flag_prompt and flag_prompt.strip() else None

    return Runtime(
        workspace=workspace,
        provider=foreground_model,
        worker_model=worker_model,
        worker_echo=bool(getattr(args, "show_worker_turns", False))
        or os.environ.get("VERBIX_WORKER_ECHO") == "1",
        system_prompt=system_prompt,
    )


def drain_stdin() -> None:
    """Discards already-buffered stdin lines so stale Enters never become phantom commands."""
    try:
        while select.select([sys.stdin], [], [], 0)[0]:
            if not os.read(sys.stdin.fileno(), 4096):
                break
    except (OSError, ValueError):
        pass


def _note_turn_failed(task: asyncio.Task) -> None:
    """Logs background voice-turn failures; cancellations are routine."""
    try:
        task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        logger.exception("Background voice turn failed")


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

    def print_user(self, text: str, duration: float | None = None) -> None:
        """Logs finalized recognized user speech transcript with timestamp and optional duration."""
        cleaned = (text or "").strip()
        if not cleaned:
            return
        if self._prefix_open or self._wrote_text:
            sys.stdout.write("\n")
            self._prefix_open = False
            self._wrote_text = False
        now_str = datetime.now().astimezone().strftime("%H:%M:%S")
        if duration is not None:
            sys.stdout.write(f"\r\033[K[{now_str}] you ({duration:.1f}s): {cleaned}\n")
        else:
            sys.stdout.write(f"\r\033[K[{now_str}] you: {cleaned}\n")
        sys.stdout.flush()

    def reset_delivered(self) -> None:
        self.in_thought = False
        self._delivered.clear()
        self._prefix_open = False
        self._wrote_text = False
def _build_voice_pipeline(
    runtime: Runtime,
    args: argparse.Namespace,
    printer: CliPrinter | None = None,
) -> tuple[Any, Any, Any]:
    from verbix.voice.callback_sink import select_sink
    from verbix.voice.capture import PTTCapture
    from verbix.voice.pipeline import VoicePipeline
    from verbix.voice.player import AudioPlayer
    from verbix.voice.sherpa_tts import SherpaTTS
    from verbix.voice.transcribe_stt import TranscribeSTT

    stt_model = load_config(runtime.workspace, args).get("stt_model")
    stt = TranscribeSTT(model_path=stt_model, workspace=runtime.workspace)
    tts = SherpaTTS()
    sink = select_sink(
        getattr(args, "audio_sink", "auto"),
        sample_rate=24000,
        audio_format="f32",
    )
    player = AudioPlayer(tts=tts, sink=sink)
    capture = PTTCapture()
    kwargs = {"printer": printer} if printer is not None else {}
    if getattr(args, "dump_mic_dir", None):
        kwargs["dump_mic_dir"] = args.dump_mic_dir
    pipeline = VoicePipeline(
        runtime=runtime,
        stt=stt,
        player=player,
        capture=capture,
        **kwargs,
    )
    return pipeline, tts, sink


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
        pipeline: Any = None
        tts: Any = None
        sink: Any = None
        if args.voice:
            pipeline, tts, sink = _build_voice_pipeline(runtime, args)

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
        printer = CliPrinter(show_thoughts=getattr(args, "show_thoughts", False))
        pipeline, tts, sink = _build_voice_pipeline(runtime, args, printer=printer)

        voice_mode = getattr(args, "voice_mode", "continuous")
        sys.stdout.write(
            f"verbix voice mode: {voice_mode} (model: {args.model}, workspace: {runtime.workspace})\n"
        )
        if voice_mode == "ptt":
            sys.stdout.write(
                "PTT Mode: Press [Enter] to start recording, speak, then press [Enter] to stop.\n"
                "If audio is playing, [Enter] halts playback and begins recording immediately.\n"
                "Or type a prompt directly. Type 'exit' or 'quit' to end session.\n\n"
            )
        else:
            sys.stdout.write(
                "Continuous Mode: Press [Enter] once to start hands-free session, then speak.\n"
                "Verbix replies and automatically listens for your next turn.\n"
                "Press [Enter] again to close session. Type 'exit' or 'quit' to end.\n\n"
            )
        sys.stdout.flush()

        loop = asyncio.get_running_loop()
        try:
            while True:
                now_str = datetime.now().astimezone().strftime("%H:%M:%S")
                sys.stdout.write(f"[{now_str}] you> ")
                sys.stdout.flush()
                await loop.run_in_executor(None, drain_stdin)
                line = await loop.run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                text = line.strip()
                if text.lower() in ("exit", "quit"):
                    break

                if not text:
                    if voice_mode == "continuous":
                        if pipeline.is_continuous_active:
                            await pipeline.stop_continuous_session()
                            sys.stdout.write("[Session closed. Mic idle.]\n")
                            sys.stdout.flush()
                        else:
                            was_playing = (
                                pipeline.turn_output.is_playing
                                or not pipeline.runtime.is_idle
                            )
                            await pipeline.start_continuous_session()
                            if was_playing:
                                sys.stdout.write("[Playback interrupted]\n")
                            sys.stdout.write(
                                "[Continuous session open... Speak freely. Press Enter to end session.]\n"
                            )
                            sys.stdout.flush()

                    else:
                        res = await pipeline.toggle()
                        state = res.get("action")
                        if state in ("recording", "interrupted_and_recording"):
                            if state == "interrupted_and_recording":
                                sys.stdout.write("[Playback interrupted]\n")
                            sys.stdout.write(
                                "[Recording mic... Press Enter when done speaking]\n"
                            )
                            sys.stdout.flush()
                            await loop.run_in_executor(None, drain_stdin)
                            await loop.run_in_executor(None, sys.stdin.readline)
                            sys.stdout.write("[Capture stopped. Earcon cue playing...]\n")
                            sys.stdout.flush()
                            stop_res = await pipeline.toggle()
                            transcribed = stop_res.get("transcription", "")
                            if not transcribed:
                                detail = stop_res.get("detail", "")
                                msg = "[No speech detected or audio too short]"
                                if detail:
                                    msg += f" ({detail})"
                                sys.stdout.write(msg + "\n")
                            sys.stdout.flush()
                            active_task = getattr(pipeline, "_active_voice_task", None)
                            if active_task:
                                active_task.add_done_callback(_note_turn_failed)
                else:
                    live = getattr(pipeline, "_active_voice_task", None)
                    if live is not None and not live.done():
                        snapshot = pipeline.turn_output.interrupt()
                        heard = snapshot.confirmed_heard if snapshot is not None else ""
                        await pipeline.runtime.cancel_turn(heard=heard)
                        sys.stdout.write("[Interrupted]\n")
                        sys.stdout.flush()
                    pipeline._active_voice_task = asyncio.create_task(
                        pipeline.execute_voice_turn(text)
                    )
                    pipeline._active_voice_task.add_done_callback(_note_turn_failed)
        except (KeyboardInterrupt, asyncio.CancelledError):
            sys.stdout.write("\n")
        finally:
            if getattr(pipeline, "is_continuous_active", False):
                await pipeline.stop_continuous_session()
            live = getattr(pipeline, "_active_voice_task", None)
            if live is not None and not live.done():
                live.cancel()
                try:
                    await asyncio.wait_for(live, timeout=5.0)
                except asyncio.CancelledError:
                    pass
                except TimeoutError:
                    logger.warning("Voice turn did not end after cancel; leaving it behind")
                except Exception:
                    logger.exception("Background voice turn failed")
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
    ensure_global_scaffold()
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
