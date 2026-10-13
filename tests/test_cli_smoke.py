
import pytest

from verbix.cli import CliPrinter, build_parser, build_runtime
from verbix.core.events import TextDelta, ToolCallStarted, TurnComplete
from verbix.runtime.runtime import Runtime


def test_build_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    parser = build_parser()
    args = parser.parse_args(["--workspace", str(tmp_path)])
    runtime = build_runtime(args)
    assert isinstance(runtime, Runtime)
    assert runtime.workspace == tmp_path.resolve()


@pytest.mark.asyncio
async def test_cli_printer_renders_standalone_turn_complete(capsys):
    printer = CliPrinter()
    # 1. First turn: streaming text followed by tool call
    await printer.emit(TextDelta(text="I am checking the current time for you."))
    await printer.emit(
        ToolCallStarted(
            call_id="call_1",
            tool_name="spawn",
            arguments={"objective": "time"},
        )
    )
    await printer.emit(
        TurnComplete(final_text="I am checking the current time for you.")
    )

    import re
    captured = capsys.readouterr()
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] verbix: I am checking the current time for you\.", captured.out)

    # 2. Standalone completion: worker finishes and flushes announcement
    await printer.emit(TurnComplete(final_text="The current time is 12:45 AM."))
    captured_announcement = capsys.readouterr()
    assert re.search(
        r"\[\d{2}:\d{2}:\d{2}\] verbix: The current time is 12:45 AM\.",
        captured_announcement.out,
    )


def test_drain_stdin_discards_stale_enters(monkeypatch):
    import io
    import os
    import select

    from verbix.cli import drain_stdin

    r, w = os.pipe()
    try:
        os.write(w, b"\n\n")
        monkeypatch.setattr("sys.stdin", io.TextIOWrapper(os.fdopen(os.dup(r), "rb")))
        drain_stdin()
        assert select.select([r], [], [], 0)[0] == []
    finally:
        os.close(r)
        os.close(w)


def test_drain_stdin_never_raises_on_non_selectable(monkeypatch, capsys):
    import io

    from verbix.cli import drain_stdin

    monkeypatch.setattr("sys.stdin", io.StringIO("\n"))
    drain_stdin()  # must not raise; StringIO has no fileno for select
    capsys.readouterr()


def test_cli_printer_renders_user_transcript(capsys):
    import re

    printer = CliPrinter()
    printer.print_user("repeat the story")
    out = capsys.readouterr().out
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] you: repeat the story\n", out)

    # Empty transcripts are ignored
    printer.print_user("")
    printer.print_user("   \n\t")
    assert capsys.readouterr().out == ""


def test_build_runtime_worker_max_tokens_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    parser = build_parser()
    args = parser.parse_args(["--workspace", str(tmp_path), "--worker-max-tokens", "8192"])
    runtime = build_runtime(args)
    assert runtime.worker_model is not None
    assert runtime.worker_model.config.model_kwargs.get("max_tokens") == 8192
    # Foreground model retains default
    assert runtime.provider.config.model_kwargs.get("max_tokens") == 2048


def test_build_runtime_worker_max_tokens_default_uncapped(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.delenv("VERBIX_WORKER_MAX_TOKENS", raising=False)
    parser = build_parser()
    args = parser.parse_args(["--workspace", str(tmp_path)])
    runtime = build_runtime(args)
    assert runtime.worker_model is not None
    assert "max_tokens" not in runtime.worker_model.config.model_kwargs


def test_build_runtime_worker_max_tokens_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    monkeypatch.setenv("VERBIX_WORKER_MAX_TOKENS", "4096")
    parser = build_parser()
    args = parser.parse_args(["--workspace", str(tmp_path)])
    runtime = build_runtime(args)
    assert runtime.worker_model is not None
    assert runtime.worker_model.config.model_kwargs.get("max_tokens") == 4096

