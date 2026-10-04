import io
import sys
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from lmterminal import cli, gpt_integration, lib
from lmterminal.diagnostics import RequestDiagnostics


def chunk(text):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
def test_verbose_cli_is_metadata_only_and_preserves_stdout(monkeypatch, tmp_path, verbosity):
    monkeypatch.setattr(lib, "get_api_key", lambda: "sk-private-key")
    monkeypatch.setattr(lib, "get_config_path", lambda: tmp_path / "missing.json")
    monkeypatch.setattr(
        lib,
        "handle_template",
        lambda *args: ("private-system", "private-template-prompt", "4o"),
    )
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return iter(
            [
                chunk("private-response"),
                SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(
                        prompt_tokens=12,
                        completion_tokens=3,
                        total_tokens=15,
                        private_data="private-usage",
                    ),
                ),
            ]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    args = ["--template", "synthetic", "-o", "metadata.secret=private-option"]
    if verbosity:
        args.append("-" + "v" * verbosity)
    result = CliRunner().invoke(cli.lmt, args, input="private-prompt")

    assert result.exit_code == 0, result.output
    assert result.stdout == "private-response"
    assert calls[0]["model"] == "gpt-4o"
    assert calls[0]["stream"] is True
    assert calls[0]["metadata"] == {"secret": "private-option"}
    for secret in (
        "sk-private",
        "private-system",
        "private-template",
        "private-prompt",
        "private-response",
        "private-option",
        "private-usage",
    ):
        assert secret not in result.stderr
    if not verbosity:
        assert result.stderr == ""
        return
    assert "model='gpt-4o' stream=true route=/chat/completions" in result.stderr
    assert "first text received" in result.stderr
    assert "first text flushed" in result.stderr
    assert "request_ttft_s=" in result.stderr
    assert ("first stream event received" in result.stderr) == (verbosity >= 2)
    assert ("prompt handling started" in result.stderr) == (verbosity >= 2)
    assert (
        "events=2 text_chunks=1 prompt_tokens=12 completion_tokens=3 total_tokens=15"
        in result.stderr
    ) == (verbosity >= 3)


@pytest.mark.parametrize("markdown", [False, True])
def test_stream_timings_distinguish_empty_events_text_and_output(monkeypatch, markdown):
    from lib_test import _prepare_generate_response, _use_terminal

    _prepare_generate_response(monkeypatch)
    output = _use_terminal(monkeypatch, color_system="truecolor")
    first_text = "hello"
    if markdown:
        monkeypatch.setattr(lib, "get_markdown_code_block_theme", lambda: "alabaster")
        monkeypatch.setattr(lib, "get_markdown_inline_code_theme", lambda: "#325cc0 on #f0f0f0")
        first_text = 'hello `fib`.\n\n```python\ndef fib():\n    return "hello"\n```\n'
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)
    now = [100.0]
    diagnostics = RequestDiagnostics(3, clock=lambda: now[0])
    chunks = [SimpleNamespace(choices=[]), chunk(""), chunk(first_text), chunk(" world")]
    chunks[0].model = "gpt-5-nano-synthetic"

    def events():
        now[0] = 103.0
        yield chunks[0]
        assert "first stream event received" in stderr.getvalue()
        assert "first text received" not in stderr.getvalue()
        now[0] = 104.0
        yield chunks[1]
        assert "first text received" not in stderr.getvalue()
        now[0] = 107.0
        yield chunks[2]
        # Both diagnostics and bytes must be available before the next event.
        rendered = output.buffer.getvalue().decode()
        assert "hello" in rendered
        if markdown:
            assert "38;2;122;62;157" in rendered
            assert "38;2;50;92;192;48;2;240;240;240" in rendered
        assert "[lmt" not in rendered
        assert first_text not in stderr.getvalue()
        assert "[lmt +7.000s] first text received" in stderr.getvalue()
        assert "request complete" not in stderr.getvalue()
        now[0] = 109.0
        yield chunks[3]
        now[0] = 110.0

    def create(**kwargs):
        now[0] = 102.0
        return events()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    result = lib.generate_response(prompt=[], raw=not markdown, diagnostics=diagnostics)

    assert result[0] == first_text + " world\n"
    assert result[2] == chunks
    assert "[lmt" not in output.buffer.getvalue().decode()
    trace = stderr.getvalue()
    assert "[lmt +2.000s] stream ready" in trace
    assert "first stream event received model='gpt-5-nano-synthetic'" in trace
    assert "request_s=10.000 request_ttft_s=7.000" in trace
    if markdown:
        assert "first_refresh_s=7.000" in trace
        assert "first text flushed" not in trace
        assert trace.index("first text submitted to Markdown") < trace.index(
            "first Markdown refresh returned"
        )
    else:
        assert "first_flush_s=7.000" in trace
        assert trace.index("first text submitted to output") < trace.index("first text flushed")
    assert trace.count("first Markdown refresh returned" if markdown else "first text flushed") == 1
    assert "events=4 text_chunks=2 usage=unavailable" in trace
    assert trace.index("request complete") < trace.index("output complete")


@pytest.mark.parametrize("text", ["hello", None])
def test_nonstream_timings_report_output_after_response(monkeypatch, tmp_path, text):
    monkeypatch.setattr(lib, "get_api_key", lambda: "synthetic-key")
    monkeypatch.setattr(lib, "get_config_path", lambda: tmp_path / "missing.json")
    now = [0.0]
    monkeypatch.setattr("lmterminal.diagnostics.time.monotonic", lambda: now[0])

    def create(**kwargs):
        assert kwargs["stream"] is False
        now[0] = 5.0
        return SimpleNamespace(
            model="gpt-5-nano-synthetic",
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    result = CliRunner().invoke(cli.lmt, ["--verbose", "-vv", "--no-stream"], input="hi")
    assert result.exit_code == 0, result.output
    assert result.stdout == ("hello\n" if text else "\n")
    assert "stream=false" in result.stderr
    assert "response received model='gpt-5-nano-synthetic'" in result.stderr
    assert "stream ready" not in result.stderr
    assert "first stream event" not in result.stderr
    if text:
        assert "request_s=5.000 request_ttft_s=5.000 first_flush_s=5.000" in result.stderr
        assert result.stderr.index("request complete") < result.stderr.index("first text flushed")
    else:
        assert "request_ttft_s=unavailable first_output_s=unavailable" in result.stderr
        assert "first text received" not in result.stderr
        assert "first text flushed" not in result.stderr
