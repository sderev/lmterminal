import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import openai
import pytest
from click.testing import CliRunner

from lmterminal import cli, cli_output


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["templates", "edit", "missing"], "lmt templates add missing"),
        (["templates", "delete", "missing"], "does not exist"),
        (["templates", "rename", "missing"], "does not exist"),
        (["key", "set"], "lmt key edit"),
    ],
)
@pytest.mark.parametrize("color", [False, True])
def test_management_failures_use_stderr(monkeypatch, arguments, message, color):
    monkeypatch.setattr(cli, "read_api_key", lambda _: "synthetic-key")
    result = CliRunner().invoke(cli.lmt, arguments, color=color)
    assert result.exit_code == 1
    assert result.stdout == ""
    assert message in result.stderr
    assert "synthetic-key" not in result.stderr
    if color:
        assert "\x1b[31mError:\x1b[0m" in result.stderr
        if arguments[0] == "templates":
            assert "\x1b[31mmissing\x1b[0m" in result.stderr
        if message.startswith("lmt "):
            assert f"\x1b[34m{message}\x1b[0m" in result.stderr
    else:
        assert "\x1b" not in result.stderr


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "status,body,message",
    [
        (404, {"message": "Model unavailable", "private": "PRIVATE"}, "Model unavailable"),
        (503, "<html>PRIVATE</html>", "OpenAI request failed (HTTP 503)."),
        (429, {"message": "Request limit reached", "private": "PRIVATE"}, "Request limit reached"),
        (401, {"message": "Invalid PRIVATE key"}, "lmt key edit"),
    ],
)
def test_provider_failures_preserve_partial_text_and_close_client(
    monkeypatch, stream, status, body, message
):
    error_types = {401: openai.AuthenticationError, 429: openai.RateLimitError}
    error = error_types.get(status, openai.APIStatusError)(
        "SDK formatted PRIVATE body",
        response=httpx.Response(status, request=httpx.Request("POST", "https://example.invalid")),
        body=body,
    )
    closed = []

    def create(**kwargs):
        if not kwargs["stream"]:
            raise error

        def events():
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))]
            )
            raise error

        return events()

    class Client:
        chat = SimpleNamespace(completions=SimpleNamespace(create=create))

        def __enter__(self):
            return self

        def __exit__(self, *args):
            closed.append(True)

    monkeypatch.setattr(openai, "OpenAI", lambda **_: Client())
    monkeypatch.setattr(cli_output, "read_api_key", lambda _: "synthetic-key")
    result = CliRunner().invoke(
        cli.lmt, ["--raw", *([] if stream else ["--no-stream"]), "fixture"], input=""
    )
    assert result.exit_code == 1
    assert result.stdout == ("partial" if stream else "")
    assert message in result.stderr
    assert "PRIVATE" not in result.stderr
    assert "\x1b" not in result.stderr
    assert closed == [True]


def test_connection_failure_uses_fixed_message(monkeypatch):
    def fail(**kwargs):
        raise openai.APIConnectionError(
            message="PRIVATE", request=httpx.Request("POST", "https://example.invalid")
        )

    monkeypatch.setattr(openai, "OpenAI", fail)
    monkeypatch.setattr(cli_output, "read_api_key", lambda _: "synthetic-key")
    result = CliRunner().invoke(cli.lmt, ["fixture"], input="")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "Could not connect to OpenAI" in result.stderr
    assert "PRIVATE" not in result.stderr


def test_early_closing_pipe_is_quiet():
    harness = Path(__file__).with_name("manual_cli.py")
    with subprocess.Popen(
        [sys.executable, "-I", str(harness), "--fixture", "pipe", "--raw", "fixture"],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        assert process.stdout.read(1) == b"x"
        process.stdout.close()
        stderr = process.stderr.read()
        assert process.wait(timeout=10) == 1
    assert stderr == b""
