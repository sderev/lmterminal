import io
import sys
from types import SimpleNamespace

import pytest

from lmterminal import lib


class TerminalOutput(io.TextIOWrapper):
    def __init__(self, is_terminal):
        super().__init__(io.BytesIO(), encoding="UTF-8")
        self._is_terminal = is_terminal

    def isatty(self):
        return self._is_terminal


def _use_terminal(monkeypatch, *, is_terminal=True, term="xterm", tty_interactive=None):
    output = TerminalOutput(is_terminal)
    monkeypatch.setattr(sys, "stdout", output)
    terminal_env = {"TERM": term}
    if tty_interactive is not None:
        terminal_env["TTY_INTERACTIVE"] = tty_interactive
    console_type = lib.Console
    monkeypatch.setattr(
        lib,
        "Console",
        lambda **kwargs: console_type(
            force_terminal=is_terminal,
            _environ=terminal_env,
            color_system=None,
            width=80,
            **kwargs,
        ),
    )
    return output


class DummyRateLimitError(Exception):
    pass


class DummyAuthenticationError(Exception):
    pass


class DummyAPIConnectionError(Exception):
    pass


def _prepare_generate_response(monkeypatch):
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(lib, "get_markdown_code_block_theme", lambda: "monokai")
    monkeypatch.setattr(lib, "get_markdown_inline_code_theme", lambda: "blue on black")


def test_generate_response_handles_rate_limit_error(monkeypatch):
    _prepare_generate_response(monkeypatch)
    monkeypatch.setattr(lib.openai, "RateLimitError", DummyRateLimitError)

    called = {"value": False}

    def fake_chatgpt_request(**_kwargs):
        raise DummyRateLimitError("quota exceeded")

    def fake_rate_limit_handler():
        called["value"] = True

    monkeypatch.setattr(lib.openai_utils, "chatgpt_request", fake_chatgpt_request)
    monkeypatch.setattr(lib.openai_utils, "handle_rate_limit_error", fake_rate_limit_handler)

    with pytest.raises(SystemExit) as exc_info:
        lib.generate_response(prompt=[{"role": "user", "content": "hello"}])

    assert exc_info.value.code == 1
    assert called["value"] is True


def test_generate_response_handles_authentication_error(monkeypatch):
    _prepare_generate_response(monkeypatch)
    monkeypatch.setattr(lib.openai, "AuthenticationError", DummyAuthenticationError)

    called = {"value": False}

    def fake_chatgpt_request(**_kwargs):
        raise DummyAuthenticationError("invalid key")

    def fake_auth_handler():
        called["value"] = True

    monkeypatch.setattr(lib.openai_utils, "chatgpt_request", fake_chatgpt_request)
    monkeypatch.setattr(lib.openai_utils, "handle_authentication_error", fake_auth_handler)

    with pytest.raises(SystemExit) as exc_info:
        lib.generate_response(prompt=[{"role": "user", "content": "hello"}])

    assert exc_info.value.code == 1
    assert called["value"] is True


def test_generate_response_handles_api_connection_error(monkeypatch, capsys):
    _prepare_generate_response(monkeypatch)
    monkeypatch.setattr(lib.openai, "APIConnectionError", DummyAPIConnectionError)

    def fake_chatgpt_request(**_kwargs):
        raise DummyAPIConnectionError("network down")

    monkeypatch.setattr(lib.openai_utils, "chatgpt_request", fake_chatgpt_request)

    with pytest.raises(SystemExit) as exc_info:
        lib.generate_response(prompt=[{"role": "user", "content": "hello"}])

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "network down" in captured.err


@pytest.mark.parametrize(
    "raw, is_terminal, term, tty_interactive",
    [
        pytest.param(True, True, "xterm", None, id="raw-terminal"),
        pytest.param(False, True, "xterm", None, id="formatted-terminal"),
        pytest.param(False, True, "dumb", None, id="dumb-terminal"),
        pytest.param(False, False, "xterm", None, id="pipe"),
        pytest.param(False, True, "xterm", "0", id="noninteractive-terminal"),
    ],
)
def test_generate_response_stream_is_visible_before_completion(
    monkeypatch, raw, is_terminal, term, tty_interactive
):
    _prepare_generate_response(monkeypatch)
    output = _use_terminal(
        monkeypatch, is_terminal=is_terminal, term=term, tty_interactive=tty_interactive
    )
    plain = raw or not is_terminal or term == "dumb" or tty_interactive == "0"
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
        for text in ("**Hel", None, "lo**")
    ]
    checkpoints = []

    def gated_chunks():
        yield chunks[0]
        # Read the bytes beneath TextIOWrapper without flushing it. Both the real
        # request loop and Rich renderer must deliver text before asking for more.
        first_output = output.buffer.getvalue().decode("UTF-8")
        assert "Hel" in first_output
        if plain:
            assert first_output == "**Hel"
        checkpoints.append(first_output)
        yield from chunks[1:]

    def fake_create(**kwargs):
        assert kwargs["stream"] is True
        return gated_chunks()

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(lib.openai_utils, "_get_client", lambda _api_key: client)

    content, response_time, response = lib.generate_response(
        prompt=[{"role": "user", "content": "hello"}],
        raw=raw,
        stream=True,
    )

    assert len(checkpoints) == 1
    assert content == "**Hello**\n"
    assert isinstance(response_time, float)
    assert response == chunks
    final_output = output.buffer.getvalue().decode("UTF-8")
    if plain:
        assert final_output == "**Hello**"
    else:
        assert "Hello" in final_output
        assert "**Hello**" not in final_output


def test_generate_response_non_stream_preserves_plain_text_and_response(monkeypatch):
    _prepare_generate_response(monkeypatch)
    output = _use_terminal(monkeypatch)
    text = "**Hello** [world]\n"
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

    def fake_create(**kwargs):
        assert kwargs["stream"] is False
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(lib.openai_utils, "_get_client", lambda _api_key: client)

    content, response_time, response_payload = lib.generate_response(
        prompt=[{"role": "user", "content": "hello"}], raw=False, stream=False
    )
    output.flush()

    assert text in output.buffer.getvalue().decode("UTF-8")
    assert content == text
    assert isinstance(response_time, float)
    assert response_payload is response
