import io
import sys
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from pygments.token import Keyword, Name
from rich.syntax import PygmentsSyntaxTheme

from lmterminal import cli, lib
from lmterminal.code_themes import AlabasterStyle, resolve_code_theme
from lmterminal.estimation import InputEstimate


class TerminalOutput(io.TextIOWrapper):
    def __init__(self, is_terminal):
        super().__init__(io.BytesIO(), encoding="UTF-8")
        self._is_terminal = is_terminal

    def isatty(self):
        return self._is_terminal


def _use_terminal(
    monkeypatch, *, is_terminal=True, term="xterm", tty_interactive=None, color_system=None
):
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
            color_system=color_system,
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


def test_alabaster_resolution_preserves_palette_and_installed_styles():
    assert AlabasterStyle.background_color == "#f0f0f0"
    assert AlabasterStyle.style_for_token(Keyword)["color"] == "7a3e9d"
    assert AlabasterStyle.style_for_token(Name.Function)["color"] == "325cc0"
    theme = resolve_code_theme("alabaster")
    assert isinstance(theme, PygmentsSyntaxTheme)
    assert theme.get_style_for_token(Keyword).color.triplet == (122, 62, 157)
    assert resolve_code_theme("monokai").get_style_for_token(Keyword).color is not None
    assert resolve_code_theme("ansi_dark").get_style_for_token(Keyword).color is not None


def test_generate_response_renders_configured_alabaster_and_inline_colors(monkeypatch):
    _prepare_generate_response(monkeypatch)
    monkeypatch.setattr(lib, "get_markdown_code_block_theme", lambda: "alabaster")
    monkeypatch.setattr(lib, "get_markdown_inline_code_theme", lambda: "#325cc0 on #f0f0f0")
    output = _use_terminal(monkeypatch, color_system="truecolor")
    text = 'Inline `fib`.\n\n```python\ndef fib():\n    return "hello"\n```\n'

    def send(**kwargs):
        kwargs["update_markdown_stream"](text)
        return text, 0, None

    monkeypatch.setattr(lib.openai_utils, "send_prepared_request", send)
    lib.generate_response(prompt=[{"role": "user", "content": "fixture"}])
    rendered = output.buffer.getvalue().decode("UTF-8")
    assert "38;2;122;62;157;48;2;240;240;240mdef" in rendered  # Purple code on grey.
    assert "\x1b[48;2;240;240;240m" + " " * 80 + "\x1b[0m" in rendered  # Block padding.
    assert "Inline \x1b[38;2;50;92;192;48;2;240;240;240mfib\x1b[0m." in rendered


def test_unknown_code_theme_fails_before_credentials_or_provider(monkeypatch):
    _use_terminal(monkeypatch)
    monkeypatch.setattr(lib, "get_markdown_code_block_theme", lambda: "missing-lmt-style")

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid formatted theme must fail before credentials or provider work")

    monkeypatch.setattr(lib, "get_api_key", forbidden)
    monkeypatch.setattr(lib.openai_utils, "send_prepared_request", forbidden)
    with pytest.raises(click.ClickException, match="missing-lmt-style.*unavailable") as error:
        lib.generate_response(prompt=[{"role": "user", "content": "fixture"}])
    assert "code_block_theme" in error.value.format_message()
    assert "--raw" in error.value.format_message()


@pytest.mark.parametrize(
    "raw,stream,is_terminal", [(True, True, True), (False, False, True), (False, True, False)]
)
def test_plain_response_skips_theme_validation(monkeypatch, raw, stream, is_terminal):
    _prepare_generate_response(monkeypatch)
    _use_terminal(monkeypatch, is_terminal=is_terminal)

    def forbidden(*args, **kwargs):
        pytest.fail("Plain responses must not load or validate formatting preferences")

    monkeypatch.setattr(lib, "get_markdown_code_block_theme", forbidden)
    monkeypatch.setattr(lib, "get_markdown_inline_code_theme", forbidden)

    def send(**kwargs):
        assert kwargs["update_markdown_stream"] is None
        return "hello", 0, None

    monkeypatch.setattr(lib.openai_utils, "send_prepared_request", send)
    assert (
        lib.generate_response(
            prompt=[{"role": "user", "content": "fixture"}], raw=raw, stream=stream
        )[0]
        == "hello\n"
    )


def test_token_estimate_skips_response_theme_validation(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Token estimates must not load themes or credentials")

    monkeypatch.setattr(lib, "get_markdown_code_block_theme", forbidden)
    monkeypatch.setattr(lib, "get_api_key", forbidden)
    monkeypatch.setattr(lib, "estimate_request", lambda _: InputEstimate(model="gpt-5-nano"))
    result = CliRunner().invoke(cli.lmt, ["--tokens"], input="fixture")
    assert result.exit_code == 1  # Explicit unavailable estimate, independent of themes.
    assert "Request input tokens and cost: unavailable" in result.output


def test_generate_response_handles_rate_limit_error(monkeypatch):
    _prepare_generate_response(monkeypatch)
    monkeypatch.setattr(lib.openai, "RateLimitError", DummyRateLimitError)

    called = {"value": False}

    def fake_chatgpt_request(**_kwargs):
        raise DummyRateLimitError("quota exceeded")

    def fake_rate_limit_handler():
        called["value"] = True

    monkeypatch.setattr(lib.openai_utils, "send_prepared_request", fake_chatgpt_request)
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

    monkeypatch.setattr(lib.openai_utils, "send_prepared_request", fake_chatgpt_request)
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

    monkeypatch.setattr(lib.openai_utils, "send_prepared_request", fake_chatgpt_request)

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


@pytest.mark.parametrize(
    "config_bytes, expected_config, expected_themes",
    [
        pytest.param(b'{"code_block_theme":', {}, ("monokai", "blue on black"), id="invalid-json"),
        pytest.param(b'["default"]\n', {}, ("monokai", "blue on black"), id="non-object-json"),
        pytest.param(
            b'{"code_block_theme": "default", "inline_code_theme": "cyan", '
            b'"unrelated": {"keep": true}}\n',
            {
                "code_block_theme": "default",
                "inline_code_theme": "cyan",
                "unrelated": {"keep": True},
            },
            ("default", "cyan"),
            id="custom-themes-and-unrelated-keys",
        ),
    ],
)
def test_theme_reads_preserve_config_bytes(
    monkeypatch, tmp_path, config_bytes, expected_config, expected_themes
):
    config_path = tmp_path / "config.json"
    config_path.write_bytes(config_bytes)
    monkeypatch.setattr(lib, "get_config_path", lambda: config_path)

    assert lib.load_config() == expected_config
    assert (
        lib.get_markdown_code_block_theme(),
        lib.get_markdown_inline_code_theme(),
    ) == expected_themes
    assert config_path.read_bytes() == config_bytes


def test_theme_reads_do_not_create_missing_config(monkeypatch, tmp_path):
    config_path = tmp_path / "missing" / "config.json"
    monkeypatch.setattr(lib, "get_config_path", lambda: config_path)

    assert lib.load_config() == {}
    assert lib.get_markdown_code_block_theme() == "monokai"
    assert lib.get_markdown_inline_code_theme() == "blue on black"
    assert not config_path.exists()
    assert not config_path.parent.exists()


def test_prepare_response_composes_messages_and_forwards_controls(monkeypatch):
    calls = []
    payload = object()

    def fake_generate(request, raw, stream, *, diagnostics=None):
        calls.append((request.model, request.messages, raw, stream, request.controls))
        return "hello\n", 0.1, payload

    monkeypatch.setattr(lib, "_generate_prepared_response", fake_generate)
    result = lib.prepare_and_generate_response(
        system="Reply concisely.",
        template=None,
        model="gpt-5.4",
        emoji=False,
        prompt_input="Say hello.",
        temperature=0.3,
        tokens=False,
        no_stream=True,
        raw=True,
        debug=False,
        reasoning_effort="none",
        request_options={"verbosity": "low", "max_completion_tokens": 100},
    )

    assert result == ("hello\n", 0.1, payload)
    assert calls == [
        (
            "gpt-5.4",
            [
                {"role": "system", "content": "Reply concisely."},
                {"role": "user", "content": "Say hello."},
            ],
            True,
            False,
            {
                "temperature": 0.3,
                "reasoning_effort": "none",
                "verbosity": "low",
                "max_completion_tokens": 100,
            },
        )
    ]
