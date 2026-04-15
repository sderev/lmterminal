import pytest

from lmterminal import lib


class DummyLive:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.args = _args
        self.kwargs = _kwargs
        self.updates = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args, **_kwargs):
        return False

    def update(self, *_args, **_kwargs):
        self.updates.append((_args, _kwargs))
        return None


class DummyConsole:
    def __init__(self, *, is_terminal=True, is_dumb_terminal=False):
        self.is_terminal = is_terminal
        self.is_dumb_terminal = is_dumb_terminal


class DummyRateLimitError(Exception):
    pass


class DummyAuthenticationError(Exception):
    pass


class DummyAPIConnectionError(Exception):
    pass


def _prepare_generate_response(monkeypatch, *, is_terminal=True, is_dumb_terminal=False):
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(lib, "get_markdown_code_block_theme", lambda: "monokai")
    monkeypatch.setattr(lib, "get_markdown_inline_code_theme", lambda: "blue on black")
    DummyLive.instances.clear()
    monkeypatch.setattr(lib, "Live", DummyLive)
    monkeypatch.setattr(
        lib,
        "Console",
        lambda **_kwargs: DummyConsole(
            is_terminal=is_terminal,
            is_dumb_terminal=is_dumb_terminal,
        ),
    )


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


def test_generate_response_raw_stream_prints_with_flush(monkeypatch):
    _prepare_generate_response(monkeypatch)
    monkeypatch.setattr(lib.openai, "RateLimitError", DummyRateLimitError)
    monkeypatch.setattr(lib.openai, "AuthenticationError", DummyAuthenticationError)
    monkeypatch.setattr(lib.openai, "APIConnectionError", DummyAPIConnectionError)

    printed_calls = []

    def fake_print(*args, **kwargs):
        printed_calls.append((args, kwargs))

    def fake_chatgpt_request(**kwargs):
        assert kwargs["update_markdown_stream"] is None
        print("Hel", end="", flush=True)
        print("lo", end="", flush=True)
        return "Hello", 0.01, object()

    monkeypatch.setattr(lib.openai_utils, "chatgpt_request", fake_chatgpt_request)
    monkeypatch.setattr("builtins.print", fake_print)

    content, _response_time, _response = lib.generate_response(
        prompt=[{"role": "user", "content": "hello"}],
        raw=True,
        stream=True,
    )

    assert content == "Hello\n"
    assert printed_calls == [
        (("Hel",), {"end": "", "flush": True}),
        (("lo",), {"end": "", "flush": True}),
    ]


def test_generate_response_live_stream_refreshes_each_chunk(monkeypatch):
    _prepare_generate_response(monkeypatch)
    monkeypatch.setattr(lib.openai, "RateLimitError", DummyRateLimitError)
    monkeypatch.setattr(lib.openai, "AuthenticationError", DummyAuthenticationError)
    monkeypatch.setattr(lib.openai, "APIConnectionError", DummyAPIConnectionError)

    def fake_chatgpt_request(**kwargs):
        kwargs["update_markdown_stream"]("Hel")
        kwargs["update_markdown_stream"]("lo")
        return "Hello", 0.01, object()

    monkeypatch.setattr(lib.openai_utils, "chatgpt_request", fake_chatgpt_request)

    content, _response_time, _response = lib.generate_response(
        prompt=[{"role": "user", "content": "hello"}],
        raw=False,
        stream=True,
    )

    assert content == "Hello\n"
    assert len(DummyLive.instances) == 1
    live = DummyLive.instances[0]
    assert live.kwargs["auto_refresh"] is False
    assert [kwargs for _args, kwargs in live.updates] == [
        {"refresh": True},
        {"refresh": True},
    ]


def test_generate_response_dumb_terminal_stream_falls_back_to_plain_output(monkeypatch):
    _prepare_generate_response(monkeypatch, is_dumb_terminal=True)
    monkeypatch.setattr(lib.openai, "RateLimitError", DummyRateLimitError)
    monkeypatch.setattr(lib.openai, "AuthenticationError", DummyAuthenticationError)
    monkeypatch.setattr(lib.openai, "APIConnectionError", DummyAPIConnectionError)

    printed_calls = []

    def fake_print(*args, **kwargs):
        printed_calls.append((args, kwargs))

    def fake_chatgpt_request(**kwargs):
        assert kwargs["update_markdown_stream"] is None
        print("Hel", end="", flush=True)
        print("lo", end="", flush=True)
        return "Hello", 0.01, object()

    monkeypatch.setattr("builtins.print", fake_print)
    monkeypatch.setattr(lib.openai_utils, "chatgpt_request", fake_chatgpt_request)

    content, _response_time, _response = lib.generate_response(
        prompt=[{"role": "user", "content": "hello"}],
        raw=False,
        stream=True,
    )

    assert content == "Hello\n"
    assert DummyLive.instances == []
    assert printed_calls == [
        (("Hel",), {"end": "", "flush": True}),
        (("lo",), {"end": "", "flush": True}),
    ]


def test_theme_getters_fall_back_without_overwriting_invalid_config(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    invalid_config = '{\n  "code_block_theme": "dracula",\n'
    config_path.write_text(invalid_config, encoding="UTF-8")
    monkeypatch.setattr(lib, "get_config_path", lambda: config_path)

    assert lib.get_markdown_code_block_theme() == lib.DEFAULT_CODE_BLOCK_THEME
    assert lib.get_markdown_inline_code_theme() == lib.DEFAULT_INLINE_CODE_THEME
    assert config_path.read_text(encoding="UTF-8") == invalid_config


def test_theme_getters_do_not_create_config_file_when_missing(monkeypatch, tmp_path):
    config_path = tmp_path / "config.json"
    monkeypatch.setattr(lib, "get_config_path", lambda: config_path)

    assert lib.get_markdown_code_block_theme() == lib.DEFAULT_CODE_BLOCK_THEME
    assert lib.get_markdown_inline_code_theme() == lib.DEFAULT_INLINE_CODE_THEME
    assert not config_path.exists()
