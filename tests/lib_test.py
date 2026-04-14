import pytest

from lmterminal import lib


class DummyLive:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args, **_kwargs):
        return False

    def update(self, *_args, **_kwargs):
        return None


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
    monkeypatch.setattr(lib, "Live", DummyLive)


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

    def fake_chatgpt_request(**kwargs):
        kwargs["update_markdown_stream"]("Hel")
        kwargs["update_markdown_stream"]("lo")
        return "Hello", 0.01, object()

    monkeypatch.setattr(lib.openai_utils, "chatgpt_request", fake_chatgpt_request)

    printed_calls = []

    def fake_print(*args, **kwargs):
        printed_calls.append((args, kwargs))

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
