from types import SimpleNamespace

import click
import pytest

from lmterminal import gpt_integration, lib


def _build_client(response):
    class FakeCompletions:
        def __init__(self, expected_response):
            self.expected_response = expected_response
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return self.expected_response

    completions = FakeCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_chatgpt_request_non_stream_uses_v2_client_call(monkeypatch):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(response)
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _api_key: client)

    prompt = [{"role": "user", "content": "ping"}]
    generated_text, response_time, response_payload = gpt_integration.chatgpt_request(
        api_key="test-key",
        prompt=prompt,
        model="gpt-4o",
        n=1,
        temperature=0.3,
        stream=False,
    )

    assert generated_text == "pong"
    assert isinstance(response_time, float)
    assert response_payload is response
    assert completions.calls == [
        {
            "messages": prompt,
            "model": "gpt-4o",
            "n": 1,
            "temperature": 0.3,
            "stream": False,
        }
    ]


def test_nonstream_tool_call_preserves_wrapper_text_and_payload(monkeypatch, capsys):
    tool_calls = [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))]
    )
    client, completions = _build_client(response)
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(lib, "get_markdown_code_block_theme", lambda: "monokai")
    monkeypatch.setattr(lib, "get_markdown_inline_code_theme", lambda: "blue on black")
    options = {
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "required",
    }

    content, elapsed, payload = lib.generate_response(
        model="gpt-4o", prompt=[], stream=False, request_options=options
    )

    assert content == "\n"
    assert isinstance(elapsed, float)
    assert capsys.readouterr().out == "\n"
    assert payload is response
    assert payload.choices[0].message.tool_calls is tool_calls
    assert completions.calls[0]["tools"] == options["tools"]
    assert completions.calls[0]["tool_choice"] == "required"


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5-pro-2025-10-06",
        "gpt-5.2-pro-2025-12-11",
        "gpt-5.4-pro-2026-03-05",
        "o3-pro-2025-06-10",
        "o1-pro-2025-03-19",
    ],
)
def test_responses_only_snapshots_fail_before_key_or_client(monkeypatch, model):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported snapshots must fail before key or client access")

    monkeypatch.setattr(lib, "get_api_key", forbidden)
    monkeypatch.setattr(gpt_integration, "_get_client", forbidden)
    with pytest.raises(click.BadParameter, match="not supported for Chat Completions"):
        lib.generate_response(model=model, prompt=[])
    with pytest.raises(ValueError, match="not supported for Chat Completions"):
        gpt_integration.chatgpt_request("test-key", [], model=model)


def test_unknown_snapshot_keeps_api_validation(monkeypatch):
    model = "gpt-5.4-pro-2099-01-01"
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(response)
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    assert gpt_integration.chatgpt_request("test-key", [], model=model)[0] == "pong"
    assert completions.calls[0]["model"] == model


def test_chatgpt_request_stream_returns_collected_chunks(monkeypatch):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
    ]
    client, completions = _build_client(iter(chunks))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _api_key: client)

    streamed_updates = []
    generated_text, response_time, response_payload = gpt_integration.chatgpt_request(
        api_key="test-key",
        prompt=[{"role": "user", "content": "hello"}],
        model="gpt-5-nano",
        stream=True,
        update_markdown_stream=streamed_updates.append,
    )

    assert generated_text == "Hello"
    assert isinstance(response_time, float)
    assert response_payload == chunks
    assert streamed_updates == ["Hel", "", "lo"]
    assert completions.calls[0]["stream"] is True


def test_chatgpt_request_stream_prints_with_flush_when_no_callback(monkeypatch):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
    ]
    client, _completions = _build_client(iter(chunks))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _api_key: client)

    printed_calls = []

    def fake_print(*args, **kwargs):
        printed_calls.append((args, kwargs))

    monkeypatch.setattr("builtins.print", fake_print)

    generated_text, _response_time, response_payload = gpt_integration.chatgpt_request(
        api_key="test-key",
        prompt=[{"role": "user", "content": "hello"}],
        model="gpt-5-nano",
        stream=True,
        update_markdown_stream=None,
    )

    assert generated_text == "Hello"
    assert response_payload == chunks
    assert printed_calls == [
        (("Hel",), {"end": "", "flush": True}),
        (("lo",), {"end": "", "flush": True}),
    ]


@pytest.mark.parametrize("stream", [False, True])
def test_explicit_controls_and_usage_event(monkeypatch, stream):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="pong"))]),
        SimpleNamespace(choices=[]),
    ]
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(iter(chunks) if stream else response)
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    options = {"verbosity": "low", "max_completion_tokens": 100}
    if stream:
        options["stream_options"] = {"include_usage": True}
    updates = []
    content, _, payload = gpt_integration.chatgpt_request(
        "test-key",
        [],
        "gpt-5.4",
        stream=stream,
        update_markdown_stream=updates.append,
        reasoning_effort="high",
        request_options=options,
    )
    assert content == "pong"
    assert payload == chunks if stream else payload is response
    assert updates == (["pong"] if stream else [])
    assert completions.calls == [
        dict(messages=[], model="gpt-5.4", n=1, stream=stream, reasoning_effort="high", **options)
    ]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "options",
    [
        {"stream": True},
        {"stream": False},
        {"messages": []},
        {"model": "gpt-4o"},
        {"n": 2},
        {"temperature": 0.3},
        {"reasoning_effort": "low"},
        {"stop": "end"},
        {"extra_body": {"stream": True}},
    ],
)
def test_request_options_cannot_override_owned_controls(monkeypatch, stream, options):
    monkeypatch.setattr(
        gpt_integration, "_get_client", lambda _: pytest.fail("No request expected")
    )
    with pytest.raises(ValueError, match="reserved"):
        gpt_integration.chatgpt_request("test-key", [], stream=stream, request_options=options)


def test_old_positional_callback_contract(monkeypatch):
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])
    client, _ = _build_client(iter([chunk]))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    updates = []
    assert (
        gpt_integration.chatgpt_request(
            "test-key",
            [],
            "gpt-4o",
            1,
            1,
            None,
            True,
            updates.append,
        )[0]
        == "hello"
    )
    assert updates == ["hello"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "model, effort, temperature, expected",
    [
        ("gpt-5-nano", None, 1, None),
        ("gpt-5", "minimal", 1, None),
        ("gpt-5.1", None, 0.3, 0.3),
        ("gpt-5.2", "none", 0.3, 0.3),
        ("gpt-5.4", "high", 1, None),
        ("gpt-5.4-mini", "none", 0.3, 0.3),
        ("o3-2025-04-16", "low", 1, None),
        ("gpt-4o", None, 0.3, 0.3),
    ],
)
def test_sampling_and_reasoning_controls(monkeypatch, stream, model, effort, temperature, expected):
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    client, completions = _build_client(iter([chunk]) if stream else response)
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    gpt_integration.chatgpt_request(
        "test-key",
        [],
        model,
        temperature=temperature,
        stream=stream,
        reasoning_effort=effort,
        update_markdown_stream=lambda _: None,
    )
    call = completions.calls[0]
    assert call.get("temperature") == expected
    if effort is None:
        assert "reasoning_effort" not in call
    else:
        assert call["reasoning_effort"] == effort


@pytest.mark.parametrize(
    "model, effort, temperature, options",
    [
        ("gpt-5-nano", None, 0.3, {}),
        ("gpt-5.4", "high", 0.3, {}),
        ("gpt-5.4", "high", 1, {"top_p": 0.5}),
        ("o3", None, 1, {"extra_body": {"logprobs": True}}),
        ("gpt-5.4-pro", None, 1, {}),
    ],
)
def test_unsupported_controls_fail_before_client(monkeypatch, model, effort, temperature, options):
    monkeypatch.setattr(
        gpt_integration, "_get_client", lambda _: pytest.fail("No request expected")
    )
    with pytest.raises(ValueError):
        gpt_integration.chatgpt_request(
            "test-key",
            [],
            model,
            temperature=temperature,
            reasoning_effort=effort,
            request_options=options,
        )
