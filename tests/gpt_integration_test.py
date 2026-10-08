from types import SimpleNamespace

import pytest

from lmterminal import gpt_integration
from lmterminal.request_options import prepare_request


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


@pytest.mark.parametrize(
    "options, model, controls",
    [
        ({}, "gpt-6-luna", {"reasoning_effort": "none", "temperature": 1}),
        ({"model": "luna"}, "gpt-6-luna", {}),
        ({"model": "sol"}, "gpt-6.1-sol", {}),
        ({"reasoning_effort": "high"}, "gpt-6-luna", {"reasoning_effort": "high"}),
        ({"reasoning_effort": None}, "gpt-6-luna", {}),
        ({"temperature": None}, "gpt-6-luna", {"reasoning_effort": "none"}),
        ({"model": "gpt-4o", "temperature": None}, "gpt-4o", {}),
        ({"model": "gpt-5.4", "temperature": None}, "gpt-5.4", {}),
        ({"model": "5.6"}, "gpt-5.6-sol", {}),
        ({"model": "gpt-4o"}, "gpt-4o", {"temperature": 1}),
    ],
)
def test_preparation_and_transport_preserve_implicit_defaults_and_explicit_omission(
    options, model, controls
):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(response)
    request = prepare_request(messages=[], **options)
    gpt_integration.send_prepared_request(client, request)
    assert completions.calls == [dict(messages=[], model=model, n=1, stream=False, **controls)]


@pytest.mark.parametrize(
    "model, canonical, temperature, controls",
    [("gpt-4o", "gpt-4o", 0.3, {"temperature": 0.3}), ("sol", "gpt-6.1-sol", None, {})],
)
def test_transport_non_stream_uses_v2_client_call(model, canonical, temperature, controls):
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(response)
    prompt = [{"role": "user", "content": "ping"}]
    generated_text, response_time, response_payload = gpt_integration.send_prepared_request(
        client, prepare_request(model, prompt, temperature=temperature), stream=False
    )
    assert generated_text == "pong"
    assert isinstance(response_time, float)
    assert response_payload is response
    assert completions.calls == [
        {"messages": prompt, "model": canonical, "n": 1, "stream": False, **controls}
    ]


def test_nonstream_tool_call_returns_empty_text_and_raw_payload(capsys):
    tool_calls = [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=tool_calls))]
    )
    client, completions = _build_client(response)
    options = {
        "tools": [{"type": "function", "function": {"name": "lookup"}}],
        "tool_choice": "required",
    }
    content, elapsed, payload = gpt_integration.send_prepared_request(
        client, prepare_request("gpt-4o", [], request_options=options), stream=False
    )
    assert content == ""
    assert isinstance(elapsed, float)
    assert capsys.readouterr().out == ""
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
        "gpt-5.5-pro",
        "gpt-5.5-pro-2026-04-23",
        "o3-pro-2025-06-10",
        "o1-pro-2025-03-19",
    ],
)
def test_responses_only_snapshots_fail_before_key_or_client(model):

    with pytest.raises(ValueError, match="not supported for Chat Completions"):
        prepare_request(model, [])


def test_unknown_snapshot_keeps_api_validation():
    model = "gpt-5.4-pro-2099-01-01"
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(response)
    assert gpt_integration.send_prepared_request(client, prepare_request(model, []))[0] == "pong"
    assert completions.calls[0]["model"] == model


def test_transport_stream_returns_collected_chunks():
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hel"))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=None))]),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="lo"))]),
    ]
    client, completions = _build_client(iter(chunks))
    streamed_updates = []
    generated_text, response_time, response_payload = gpt_integration.send_prepared_request(
        client,
        prepare_request("gpt-5-nano", [{"role": "user", "content": "hello"}]),
        stream=True,
        on_text=streamed_updates.append,
    )
    assert generated_text == "Hello"
    assert isinstance(response_time, float)
    assert response_payload == chunks
    assert streamed_updates == ["Hel", "lo"]
    assert completions.calls[0]["stream"] is True


def test_stream_without_callback_is_silent(capsys):
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))])]
    client, _ = _build_client(iter(chunks))
    content, _, payload = gpt_integration.send_prepared_request(
        client, prepare_request("4o", []), stream=True
    )
    assert content == "hello"
    assert payload == chunks
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("stream", [False, True])
def test_explicit_controls_and_usage_event(stream):
    chunks = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="pong"))]),
        SimpleNamespace(choices=[]),
    ]
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(iter(chunks) if stream else response)
    options = {"verbosity": "low", "max_completion_tokens": 100}
    if stream:
        options["stream_options"] = {"include_usage": True}
    updates = []
    content, _, payload = gpt_integration.send_prepared_request(
        client,
        prepare_request("gpt-5.4", [], reasoning_effort="high", request_options=options),
        stream=stream,
        on_text=updates.append,
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
def test_request_options_cannot_override_owned_controls(stream, options):
    with pytest.raises(ValueError, match="reserved"):
        prepare_request(messages=[], request_options=options)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "model, effort, temperature, expected",
    [
        ("gpt-5-nano", None, None, None),
        ("gpt-5", "minimal", None, None),
        ("gpt-5.1", None, 0.3, 0.3),
        ("gpt-5.2", "none", 0.3, 0.3),
        ("gpt-5.4", "high", None, None),
        ("gpt-5.4-mini", "none", 0.3, 0.3),
        ("o3-2025-04-16", "low", None, None),
        ("gpt-4o", None, 0.3, 0.3),
        ("gpt-6-luna", None, None, None),
        ("gpt-6.1-sol", None, None, None),
        ("gpt-6-astra", None, None, None),
        ("gpt-6-luna", "none", 0.3, 0.3),
        ("gpt-6.1-sol", "max", None, None),
        ("gpt-6-astra", "low", None, None),
    ],
)
def test_sampling_and_reasoning_controls(stream, model, effort, temperature, expected):
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="hi"))])
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])
    client, completions = _build_client(iter([chunk]) if stream else response)
    gpt_integration.send_prepared_request(
        client,
        prepare_request(model, [], temperature=temperature, reasoning_effort=effort),
        stream=stream,
        on_text=lambda _: None,
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
        ("gpt-5.4", "high", None, {"top_p": 0.5}),
        ("o3", None, None, {"extra_body": {"logprobs": True}}),
        ("gpt-5.4-pro", None, 1, {}),
        ("gpt-6.1-sol", "none", 1, {}),
        ("gpt-6-astra", "minimal", 1, {}),
        ("gpt-6-luna", None, 0.3, {}),
    ],
)
def test_unsupported_controls_fail_during_preparation(model, effort, temperature, options):
    with pytest.raises(ValueError):
        prepare_request(
            model, [], temperature=temperature, reasoning_effort=effort, request_options=options
        )


@pytest.mark.parametrize(
    "model,stream,effort,temperature,options,expected",
    [
        ("5.6", False, None, 1, {"top_p": 0.9}, {"temperature": 1, "top_p": 0.9}),
        ("6-sol", True, "none", 0.3, {}, {"reasoning_effort": "none", "temperature": 0.3}),
    ],
)
def test_new_models_use_chat_transport(model, stream, effort, temperature, options, expected):
    chunks = [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="pong"))])]
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])
    client, completions = _build_client(iter(chunks) if stream else response)
    result = gpt_integration.send_prepared_request(
        client,
        prepare_request(
            model, [], temperature=temperature, reasoning_effort=effort, request_options=options
        ),
        stream=stream,
        on_text=lambda _: None,
    )
    assert result[0] == "pong"
    assert result[2] == chunks if stream else result[2] is response
    assert completions.calls == [
        dict(
            messages=[],
            model="gpt-6-sol" if stream else "gpt-5.6-sol",
            n=1,
            stream=stream,
            **expected,
        )
    ]


def test_library_rejects_explicit_one_before_key_or_client():

    with pytest.raises(ValueError, match="Temperature is not supported"):
        prepare_request("6-sol", [], temperature=1)


@pytest.mark.parametrize("failure", ["dispatch", "stream", "callback"])
def test_library_errors_propagate_silently_and_leave_client_open(capsys, failure):
    expected = RuntimeError("synthetic failure")
    closed = []

    def events():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="part"))])
        if failure == "stream":
            raise expected

    def create(**kwargs):
        if failure == "dispatch":
            raise expected
        return events()

    def callback(text):
        assert text == "part"
        if failure == "callback":
            raise expected

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=lambda: closed.append(True),
    )
    with pytest.raises(RuntimeError) as caught:
        gpt_integration.send_prepared_request(
            client, prepare_request("4o", []), stream=True, on_text=callback
        )
    assert caught.value is expected
    assert closed == []
    assert capsys.readouterr() == ("", "")


def test_stream_callback_runs_before_next_event_and_retains_raw_payload(capsys):
    updates = []
    first = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="first"))])
    metadata = SimpleNamespace(choices=[], usage=object())

    def events():
        yield first
        assert updates == ["first"]
        yield metadata

    client, _ = _build_client(events())
    text, _, payload = gpt_integration.send_prepared_request(
        client, prepare_request("4o", []), stream=True, on_text=updates.append
    )
    assert text == "first"
    assert payload == [first, metadata]
    assert payload[1] is metadata
    assert capsys.readouterr() == ("", "")
