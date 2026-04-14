from types import SimpleNamespace

from lmterminal import gpt_integration


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
        model="gpt-5-nano",
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
            "model": "gpt-5-nano",
            "n": 1,
            "temperature": 0.3,
            "stream": False,
        }
    ]


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
