import pytest

from lmterminal.request_options import prepare_request


@pytest.mark.parametrize(
    "model, supported, rejected",
    [
        ("gpt-6-luna", ("none", "low", "medium", "high", "xhigh", "max"), ("minimal", "bad")),
        ("gpt-6.1-sol", ("low", "medium", "high", "xhigh", "max"), ("none", "minimal", "bad")),
        ("gpt-6-astra", ("low", "medium", "high", "xhigh", "max"), ("none", "minimal", "bad")),
    ],
)
def test_current_reasoning_and_sampling_policy(model, supported, rejected):
    alias = model.rsplit("-", 1)[1]
    request = prepare_request(alias, [])
    assert request.model == model
    assert request.controls == {}
    with pytest.raises(ValueError, match="Temperature is not supported"):
        prepare_request(model, [], temperature=0.3)
    for effort in supported:
        request = prepare_request(model.removeprefix("gpt-"), [], reasoning_effort=effort)
        assert request.model == model
        expected = {"reasoning_effort": effort}
        if effort == "none":
            expected["temperature"] = 1
        assert request.controls == expected
    for effort in rejected:
        with pytest.raises(ValueError, match="Use --reasoning-effort with one of:"):
            prepare_request(alias, [], reasoning_effort=effort)


@pytest.mark.parametrize("key", ["top_p", "logprobs", "top_logprobs"])
@pytest.mark.parametrize("in_body", [False, True])
def test_current_sampling_options_require_explicit_none(key, in_body):
    value = True if key == "logprobs" else 1
    options = {"extra_body": {key: value}} if in_body else {key: value}
    for effort in (None, "max"):
        with pytest.raises(ValueError, match=f"Option `{key}` is not supported"):
            prepare_request("gpt-6-luna", [], reasoning_effort=effort, request_options=options)
    request = prepare_request(
        "gpt-6-luna", [], temperature=0.3, reasoning_effort="none", request_options=options
    )
    assert request.controls == {"reasoning_effort": "none", "temperature": 0.3, **options}
