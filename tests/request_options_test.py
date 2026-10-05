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


@pytest.mark.parametrize(
    "model, efforts",
    [
        ("gpt-5.5", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.5-2026-04-23", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.6-sol", ("none", "low", "medium", "high", "xhigh", "max")),
        ("gpt-5.6-terra", ("none", "low", "medium", "high", "xhigh", "max")),
        ("gpt-5.6-luna", ("none", "low", "medium", "high", "xhigh", "max")),
        ("gpt-6-sol", ("none", "low", "medium", "high", "xhigh", "max")),
    ],
)
def test_new_model_published_efforts_preserve_omission(model, efforts):
    assert prepare_request(model, []).controls == {}
    for effort in efforts:
        expected = {"reasoning_effort": effort}
        if model == "gpt-6-sol" and effort == "none":
            expected["temperature"] = 1
        assert prepare_request(model, [], reasoning_effort=effort).controls == expected
    for effort in ("minimal", "bad", "max"):
        if effort not in efforts:
            with pytest.raises(ValueError, match="Reasoning effort .* is not supported"):
                prepare_request(model, [], reasoning_effort=effort)


@pytest.mark.parametrize(
    "model, effort, implicit, explicit_allowed",
    [
        ("gpt-4o", None, {"temperature": 1}, True),
        ("gpt-5-nano", None, {}, False),
        ("o3", None, {}, False),
        ("gpt-5.4", None, {"temperature": 1}, True),
        ("gpt-5.4", "high", {"reasoning_effort": "high"}, False),
        ("gpt-6-sol", None, {}, False),
        ("gpt-6-sol", "none", {"reasoning_effort": "none", "temperature": 1}, True),
        ("gpt-6.1-sol", None, {}, False),
        ("gpt-5.5", None, {}, True),
        ("gpt-5.6-sol", "none", {"reasoning_effort": "none"}, True),
        ("unknown-library-model", None, {"temperature": 1}, True),
    ],
)
def test_sampling_policy_distinguishes_implicit_and_explicit_one(
    model, effort, implicit, explicit_allowed
):
    assert prepare_request(model, [], reasoning_effort=effort).controls == implicit
    assert prepare_request(model, [], temperature=None, reasoning_effort=effort).controls == {
        key: value for key, value in implicit.items() if key != "temperature"
    }
    for value in (1, 0.3):
        if explicit_allowed:
            assert prepare_request(
                model, [], temperature=value, reasoning_effort=effort
            ).controls == {**implicit, "temperature": value}
        else:
            with pytest.raises(ValueError, match="Temperature is not supported"):
                prepare_request(model, [], temperature=value, reasoning_effort=effort)


@pytest.mark.parametrize(
    "model", ["gpt-5.5", "gpt-5.5-2026-04-23", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
)
def test_unverified_sampling_forwards_explicit_controls_without_mutation(model):
    options = {"top_p": 0.9, "extra_body": {"logprobs": True, "top_logprobs": 2}}
    for effort in (None, "none", "high"):
        controls = prepare_request(
            model, [], temperature=1, reasoning_effort=effort, request_options=options
        ).controls
        assert controls == {
            **options,
            "temperature": 1,
            **({"reasoning_effort": effort} if effort else {}),
        }
        assert options == {"top_p": 0.9, "extra_body": {"logprobs": True, "top_logprobs": 2}}
        implicit = prepare_request(model, [], reasoning_effort=effort, request_options=options)
        assert "temperature" not in implicit.controls


@pytest.mark.parametrize(
    "model,efforts",
    [
        ("gpt-5", ("minimal", "low", "medium", "high")),
        ("gpt-5.1-2025-11-13", ("none", "low", "medium", "high")),
        ("gpt-5.2-2025-12-11", ("none", "low", "medium", "high", "xhigh")),
        ("gpt-5.4-2026-03-05", ("none", "low", "medium", "high", "xhigh")),
    ],
)
def test_registered_parent_effort_contract_is_independent_of_sampling(model, efforts):
    assert "reasoning_effort" not in prepare_request(model, []).controls
    for effort in efforts:
        assert (
            prepare_request(model, [], reasoning_effort=effort).controls["reasoning_effort"]
            == effort
        )
    for effort in ("max", "bad"):
        with pytest.raises(ValueError, match="Reasoning effort .* is not supported"):
            prepare_request(model, [], reasoning_effort=effort)
