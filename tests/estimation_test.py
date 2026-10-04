import hashlib
import os
import shutil
import socket
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
import tiktoken.load
import tiktoken.registry
from click.testing import CliRunner

from lmterminal import cli, estimation, gpt_integration, lib
from lmterminal.request_options import prepare_request


@pytest.mark.parametrize(
    "name, model",
    [
        ("gpt-6-luna", "gpt-6-luna"),
        ("gpt-6.1-sol", "gpt-6.1-sol"),
        ("gpt-6-astra", "gpt-6-astra"),
        ("sol", "gpt-6.1-sol"),
    ],
)
def test_current_ids_have_no_guessed_tokenizer(monkeypatch, name, model):
    def unknown_tokenizer(name):
        assert name == model
        raise KeyError(name)

    monkeypatch.setattr(estimation.tiktoken, "encoding_name_for_model", unknown_tokenizer)
    result = estimation.estimate_request(prepare_request(name, [{"role": "user", "content": "hi"}]))
    assert result.model == model
    assert result.encoding is None
    assert result.message_tokens is None
    assert result.input_tokens is None
    assert result.input_cost_usd is None
    assert result.warnings == ("No known tokenizer for this model.",)


@pytest.fixture(autouse=True)
def no_provider_or_download(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail(
            "Estimation tests must not access credentials, providers or tokenizer downloads"
        )

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(lib, "get_api_key", forbidden)
    monkeypatch.setattr(gpt_integration, "_get_client", forbidden)
    monkeypatch.setattr(gpt_integration.openai, "OpenAI", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(tiktoken.load, "read_file", forbidden)
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})


@pytest.fixture
def fake_encoder(monkeypatch):
    # Only tests about request scope/control flow use this non-tokenizer stand-in.
    encoder = SimpleNamespace(encode_ordinary=lambda text: list(text))
    monkeypatch.setattr(estimation.tiktoken, "get_encoding", lambda _: encoder)
    return encoder


@pytest.fixture
def real_encoder(monkeypatch, tmp_path):
    # Explicit opt-in asset source. Never consult a user's implicit/default cache.
    source = os.environ.get("LMT_TEST_TIKTOKEN_CACHE")
    if not source:
        pytest.skip("Set LMT_TEST_TIKTOKEN_CACHE to a provisioned o200k_base cache")
    url = "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
    filename = hashlib.sha1(url.encode()).hexdigest()
    target = tmp_path / "oracle-cache"
    target.mkdir()
    shutil.copyfile(Path(source) / filename, target / filename)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(target))
    return tiktoken.get_encoding("o200k_base")


def test_real_text_oracles_and_literal_cli(real_encoder):
    # Tokenizer oracles, not measurements of provider usage.
    assert len(real_encoder.encode_ordinary("お誕生日おめでとう")) == 8
    assert len(real_encoder.encode_ordinary("<|endoftext|>")) == 7
    request = prepare_request("5-nano", gpt_integration.format_prompt("", "hello"))
    result = estimation.estimate_request(request)
    assert result.message_tokens == result.input_tokens == 12
    assert result.encoding == "o200k_base"
    assert result.input_cost_usd == Decimal("0.00000060")
    named = estimation.estimate_request(
        prepare_request("5-nano", [{"role": "user", "content": "hello", "name": "Bob"}])
    )
    assert named.input_tokens == 10  # 3 framing + role/content/name + 1 name + 3 priming
    output = CliRunner().invoke(cli.lmt, ["--tokens"], input="<|endoftext|>")
    assert output.exit_code == 0, output.output
    assert "Estimated input tokens: ~18" in output.output
    assert "Standard uncached input" in output.output
    assert "USD 0.0000009" in output.output
    assert "cached tokens" in output.output


@pytest.mark.parametrize(
    "model,tokens,rate,cost,tier",
    [
        ("gpt-5-nano", 12, "0.05", "0.00000060", None),
        ("gpt-5.4", 271999, "2.5", "0.6799975", "short"),
        ("gpt-5.4", 272000, "2.5", "0.68", "short"),
        ("gpt-5.4", 272001, "5", "1.360005", "long"),
        ("gpt-4.1", 300000, "2", "0.6", None),
    ],
)
def test_numeric_prices(model, tokens, rate, cost, tier):
    assert estimation.input_price(model, tokens) == (Decimal(rate), Decimal(cost), tier)


@pytest.mark.parametrize("scope", ["full", "partial", "unavailable"])
def test_cli_estimate_colors_and_plain_output(monkeypatch, scope):
    estimate = estimation.InputEstimate(
        model="gpt-5.4",
        message_tokens=12 if scope != "unavailable" else None,
        input_tokens=12 if scope == "full" else None,
        input_cost_usd=Decimal("0.000030") if scope == "full" else None,
        input_rate_usd_per_million=Decimal("2.5") if scope == "full" else None,
        pricing_context="short" if scope == "full" else None,
        warnings=("Option `tools` is not counted locally.",) if scope == "partial" else (),
    )
    monkeypatch.setattr(lib, "estimate_request", lambda _: estimate)
    runner = CliRunner()
    colored = runner.invoke(cli.lmt, ["--tokens"], input="hello", color=True)
    plain = runner.invoke(cli.lmt, ["--tokens"], input="hello")
    assert colored.exit_code == plain.exit_code == (1 if scope == "unavailable" else 0)
    assert click.style("gpt-5.4", fg="blue") in colored.output
    if scope != "unavailable":
        assert click.style("~12", fg="yellow") in colored.output
    if scope == "full":
        for value in ("USD 0.000030", "USD 2.5", "short"):
            assert click.style(value, fg="yellow") in colored.output
    else:
        assert "Request input tokens and cost: unavailable" in colored.output
    assert "\x1b[" not in plain.output
    assert click.unstyle(colored.output) == plain.output
    assert "Excludes output/reasoning, tool fees and service-tier adjustments; not a bill." in (
        colored.output
    )


@pytest.mark.parametrize(
    "option",
    [
        "tools=[]",
        "functions=[]",
        "response_format.type=json_schema",
        "extra_body.tools=[]",
        "extra_body.response_format.type=json_schema",
        "unknown=value",
    ],
)
def test_cli_partial_options(fake_encoder, option):
    result = CliRunner().invoke(cli.lmt, ["--tokens", "-o", option], input="hello")
    assert result.exit_code == 0, result.output
    assert "Message-only token estimate:" in result.output
    assert "Request input tokens and cost: unavailable" in result.output
    assert option.split("=")[0].split(".type")[0] in result.output
    assert "USD" not in result.output
    assert "Estimated input tokens:" not in result.output


@pytest.mark.parametrize(
    "messages",
    [
        None,
        ["hello"],
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        [{"role": "assistant", "content": None}],
        [{"role": "assistant", "content": "hi", "tool_calls": []}],
        [{"role": "user", "content": "hi", "name": 12}],
        [{"role": "tool", "content": "hi"}],
    ],
)
def test_unsupported_message_shape(fake_encoder, messages):
    result = estimation.estimate_request(prepare_request("gpt-5-nano", messages))
    assert result.message_tokens is result.input_tokens is result.input_cost_usd is None
    assert not result.request_complete
    assert result.warnings


def test_unknown_encoder_and_known_encoder_unknown_price(fake_encoder):
    request = prepare_request("unknown-model", [{"role": "user", "content": "hello"}])
    result = estimation.estimate_request(request)
    assert result.input_tokens is result.input_cost_usd is None
    assert "No known tokenizer" in result.warnings[0]
    result = estimation.estimate_request(prepare_request("gpt-4o-2099-01-01", request.messages))
    assert result.input_tokens is not None
    assert result.request_complete
    assert result.input_cost_usd is None
    assert "No known Standard input price" in result.warnings[0]


@pytest.mark.parametrize("option", ["service_tier=flex", "extra_body.service_tier=priority"])
def test_service_tier_unpriced(fake_encoder, option):
    result = CliRunner().invoke(cli.lmt, ["--tokens", "-o", option], input="hello")
    assert result.exit_code == 0
    assert "Estimated input tokens:" in result.output
    assert "Input cost: unavailable" in result.output
    assert "service_tier" in result.output
    assert "USD" not in result.output


def test_missing_assets_are_actionable(monkeypatch):
    downloads = []

    def denied(path):
        downloads.append(path)
        raise OSError("Public asset unavailable")

    monkeypatch.setattr(tiktoken.load, "read_file", denied)
    result = CliRunner().invoke(cli.lmt, ["--tokens"], input="hello")
    assert result.exit_code == 1
    assert "Tokenizer data for o200k_base is unavailable" in result.output
    assert "TIKTOKEN_CACHE_DIR" in result.output
    assert "Estimated input tokens:" not in result.output
    assert len(downloads) == 1


@pytest.mark.parametrize("tokens", [False, True])
@pytest.mark.parametrize(
    "args,error",
    [
        (["--temperature", "0.7"], "Temperature is not supported"),
        (["-o", "extra_body.top_p=0.5"], "top_p"),
    ],
)
def test_validation_precedes_key_and_estimation(tokens, args, error):
    result = CliRunner().invoke(cli.lmt, args + (["--tokens"] if tokens else []), input="hi")
    assert result.exit_code == 2
    assert error in result.output


@pytest.mark.parametrize("template_model", ["gpt-5.4-pro", "unknown", None])
@pytest.mark.parametrize("tokens", [False, True])
def test_invalid_template_model_before_key(monkeypatch, template_model, tokens):
    monkeypatch.setattr(lib, "handle_template", lambda *args: ("", "hi", template_model))
    result = CliRunner().invoke(
        cli.lmt, ["-t", "fixture"] + (["--tokens"] if tokens else []), input="hi"
    )
    assert result.exit_code == 2
    assert "Invalid template model" in result.output


@pytest.mark.parametrize("model,expected_model", [(None, "gpt-4o"), ("5.4", "gpt-5.4")])
def test_estimate_and_transport_receive_same_prepared_values(
    monkeypatch, fake_encoder, model, expected_model
):
    monkeypatch.setattr(
        lib,
        "handle_template",
        lambda name, system, text, model: ("Template system.", "Prefix:" + text, "4o"),
    )
    requests = []
    estimate = estimation.estimate_request

    def capture(request):
        requests.append(request)
        return estimate(request)

    monkeypatch.setattr(lib, "estimate_request", capture)
    args = ["-t", "fixture", "--emoji", "-o", "max_completion_tokens=100", "Summarize"]
    if model:
        args += ["-m", model]
    runner = CliRunner()
    result = runner.invoke(cli.lmt, args + ["--tokens"], input="stdin text")
    assert result.exit_code == 0, result.output
    monkeypatch.setattr(lib, "get_api_key", lambda: "fixture-key")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    result = runner.invoke(cli.lmt, args + ["--no-stream"], input="stdin text")
    assert result.exit_code == 0, result.output
    prepared = requests[0]
    assert prepared.model == expected_model
    assert prepared.messages[0]["content"] == lib.add_emoji("Template system.")
    assert prepared.messages[1]["content"] == "Prefix:stdin text\n___\nSummarize"
    assert prepared.controls == {"temperature": 1, "max_completion_tokens": 100}
    assert calls == [
        dict(
            model=prepared.model, messages=prepared.messages, n=1, stream=False, **prepared.controls
        )
    ]
