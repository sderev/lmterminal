import pytest
from click.testing import CliRunner

from lmterminal import cli, gpt_integration, lib, model_registry


@pytest.mark.parametrize(
    "name, canonical",
    [
        ("5.4", "gpt-5.4"),
        ("GPT-5.4-MINI", "gpt-5.4-mini"),
        ("gpt-3.5-turbo-0125", "gpt-3.5-turbo-0125"),
        ("chatgpt", "gpt-3.5-turbo"),
    ],
)
def test_resolve_chat_model(name, canonical):
    assert model_registry.resolve_model_name(name) == canonical


@pytest.mark.parametrize(
    "name",
    [
        "gpt-5.4-pro",
        "5.4-pro",
        "o3-pro",
        "gpt-5.3-codex",
        "gpt-5-pro",
        "o1-pro",
        "codex-mini-latest",
        "gpt-3.5-turbo-instruct",
        "gpt-5.3-chat-latest",
        "gpt-4o-search-preview",
        "o1-mini",
    ],
)
def test_cli_rejects_other_endpoints(name):
    result = CliRunner().invoke(cli.lmt, ["--model", name, "hello"])
    assert result.exit_code == 2
    assert "Invalid model name" in result.output


def test_models_list_needs_no_request_or_key(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Model listing must stay offline")

    monkeypatch.setattr(lib, "get_api_key", forbidden)
    monkeypatch.setattr(gpt_integration, "_get_client", forbidden)
    result = CliRunner().invoke(cli.lmt, ["models"])
    assert result.exit_code == 0
    assert "gpt-5.4" in result.output
    assert "gpt-3.5-turbo-0125" in result.output
    assert "gpt-5.4-pro" not in result.output
    assert "gpt-5.3-codex" not in result.output
    assert "gpt-5.3-chat-latest" not in result.output
    assert "o1-mini" not in result.output
    assert "  Alias: 5.4\n" in result.output
    assert "  Aliases: 5.4\n" not in result.output


@pytest.mark.parametrize(
    "tokens, price, tier", [(271999, 2.5, "short"), (272000, 2.5, "short"), (272001, 5.0, "long")]
)
def test_prompt_pricing_boundary(monkeypatch, tokens, price, tier):
    monkeypatch.setattr(gpt_integration, "num_tokens_from_messages", lambda *_: tokens)
    estimate = gpt_integration.estimate_prompt_cost_details([], "gpt-5.4")
    assert estimate.num_tokens == tokens
    assert estimate.price_per_1m_tokens == price
    assert estimate.pricing_context == tier
    assert estimate.cost == f"{tokens * price / 1_000_000:.6f}"


def test_tokens_command_does_not_read_key_or_request(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Token estimates must stay offline")

    monkeypatch.setattr(lib, "get_api_key", forbidden)
    monkeypatch.setattr(gpt_integration, "_get_client", forbidden)
    monkeypatch.setattr(gpt_integration, "num_tokens_from_messages", lambda *_: 272000)
    result = CliRunner().invoke(cli.lmt, ["--tokens", "-m", "5.4", "hello"])
    assert result.exit_code == 0, result.output
    assert "272000" in result.output
    assert "$0.680000" in result.output
    assert "short context" in result.output


@pytest.mark.parametrize("model, expected", [("gpt-4-0613", 30), ("gpt-4-turbo-2024-04-09", 10)])
def test_snapshot_input_prices_use_per_million_units(model, expected):
    assert model_registry.get_input_price_per_million(model, 100) == expected


def test_tokenizer_fallback_preserves_model_family(monkeypatch):
    calls = []

    def encoding_for_model(model):
        calls.append(model)
        raise KeyError(model)

    monkeypatch.setattr(gpt_integration.tiktoken, "encoding_for_model", encoding_for_model)
    monkeypatch.setattr(
        gpt_integration.tiktoken,
        "get_encoding",
        lambda _: type("Encoding", (), {"encode": lambda _, text: list(text)})(),
    )
    assert gpt_integration.num_tokens_from_string("hi", "gpt-5.4") == 2
    assert calls == ["gpt-5"]
