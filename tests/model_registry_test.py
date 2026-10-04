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
        "gpt-3.5-turbo-1106",
        "gpt-4-32k",
        "gpt-4-32k-0613",
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


@pytest.mark.parametrize("model, expected", [("gpt-4-0613", 30), ("gpt-4-turbo-2024-04-09", 10)])
def test_snapshot_input_prices_use_per_million_units(model, expected):
    assert model_registry.get_input_price_per_million(model, 100) == expected


@pytest.mark.parametrize(
    "model, input_price, output_price",
    [
        ("gpt-3.5-turbo", 0.50, 1.50),
        ("gpt-3.5-turbo-0125", 0.50, 1.50),
        ("gpt-3.5-turbo-1106", 1.00, 2.00),
        ("gpt-3.5-turbo-instruct", 1.50, 2.00),
        ("gpt-4", 30, 60),
        ("gpt-4-0613", 30, 60),
        ("gpt-4-turbo", 10, 30),
        ("gpt-4-turbo-2024-04-09", 10, 30),
        ("gpt-4-32k", 60, 120),
        ("gpt-4-32k-0613", 60, 120),
    ],
)
def test_source_backed_legacy_prices(model, input_price, output_price):
    # Fixed USD/1M values from OpenAI pricing/deprecations, checked 2026-10-03.
    band, tier = model_registry.get_price_band(model, 100)
    assert band.input == input_price
    assert band.output == output_price
    assert band.cached_input is None
    assert tier is None
