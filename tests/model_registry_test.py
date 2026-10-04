from dataclasses import replace

import pytest
from click.testing import CliRunner

from lmterminal import cli, estimation, gpt_integration, lib, model_registry


@pytest.mark.parametrize(
    "name, canonical",
    [
        ("5.4", "gpt-5.4"),
        ("GPT-5.4-MINI", "gpt-5.4-mini"),
        ("gpt-3.5-turbo-0125", "gpt-3.5-turbo-0125"),
        ("chatgpt", "gpt-3.5-turbo"),
        ("6-luna", "gpt-6-luna"),
        ("6.1-sol", "gpt-6.1-sol"),
        ("6-astra", "gpt-6-astra"),
        ("luna", "gpt-6-luna"),
        ("LUNA", "gpt-6-luna"),
        ("sol", "gpt-6.1-sol"),
        ("SOL", "gpt-6.1-sol"),
        ("astra", "gpt-6-astra"),
        ("ASTRA", "gpt-6-astra"),
    ],
)
def test_resolve_chat_model(name, canonical):
    assert model_registry.resolve_model_name(name) == canonical


@pytest.mark.parametrize("reverse", [False, True])
def test_family_alias_advances_by_supported_numeric_version(monkeypatch, reverse):
    spec = model_registry.MODEL_REGISTRY["gpt-6.1-sol"]
    entries = [
        ("gpt-6.10-sol", replace(spec, aliases=("6.10-sol",), alias_version=(6, 10))),
        ("gpt-6.9-sol", replace(spec, aliases=("6.9-sol",), alias_version=(6, 9))),
        ("gpt-7-sol", replace(spec, alias_version=(7, 0), chat_completions=False)),
        ("gpt-8-sol-codex", model_registry.ModelSpec(chat_completions=False)),
        ("gpt-8-sol-2099-01-01", model_registry.ModelSpec()),
    ]
    for name, candidate in reversed(entries) if reverse else entries:
        monkeypatch.setitem(model_registry.MODEL_REGISTRY, name, candidate)

    assert model_registry.resolve_model_name("sol") == "gpt-6.10-sol"
    assert model_registry.resolve_model_name("6.1-sol") == "gpt-6.1-sol"
    assert model_registry.resolve_model_name("gpt-6.1-sol") == "gpt-6.1-sol"
    models = model_registry.get_valid_models()
    assert models["gpt-6.10-sol"] == ("6.10-sol", "sol")
    assert [name for name, aliases in models.items() if aliases and "sol" in aliases] == [
        "gpt-6.10-sol"
    ]
    assert "gpt-7-sol" not in models
    result = CliRunner().invoke(cli.lmt, ["models"])
    assert result.exit_code == 0
    assert "gpt-6.10-sol\n  Aliases: 6.10-sol, sol\n" in result.output
    assert "gpt-6.1-sol\n  Alias: 6.1-sol\n" in result.output


def test_family_alias_rejects_duplicate_version(monkeypatch):
    monkeypatch.setitem(
        model_registry.MODEL_REGISTRY,
        "another-sol",
        model_registry.MODEL_REGISTRY["gpt-6.1-sol"],
    )
    with pytest.raises(ValueError, match="Duplicate alias version .* family `sol`"):
        model_registry.resolve_model_name("sol")


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
    monkeypatch.setattr(estimation.tiktoken, "encoding_name_for_model", forbidden)
    monkeypatch.setattr(estimation.tiktoken, "get_encoding", forbidden)
    assert model_registry.resolve_model_name("sol") == "gpt-6.1-sol"
    result = CliRunner().invoke(cli.lmt, ["models"])
    assert result.exit_code == 0
    assert "gpt-5.4" in result.output
    for model in ("gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"):
        assert model in result.output
    for aliases in ("6-luna, luna", "6.1-sol, sol", "6-astra, astra"):
        assert f"  Aliases: {aliases}\n" in result.output
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


@pytest.mark.parametrize(
    "model, short, long",
    [
        ("gpt-6-luna", (0.10, 0.01, 0.50), (0.20, 0.02, 0.75)),
        ("gpt-6.1-sol", (2.00, 0.10, 10.00), (4.00, 0.20, 15.00)),
        ("gpt-6-astra", (10.00, 1.00, 50.00), (20.00, 2.00, 75.00)),
    ],
)
def test_current_standard_prices_and_boundary(model, short, long):
    # OpenAI Standard pricing table, checked 2026-10-04; USD per million tokens.
    assert model_registry.get_price_band(model, 272_000) == (
        model_registry.PriceBand(*short),
        "short",
    )
    assert model_registry.get_price_band(model, 272_001) == (
        model_registry.PriceBand(*long),
        "long",
    )
