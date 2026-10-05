"""Registry-wide selection with real offline vocabularies, not provider usage."""

import pytest
import tiktoken

from lmterminal import estimation, model_registry
from lmterminal.request_options import prepare_request

pytestmark = pytest.mark.usefixtures("no_provider_or_download")

# Independent expectations from the upstream 0.14 resolver. Keep these literal:
# the set assertion makes new registry entries require an explicit expectation.
EXPECTED_MODELS = {
    "cl100k_base": (
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-4",
        "gpt-4-0613",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
    ),
    "o200k_base": (
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-11-20",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4.1",
        "gpt-4.1-2025-04-14",
        "gpt-4.1-mini",
        "gpt-4.1-mini-2025-04-14",
        "gpt-4.1-nano",
        "gpt-4.1-nano-2025-04-14",
        "o1",
        "o1-2024-12-17",
        "o3",
        "o3-2025-04-16",
        "o3-mini",
        "o3-mini-2025-01-31",
        "o4-mini",
        "o4-mini-2025-04-16",
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-5.1",
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
    ),
    None: ("gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"),
}
EXPECTED_ALIASES = {
    "chatgpt": "gpt-3.5-turbo",
    "3.5": "gpt-3.5-turbo",
    "4": "gpt-4",
    "gpt4": "gpt-4",
    "4t": "gpt-4-turbo",
    "4-turbo": "gpt-4-turbo",
    "gpt4-turbo": "gpt-4-turbo",
    "4o": "gpt-4o",
    "4o-mini": "gpt-4o-mini",
    "4omini": "gpt-4o-mini",
    "4om": "gpt-4o-mini",
    "4.1": "gpt-4.1",
    "4.1-mini": "gpt-4.1-mini",
    "4.1-nano": "gpt-4.1-nano",
    "5": "gpt-5",
    "gpt5": "gpt-5",
    "5-mini": "gpt-5-mini",
    "5-nano": "gpt-5-nano",
    "5.1": "gpt-5.1",
    "5.2": "gpt-5.2",
    "5.4": "gpt-5.4",
    "5.4-mini": "gpt-5.4-mini",
    "5.4-nano": "gpt-5.4-nano",
    "6-luna": "gpt-6-luna",
    "6.1-sol": "gpt-6.1-sol",
    "6-astra": "gpt-6-astra",
    "luna": "gpt-6-luna",
    "sol": "gpt-6.1-sol",
    "astra": "gpt-6-astra",
}
CORPUS = "Hello world. Bonjour Sébastien. お誕生日おめでとう\nprint(123 + 456) <|endoftext|>"


def test_expected_tables_cover_every_supported_model_and_alias():
    models = model_registry.get_valid_models()
    assert {model for group in EXPECTED_MODELS.values() for model in group} == set(models)
    assert {
        alias: model for model, aliases in models.items() for alias in (aliases or ())
    } == EXPECTED_ALIASES
    for alias, canonical in EXPECTED_ALIASES.items():
        assert model_registry.resolve_model_name(alias) == canonical


@pytest.mark.parametrize("encoding_name,japanese_tokens", [("cl100k_base", 9), ("o200k_base", 8)])
def test_vocabulary_oracles(tokenizer_cache, encoding_name, japanese_tokens):
    encoder = tiktoken.get_encoding(encoding_name)
    assert len(encoder.encode_ordinary("お誕生日おめでとう")) == japanese_tokens
    assert len(encoder.encode_ordinary("<|endoftext|>")) == 7
    tokens = encoder.encode_ordinary(CORPUS)
    assert encoder.decode(tokens) == CORPUS
    assert not set(tokens) & {encoder.eot_token}


@pytest.mark.parametrize("encoding_name", ["cl100k_base", "o200k_base"])
def test_every_known_model_and_alias_uses_real_encoder(tokenizer_cache, monkeypatch, encoding_name):
    resolve = tiktoken.encoding_for_model
    calls = []

    def record(model):
        calls.append(model)
        return resolve(model)

    monkeypatch.setattr(tiktoken, "encoding_for_model", record)
    for canonical in EXPECTED_MODELS[encoding_name]:
        aliases = [alias for alias, model in EXPECTED_ALIASES.items() if model == canonical]
        for name in (canonical, *aliases):
            request = prepare_request(name, [{"role": "user", "content": CORPUS}])
            assert request.model == canonical, name
            encoder = resolve(request.model)
            assert encoder.name == encoding_name, name
            encoded = encoder.encode_ordinary(CORPUS)
            assert encoder.decode(encoded) == CORPUS, name
            calls.clear()
            result = estimation.estimate_request(request)
            assert calls == [canonical], name
            assert result.encoding == encoding_name, name
            assert result.input_tokens == result.message_tokens == len(encoded) + 7, name
            assert result.request_complete and not result.warnings, name
            assert result.input_cost_usd is not None, name


def test_every_unknown_model_and_alias_has_no_encoder_or_cost():
    for canonical in EXPECTED_MODELS[None]:
        aliases = [alias for alias, model in EXPECTED_ALIASES.items() if model == canonical]
        for name in (canonical, *aliases):
            request = prepare_request(name, [{"role": "user", "content": CORPUS}])
            assert request.model == canonical, name
            with pytest.raises(KeyError):
                tiktoken.encoding_for_model(request.model)
            result = estimation.estimate_request(request)
            assert result.encoding is result.message_tokens is result.input_tokens is None, name
            assert result.input_cost_usd is None, name
            assert not result.request_complete, name
            assert result.warnings == ("No known tokenizer for this model.",), name
