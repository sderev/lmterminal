from types import SimpleNamespace

import pytest
from click.exceptions import BadParameter
from click.testing import CliRunner

from lmterminal import cli, lib


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("chatgpt", "gpt-3.5-turbo"),
        ("3.5", "gpt-3.5-turbo"),
        ("gpt-4", "gpt-4"),
        ("4", "gpt-4"),
        ("gpt4", "gpt-4"),
        ("gpt-4-turbo", "gpt-4-turbo"),
        ("4t", "gpt-4-turbo"),
        ("4-turbo", "gpt-4-turbo"),
        ("gpt4-turbo", "gpt-4-turbo"),
        ("gpt-4o", "gpt-4o"),
        ("4o", "gpt-4o"),
        ("gpt-4o-2024-05-13", "gpt-4o-2024-05-13"),
        ("gpt-4o-2024-08-06", "gpt-4o-2024-08-06"),
        ("gpt-4o-2024-11-20", "gpt-4o-2024-11-20"),
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("4o-mini", "gpt-4o-mini"),
        ("4omini", "gpt-4o-mini"),
        ("4om", "gpt-4o-mini"),
        ("gpt-4o-mini-2024-07-18", "gpt-4o-mini-2024-07-18"),
        ("o1", "o1"),
        ("o1-2024-12-17", "o1-2024-12-17"),
        ("gpt-4.1", "gpt-4.1"),
        ("4.1", "gpt-4.1"),
        ("gpt-4.1-2025-04-14", "gpt-4.1-2025-04-14"),
        ("gpt-4.1-mini", "gpt-4.1-mini"),
        ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini-2025-04-14"),
        ("gpt-4.1-nano", "gpt-4.1-nano"),
        ("4.1-nano", "gpt-4.1-nano"),
        ("gpt-4.1-nano-2025-04-14", "gpt-4.1-nano-2025-04-14"),
        ("o3", "o3"),
        ("o3-2025-04-16", "o3-2025-04-16"),
        ("o3-mini", "o3-mini"),
        ("o3-mini-2025-01-31", "o3-mini-2025-01-31"),
        ("o4-mini", "o4-mini"),
        ("o4-mini-2025-04-16", "o4-mini-2025-04-16"),
    ],
)
def test_validate_model_name(alias, canonical):
    assert cli.validate_model_name(None, None, alias) == canonical


def test_validate_model_name_invalid():
    with pytest.raises(BadParameter):
        cli.validate_model_name(None, None, "invalid-model")


@pytest.mark.parametrize(
    "value",
    [
        0,
        0.5,
        1.0,
        1,
        2,
    ],
)
def test_validate_temperature(value):
    assert cli.validate_temperature(None, None, value) == value


@pytest.mark.parametrize("value", [-0.1, 2.1])
def test_validate_temperature_invalid(value):
    with pytest.raises(BadParameter):
        cli.validate_temperature(None, None, value)


@pytest.mark.parametrize(
    "args, expected_prompt, stream",
    [
        ([], "piped input", True),
        (["--rich"], "piped input", True),
        (["prompt", "--raw", "Summarize"], "piped input\n___\nSummarize", True),
        (["--no-stream"], "piped input", False),
        (["prompt", "--no-stream", "Summarize"], "piped input\n___\nSummarize", False),
    ],
)
def test_piped_prompt_preserves_stream_choice(monkeypatch, tmp_path, args, expected_prompt, stream):
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(lib, "get_config_path", lambda: tmp_path / "config.json")
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if kwargs["stream"]:
            return iter(
                SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])
                for text in ("Hel", "lo")
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Hello"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(lib.openai_utils, "_get_client", lambda _api_key: client)

    result = CliRunner().invoke(
        cli.lmt, args, input="piped input\n", env={"TERM": "xterm", "FORCE_COLOR": "1"}
    )

    assert result.exit_code == 0, result.output
    assert result.stdout == ("Hello" if stream else "Hello\n")
    assert calls == [
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": expected_prompt},
            ],
            "model": lib.DEFAULT_MODEL,
            "n": 1,
            "temperature": 1,
            "stream": stream,
        }
    ]


@pytest.mark.integration
@pytest.mark.parametrize("model", list(cli.VALID_MODELS), ids=str)
def test_live_call(model, request):
    if not request.config.getoption("--run-live"):
        pytest.skip("Skipping live call test. Use --run-live to enable.")
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["--model", model, "Ping"])

    if result.exit_code != 0:
        pytest.fail(result.output)
