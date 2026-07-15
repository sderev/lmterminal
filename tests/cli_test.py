import pytest
from click.exceptions import BadParameter
from click.testing import CliRunner

from lmterminal import cli


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
        ("chatgpt", "gpt-3.5-turbo"),
        ("3.5", "gpt-3.5-turbo"),
        ("gpt-3.5-turbo-instruct", "gpt-3.5-turbo-instruct"),
        ("gpt-4", "gpt-4"),
        ("4", "gpt-4"),
        ("gpt4", "gpt-4"),
        ("gpt-4-turbo", "gpt-4-turbo"),
        ("4t", "gpt-4-turbo"),
        ("4-turbo", "gpt-4-turbo"),
        ("gpt4-turbo", "gpt-4-turbo"),
        ("gpt-4-32k", "gpt-4-32k"),
        ("4-32k", "gpt-4-32k"),
        ("gpt4-32k", "gpt-4-32k"),
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
        ("chatgpt-4o-latest", "chatgpt-4o-latest"),
        ("o1", "o1"),
        ("o1-2024-12-17", "o1-2024-12-17"),
        ("o1-preview", "o1-preview"),
        ("o1-preview-2024-09-12", "o1-preview-2024-09-12"),
        ("o1-mini", "o1-mini"),
        ("o1-mini-2024-09-12", "o1-mini-2024-09-12"),
        ("o1-pro", "o1-pro"),
        ("o1-pro-2025-03-19", "o1-pro-2025-03-19"),
        ("gpt-4.1", "gpt-4.1"),
        ("4.1", "gpt-4.1"),
        ("gpt-4.1-2025-04-14", "gpt-4.1-2025-04-14"),
        ("gpt-4.1-mini", "gpt-4.1-mini"),
        ("gpt-4.1-mini-2025-04-14", "gpt-4.1-mini-2025-04-14"),
        ("gpt-4.1-nano", "gpt-4.1-nano"),
        ("4.1-nano", "gpt-4.1-nano"),
        ("gpt-4.1-nano-2025-04-14", "gpt-4.1-nano-2025-04-14"),
        ("gpt-4.5-preview", "gpt-4.5-preview"),
        ("o3", "o3"),
        ("o3-2025-04-16", "o3-2025-04-16"),
        ("o3-mini", "o3-mini"),
        ("o3-mini-2025-01-31", "o3-mini-2025-01-31"),
        ("o4-mini", "o4-mini"),
        ("o4-mini-2025-04-16", "o4-mini-2025-04-16"),
        ("codex-mini-latest", "codex-mini-latest"),
        ("gpt-4o-search-preview", "gpt-4o-search-preview"),
        ("gpt-4o-search-preview-2025-03-11", "gpt-4o-search-preview-2025-03-11"),
        ("gpt-4o-mini-search-preview", "gpt-4o-mini-search-preview"),
        ("gpt-4o-mini-search-preview-2025-03-11", "gpt-4o-mini-search-preview-2025-03-11"),
        ("gpt-5", "gpt-5"),
        ("5", "gpt-5"),
        ("gpt5", "gpt-5"),
        ("gpt-5-mini", "gpt-5-mini"),
        ("5-mini", "gpt-5-mini"),
        ("gpt-5-nano", "gpt-5-nano"),
        ("5-nano", "gpt-5-nano"),
        ("gpt-5-chat-latest", "gpt-5-chat-latest"),
        ("gpt-5-codex", "gpt-5-codex"),
        ("gpt-5-pro", "gpt-5-pro"),
        ("5-pro", "gpt-5-pro"),
        ("gpt-5.1", "gpt-5.1"),
        ("5.1", "gpt-5.1"),
        ("gpt-5.1-chat-latest", "gpt-5.1-chat-latest"),
        ("gpt-5.1-codex", "gpt-5.1-codex"),
        ("gpt-5.1-codex-max", "gpt-5.1-codex-max"),
        ("gpt-5.1-codex-mini", "gpt-5.1-codex-mini"),
        ("gpt-5.2", "gpt-5.2"),
        ("5.2", "gpt-5.2"),
        ("gpt-5.2-chat-latest", "gpt-5.2-chat-latest"),
        ("gpt-5.2-codex", "gpt-5.2-codex"),
        ("gpt-5.2-pro", "gpt-5.2-pro"),
        ("5.2-pro", "gpt-5.2-pro"),
        ("gpt-5.3-chat-latest", "gpt-5.3-chat-latest"),
        ("gpt-5.3-codex", "gpt-5.3-codex"),
        ("gpt-5.4", "gpt-5.4"),
        ("5.4", "gpt-5.4"),
        ("gpt-5.4-mini", "gpt-5.4-mini"),
        ("5.4-mini", "gpt-5.4-mini"),
        ("gpt-5.4-nano", "gpt-5.4-nano"),
        ("5.4-nano", "gpt-5.4-nano"),
        ("gpt-5.4-pro", "gpt-5.4-pro"),
        ("5.4-pro", "gpt-5.4-pro"),
        ("o3-pro", "o3-pro"),
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


def test_reasoning_effort_rejects_unknown_value():
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["hello", "--reasoning-effort", "extreme"])

    assert result.exit_code != 0
    assert "Invalid value for '--reasoning-effort'" in result.output


def test_parse_request_options_supports_json_scalars_and_nested_keys():
    parsed_options = cli.parse_request_options(
        None,
        None,
        (
            "top_p=0.9",
            "store=true",
            'metadata.topic="cli"',
            'response_format={"type":"json_object"}',
        ),
    )

    assert parsed_options == {
        "top_p": 0.9,
        "store": True,
        "metadata": {"topic": "cli"},
        "response_format": {"type": "json_object"},
    }


def test_parse_request_options_leaves_plain_strings_unchanged():
    parsed_options = cli.parse_request_options(None, None, ("service_tier=priority",))

    assert parsed_options == {"service_tier": "priority"}


@pytest.mark.parametrize(
    "raw_option",
    [
        "missing-separator",
        "=value",
        'metadata..topic="cli"',
    ],
)
def test_parse_request_options_rejects_invalid_input(raw_option):
    with pytest.raises(BadParameter):
        cli.parse_request_options(None, None, (raw_option,))


def test_parse_request_options_rejects_duplicate_keys():
    with pytest.raises(BadParameter):
        cli.parse_request_options(None, None, ("top_p=0.9", "top_p=0.8"))


def test_parse_request_options_rejects_nested_override_after_null():
    with pytest.raises(BadParameter):
        cli.parse_request_options(None, None, ("metadata=null", 'metadata.topic="cli"'))


def test_prompt_forwards_reasoning_effort_and_request_options(monkeypatch):
    captured_call = {}

    def fake_prepare_and_generate_response(*args):
        captured_call.update(
            {
                "system": args[0],
                "template": args[1],
                "model": args[2],
                "emoji": args[3],
                "prompt_input": args[4],
                "temperature": args[5],
                "reasoning_effort": args[6],
                "request_options": args[7],
                "tokens": args[8],
                "no_stream": args[9],
                "raw": args[10],
                "debug": args[11],
            }
        )

    monkeypatch.setattr(cli, "prepare_and_generate_response", fake_prepare_and_generate_response)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)

    runner = CliRunner()
    result = runner.invoke(
        cli.lmt,
        [
            "hello",
            "--reasoning-effort",
            "high",
            "-o",
            "top_p=0.9",
            "-o",
            'metadata.topic="cli"',
        ],
    )

    assert result.exit_code == 0
    assert captured_call["model"] == cli.DEFAULT_MODEL
    assert captured_call["prompt_input"].endswith("hello")
    assert captured_call["reasoning_effort"] == "high"
    assert captured_call["request_options"] == {
        "top_p": 0.9,
        "metadata": {"topic": "cli"},
    }


@pytest.mark.parametrize(
    "reserved_key, dedicated_flag",
    [
        ("model=gpt-5.4", "--model"),
        ("n=2", None),
        ("temperature=0.9", "--temperature"),
        ("stream=true", "--no-stream"),
        ("reasoning_effort=high", "--reasoning-effort"),
    ],
)
def test_prompt_rejects_reserved_request_options(reserved_key, dedicated_flag):
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["hello", "-o", reserved_key])

    assert result.exit_code != 0
    if dedicated_flag is None:
        assert f"cannot set `{reserved_key.split('=', 1)[0]}`" in result.output
    else:
        assert dedicated_flag in result.output


@pytest.mark.integration
@pytest.mark.parametrize("model", list(cli.VALID_MODELS), ids=str)
def test_live_call(model, request):
    if not request.config.getoption("--run-live"):
        pytest.skip("Skipping live call test. Use --run-live to enable.")
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["--model", model, "Ping"])

    if result.exit_code != 0:
        pytest.fail(result.output)
