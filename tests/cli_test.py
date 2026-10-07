import io
import sys
from types import SimpleNamespace

import pytest
from click.exceptions import BadParameter
from click.testing import CliRunner

from lmterminal import cli, lib


def test_top_level_help_points_to_prompt_controls():
    result = CliRunner().invoke(cli.lmt, ["--help"])
    assert result.exit_code == 0
    assert "lmt prompt --help" in result.output
    assert "--reasoning-effort" in result.output
    assert "model-dependent" in result.output


def test_prompt_help_explains_reasoning_default_and_debug_disclosure():
    result = CliRunner().invoke(cli.lmt, ["prompt", "--help"])
    assert result.exit_code == 0
    output = " ".join(result.output.split())
    assert "none to disable reasoning where supported" in output
    assert "Omitted model/effort use gpt-6-luna with none" in output
    assert "explicit models retain their provider default when effort is omitted" in output
    assert "[none|minimal|low|medium|high|xhigh|max]" in output
    assert "Print prompts, model and requested temperature to stderr" in output


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("gpt-3.5-turbo", "gpt-3.5-turbo"),
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
        None,
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
        ([], "piped input\n", True),
        (["--rich"], "piped input\n", True),
        (["prompt", "--raw", "Summarize"], "piped input\n\n___\nSummarize", True),
        (["--no-stream"], "piped input\n", False),
        (["prompt", "--no-stream", "Summarize"], "piped input\n\n___\nSummarize", False),
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
    assert result.stdout == "Hello\n"
    assert calls == [
        {
            "messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": expected_prompt},
            ],
            "model": lib.DEFAULT_MODEL,
            "reasoning_effort": "none",
            "temperature": 1,
            "n": 1,
            "stream": stream,
        }
    ]


@pytest.mark.parametrize("command", [[], ["prompt"]])
@pytest.mark.parametrize("no_stream", [False, True])
@pytest.mark.parametrize("model, effort", [("5.4", "high"), ("6-luna", "none"), ("6.1-sol", "max")])
def test_prompt_request_controls(monkeypatch, tmp_path, command, no_stream, model, effort):
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(lib, "get_config_path", lambda: tmp_path / "config.json")
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        if kwargs["stream"]:
            return iter(
                [
                    SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="hello"))]
                    ),
                    SimpleNamespace(choices=[]),
                ]
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(lib.openai_utils, "_get_client", lambda _: client)
    args = command + [
        "-m",
        model,
        "--reasoning-effort",
        effort,
        "-o",
        "verbosity=low",
        "-o",
        "max_completion_tokens=100",
    ]
    if no_stream:
        args += ["--no-stream"]
    else:
        args += ["-o", "stream_options.include_usage=true"]
    result = CliRunner().invoke(cli.lmt, args, input="hi")
    assert result.exit_code == 0, result.output
    assert result.stdout == "hello\n"
    call = calls[0]
    assert call["model"] == f"gpt-{model}"
    assert call["reasoning_effort"] == effort
    assert call["verbosity"] == "low"
    assert call["max_completion_tokens"] == 100
    if effort == "none":
        assert call["temperature"] == 1
    else:
        assert "temperature" not in call
    if not no_stream:
        assert call["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("no_stream", [False, True], ids=["stream", "nonstream"])
@pytest.mark.parametrize(
    "text,expected",
    [
        pytest.param("**Hello**", b"**Hello**\n", id="missing-lf"),
        pytest.param("Hello\n", b"Hello\n", id="one-lf"),
        pytest.param("Hello\n\n", b"Hello\n\n", id="two-lfs"),
        pytest.param("Hello \t", b"Hello \t\n", id="whitespace-tail"),
        pytest.param("", b"\n", id="empty"),
        pytest.param(None, b"\n", id="tool-only"),
    ],
)
def test_redirected_response_preserves_text_and_final_lf(
    monkeypatch, tmp_path, no_stream, text, expected
):
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    tool_calls = [{"type": "function", "function": {"name": "lookup", "arguments": "{}"}}]

    def create(**kwargs):
        assert kwargs["stream"] is not no_stream
        message = SimpleNamespace(content=text, tool_calls=tool_calls if text is None else [])
        if kwargs["stream"]:
            return iter(
                [
                    SimpleNamespace(choices=[SimpleNamespace(delta=message)]),
                    SimpleNamespace(choices=[]),
                ]
            )
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(lib.openai_utils, "_get_client", lambda _: client)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    path = tmp_path / "response.txt"
    with path.open("w", encoding="UTF-8") as output:
        monkeypatch.setattr(sys, "stdout", output)
        args = ["--raw", "fixture"] + (["--no-stream"] if no_stream else [])
        cli.lmt.main(args, standalone_mode=False)
        # Read without flushing on behalf of the renderer.
        assert path.read_bytes() == expected


def test_redirected_failed_stream_keeps_partial_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")

    def events():
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))])
        raise RuntimeError("fixture stream failure")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: events()))
    )
    monkeypatch.setattr(lib.openai_utils, "_get_client", lambda _: client)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    path = tmp_path / "response.txt"
    with path.open("w", encoding="UTF-8") as output:
        monkeypatch.setattr(sys, "stdout", output)
        with pytest.raises(SystemExit) as error:
            cli.lmt.main(["fixture"], standalone_mode=False)
        assert error.value.code == 1
        assert path.read_bytes() == b"partial"
    assert "fixture stream failure" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option, message",
    [
        ("stream=true", "reserved"),
        ("n=2", "reserved"),
        ("temperature=0.9", "--temperature"),
        ("reasoning_effort=low", "--reasoning-effort"),
        ("extra_body.stream=true", "reserved"),
        ("verbosity", "key=value"),
        ("stream_options..include_usage=true", "empty path segments"),
    ],
)
def test_option_errors_are_cli_errors(monkeypatch, option, message):
    monkeypatch.setattr(lib, "get_api_key", lambda: pytest.fail("Must not read a key"))
    result = CliRunner().invoke(cli.lmt, ["-o", option], input="hi")
    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize("temperature", ["0.3", "1"])
def test_cli_rejects_incompatible_sampling_before_key_read(monkeypatch, temperature):
    monkeypatch.setattr(lib, "get_api_key", lambda: pytest.fail("Must not read a key"))
    result = CliRunner().invoke(
        cli.lmt, ["-m", "gpt-5-nano", "--temperature", temperature], input="hi"
    )
    assert result.exit_code == 2
    assert "Temperature is not supported" in result.output


@pytest.mark.parametrize(
    "template_model, args, default_map, model, controls",
    [
        ("gpt-4o", [], None, "gpt-4o", {"temperature": 1}),
        ("gpt-4o", ["-m", "gpt-6-luna"], None, "gpt-6-luna", {}),
        ("luna", [], None, "gpt-6-luna", {}),
        ("gpt-5.6", [], None, "gpt-5.6-sol", {}),
        (
            "gpt-4o",
            [],
            {"prompt": {"temperature": 0.7}},
            "gpt-4o",
            {"temperature": 0.7},
        ),
        (None, [], None, "gpt-6-luna", {"reasoning_effort": "none", "temperature": 1}),
        (None, ["--reasoning-effort", "high"], None, "gpt-6-luna", {"reasoning_effort": "high"}),
        (
            "gpt-4o",
            [],
            {"prompt": {"model": "luna", "reasoning_effort": "low"}},
            "gpt-6-luna",
            {"reasoning_effort": "low"},
        ),
    ],
)
def test_template_and_explicit_choices_preserve_defaults_provenance(
    monkeypatch, tmp_path, template_model, args, default_map, model, controls
):
    from lmterminal import templates

    (tmp_path / "fixture.yaml").write_text(
        f'system: "Reply concisely. "\nprompt: "Translate: "\nmodel: {template_model or "null"}\n',
        encoding="UTF-8",
    )
    monkeypatch.setattr(templates, "TEMPLATES_DIR", tmp_path)
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    monkeypatch.setattr(lib.openai_utils, "_get_client", lambda _: client)
    result = CliRunner().invoke(
        cli.lmt,
        ["--template", "fixture", "--no-stream", *args],
        input="hi",
        default_map=default_map,
    )
    assert result.exit_code == 0, result.output
    assert calls == [
        dict(
            messages=[
                {"role": "system", "content": "Reply concisely. "},
                {"role": "user", "content": "hi\n___\nTranslate: "},
            ],
            model=model,
            n=1,
            stream=False,
            **controls,
        )
    ]


@pytest.mark.parametrize(
    "model, effort",
    [("6-luna", "minimal"), ("6.1-sol", "none"), ("6-astra", "none")],
)
def test_cli_current_effort_errors_before_key_read(monkeypatch, model, effort):
    monkeypatch.setattr(lib, "get_api_key", lambda: pytest.fail("Must not read a key"))
    result = CliRunner().invoke(cli.lmt, ["-m", model, "--reasoning-effort", effort, "hello"])
    assert result.exit_code == 2
    assert f"Reasoning effort `{effort}` is not supported" in result.output
    assert "Use --reasoning-effort with one of:" in result.output


def test_cli_max_uses_shared_policy(monkeypatch):
    monkeypatch.setattr(lib, "get_api_key", lambda: pytest.fail("Must not read a key"))
    result = CliRunner().invoke(
        cli.lmt, ["-m", "6-luna", "--reasoning-effort", "max", "--temperature", "0.3", "hello"]
    )
    assert result.exit_code == 2
    assert "Temperature is not supported for `gpt-6-luna`" in result.output


@pytest.mark.parametrize(
    "values",
    [
        ("verbosity=low", "verbosity=high"),
        ("response_format=text", "response_format.type=json_object"),
    ],
)
def test_option_duplicates_and_conflicts(values):
    with pytest.raises(BadParameter):
        cli.parse_request_options(None, None, values)


@pytest.mark.integration
@pytest.mark.parametrize("model", list(cli.get_valid_models()), ids=str)
def test_live_call(model, request):
    if not request.config.getoption("--run-live"):
        pytest.skip("Skipping live call test. Use --run-live to enable.")
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["--model", model, "Ping"])

    if result.exit_code != 0:
        pytest.fail(result.output)


def test_conditional_temperature_help():
    result = CliRunner().invoke(cli.lmt, ["prompt", "--help"])
    assert result.exit_code == 0
    assert "defaults to 1 where supported" in " ".join(result.output.split())
    assert "unverified" in result.output
    assert "[default: 1]" not in result.output


@pytest.mark.parametrize("provider_rejects", [False, True])
def test_unverified_cli_and_library_controls_reach_provider(monkeypatch, provider_rejects):
    import httpx
    import openai

    from lmterminal import gpt_integration

    calls = []
    error = openai.BadRequestError(
        "sampling rejected",
        response=httpx.Response(400, request=httpx.Request("POST", "https://example.invalid")),
        body={"error": {"message": "sampling rejected"}},
    )

    def create(**kwargs):
        calls.append(kwargs)
        if provider_rejects:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(gpt_integration, "_get_client", lambda _: client)
    monkeypatch.setattr(lib, "get_api_key", lambda: "test-key")
    cli_result = CliRunner().invoke(
        cli.lmt,
        [
            "-m",
            "5.6",
            "--no-stream",
            "--temperature",
            "1",
            "--reasoning-effort",
            "none",
            "-o",
            "top_p=0.9",
            "-o",
            "extra_body.logprobs=true",
        ],
        input="hello",
    )
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "hello"}]
    options = {"top_p": 0.9, "extra_body": {"logprobs": True}}
    if provider_rejects:
        assert cli_result.exit_code == 1
        assert "sampling rejected" in cli_result.output
        with pytest.raises(openai.BadRequestError) as raised:
            gpt_integration.chatgpt_request(
                "test-key",
                messages,
                "gpt-5.6",
                temperature=1,
                reasoning_effort="none",
                request_options=options,
            )
        assert raised.value is error
    else:
        assert cli_result.exit_code == 0, cli_result.output
        assert cli_result.stdout == "pong\n"
        assert (
            gpt_integration.chatgpt_request(
                "test-key",
                messages,
                "gpt-5.6",
                temperature=1,
                reasoning_effort="none",
                request_options=options,
            )[0]
            == "pong"
        )
    assert (
        calls
        == [
            dict(
                messages=messages,
                model="gpt-5.6-sol",
                n=1,
                stream=False,
                temperature=1,
                reasoning_effort="none",
                **options,
            )
        ]
        * 2
    )
