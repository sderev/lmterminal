import io
import sys
import traceback
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from lmterminal import cli, cli_output, templates
from lmterminal.request_options import DEFAULT_MODEL
from lmterminal.resolution import RequestResolutionError, SystemTemplateConflict, resolve_request
from lmterminal.templates import Template, TemplateError, load_template


@pytest.mark.parametrize(
    "name,task,content,extra,expected",
    [
        (
            "Translate",
            "Translate into English.",
            "Bonjour.",
            "Keep product names unchanged.",
            "Bonjour.\n___\nTranslate into English.\n\nKeep product names unchanged.",
        ),
        (
            "CommitGen",
            "Write a commit message.",
            " DIFF\n",
            "Mention tests.",
            " DIFF\n\n___\nWrite a commit message.\n\nMention tests.",
        ),
    ],
)
def test_cli_library_exact_messages(monkeypatch, tmp_path, name, task, content, extra, expected):
    data = {"system": "Reply concisely.", "prompt": task, "model": "4o", "temperature": 0.7}
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data), encoding="UTF-8")
    monkeypatch.setattr(cli, "_template_directory", lambda: tmp_path)
    loaded = load_template(name, tmp_path)
    request = resolve_request(template=loaded, prompt=extra, text=content)
    assert request == resolve_request(template=Template(**data), prompt=extra, text=content)
    assert request.messages == [
        {"role": "system", "content": "Reply concisely."},
        {"role": "user", "content": expected},
    ]
    assert request.controls == {"temperature": 0.7}
    captured = []
    monkeypatch.setattr(
        cli,
        "execute_request",
        lambda request, *args, **kwargs: captured.append(request),
    )
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["-t", name, extra], input=content)
    assert result.exit_code == 0, result.output
    result = runner.invoke(cli.lmt, ["-t", name, "--text", content, extra])
    assert result.exit_code == 0, result.output
    assert captured == [request, request]


def test_scalar_precedence_and_options_replacement():
    template = Template(
        model="4o",
        temperature=0.7,
        reasoning_effort="none",
        request_options={"metadata": {"stored": "yes"}, "max_completion_tokens": 100},
    )
    # Template settings are authoritative, even with explicit model selection.
    selected = resolve_request(template=template, model=DEFAULT_MODEL)
    assert selected.model == DEFAULT_MODEL
    assert selected.controls["reasoning_effort"] == "none"
    assert selected.controls["temperature"] == 0.7
    assert resolve_request(template=template, temperature=0.3).controls["temperature"] == 0.3
    inherited = resolve_request(template=template)
    assert inherited.model == "gpt-4o"
    assert inherited.controls["temperature"] == 0.7
    explicit = resolve_request(
        template=template,
        model=DEFAULT_MODEL,
        temperature=None,
        reasoning_effort=None,
        request_options={"metadata": {"invocation": "yes"}},
    )
    assert explicit.controls == {"metadata": {"invocation": "yes"}, "max_completion_tokens": 100}
    assert resolve_request().controls == {"reasoning_effort": "none", "temperature": 1}
    assert resolve_request(model=DEFAULT_MODEL).controls == {}
    assert resolve_request(reasoning_effort=None).controls == {}
    assert resolve_request(template=Template(model="gpt-5-nano"), model="4o").model == "gpt-4o"


@pytest.mark.parametrize(
    "settings,args,model,controls",
    [
        ({}, [], "gpt-6-luna", {"reasoning_effort": "none", "temperature": 1}),
        ({"temperature": None}, [], "gpt-6-luna", {"reasoning_effort": "none", "temperature": 1}),
        ({"model": "luna"}, [], "gpt-6-luna", {}),
        ({"model": "5.6"}, [], "gpt-5.6-sol", {}),
        ({"model": "5.6", "temperature": 1}, [], "gpt-5.6-sol", {"temperature": 1}),
        ({"model": "5-nano"}, [], "gpt-5-nano", {}),
        (
            {"model": "luna", "temperature": 1},
            ["--reasoning-effort", "none", "--temperature", "0"],
            "gpt-6-luna",
            {"reasoning_effort": "none", "temperature": 0},
        ),
    ],
)
def test_catalog_temperature_provenance_matches_cli(
    monkeypatch, tmp_path, settings, args, model, controls
):
    data = {"prompt": "Task", **settings}
    (tmp_path / "fixture.yaml").write_text(yaml.safe_dump(data), encoding="UTF-8")
    monkeypatch.setattr(cli, "_template_directory", lambda: tmp_path)
    overrides = {"reasoning_effort": "none", "temperature": 0} if args else {}
    request = resolve_request(template=load_template("fixture", tmp_path), **overrides)
    assert request == resolve_request(template=Template(**data), **overrides)
    assert request.model == model
    assert request.controls == controls
    captured = []
    monkeypatch.setattr(cli, "execute_request", lambda request, **kw: captured.append(request))
    result = CliRunner().invoke(cli.lmt, ["-t", "fixture", *args])
    assert result.exit_code == 0, result.output
    assert captured == [request]


def test_explicit_none_temperature_overrides_stored_incompatible_value():
    template = Template(model="luna", temperature=1, prompt="Task")
    with pytest.raises(RequestResolutionError, match="Temperature is not supported"):
        resolve_request(template=template)
    assert resolve_request(template=template, temperature=None).controls == {}
    assert resolve_request(temperature=None).controls == {"reasoning_effort": "none"}
    assert (
        resolve_request(template=Template(model="4o", temperature=0.7), temperature=None).controls
        == {}
    )


@pytest.mark.parametrize("setting", ["temperature: 1", "request_options: {top_p: 0.9}"])
def test_template_sampling_validation_before_stdin_and_key(monkeypatch, tmp_path, setting):
    (tmp_path / "fixture.yaml").write_text(f"model: luna\n{setting}\n", encoding="UTF-8")
    monkeypatch.setattr(cli, "_template_directory", lambda: tmp_path)
    monkeypatch.setattr(cli_output, "read_api_key", lambda _path: pytest.fail("key read"))
    runner = CliRunner()
    with runner.isolation():
        monkeypatch.setattr(sys, "stdin", UnreadableStdin())
        with pytest.raises(cli.click.UsageError, match="not supported"):
            cli.lmt.main(["-t", "fixture"], standalone_mode=False)


def test_stored_content_and_empty_components_preserve_whitespace():
    request = resolve_request(template=Template(prompt="Task ", text=" Context\n"), text=" Body ")
    assert request.messages[-1]["content"] == " Context\n\n\n Body \n___\nTask "
    assert resolve_request(template=Template(prompt="Task ")).messages[-1]["content"] == "Task "
    assert resolve_request(text=" Body\n").messages[-1]["content"] == " Body\n"


@pytest.mark.parametrize("system", ["", None, "Override"])
def test_system_presence_conflicts_in_library(system):
    with pytest.raises(SystemTemplateConflict):
        resolve_request(template=Template(), system=system)


class UnreadableStdin(io.StringIO):
    def isatty(self):
        return True

    def read(self, *args):
        pytest.fail("stdin read before validation or for a template-only task")


@pytest.mark.parametrize("failure", ["system", "missing", "malformed", "reserved"])
def test_cli_errors_before_stdin_and_key(monkeypatch, tmp_path, failure):
    (tmp_path / "fixture.yaml").write_text("prompt: Task\n", encoding="UTF-8")
    if failure == "malformed":
        (tmp_path / "fixture.yaml").write_text("prompt: [PRIVATE\n", encoding="UTF-8")
    if failure == "reserved":
        (tmp_path / "fixture.yaml").write_text(
            "request_options: {extra_body: {model: PRIVATE}}\n", encoding="UTF-8"
        )
    monkeypatch.setattr(cli, "_template_directory", lambda: tmp_path)
    monkeypatch.setattr(cli_output, "read_api_key", lambda _path: pytest.fail("key read"))
    runner = CliRunner()
    with runner.isolation():
        monkeypatch.setattr(sys, "stdin", UnreadableStdin())
        args = ["-t", "missing" if failure == "missing" else "fixture"]
        if failure == "system":
            args += ["--system", ""]
        with pytest.raises(cli.click.UsageError) as error:
            cli.lmt.main(args, standalone_mode=False)
    assert "PRIVATE" not in str(error.value)


def test_template_task_does_not_read_interactive_stdin(monkeypatch, tmp_path):
    (tmp_path / "task.yaml").write_text("prompt: Task\n", encoding="UTF-8")
    monkeypatch.setattr(cli, "_template_directory", lambda: tmp_path)
    requests = []
    monkeypatch.setattr(cli, "execute_request", lambda request, **kw: requests.append(request))
    runner = CliRunner()
    with runner.isolation():
        monkeypatch.setattr(sys, "stdin", UnreadableStdin())
        cli.lmt.main(["-t", "task"], standalone_mode=False)
    assert requests[0].messages[-1]["content"] == "Task"


def test_literal_text_and_stdin_conflict_before_key(monkeypatch):
    monkeypatch.setattr(cli_output, "read_api_key", lambda _path: pytest.fail("key read"))
    result = CliRunner().invoke(cli.lmt, ["--text", "literal"], input="piped")
    assert result.exit_code == 2
    assert "nonempty stdin" in result.output


@pytest.mark.parametrize(
    "content,field",
    [
        ("[PRIVATE]", "mapping"),
        ("prompt: 3", "prompt"),
        ("temperature: true", "temperature"),
        ("model: 3", "model"),
        ("reasoning_effort: false", "reasoning_effort"),
        ("request_options: []", "request_options"),
        ("user: PRIVATE", "replace `user`"),
        ("unknown: PRIVATE", "unsupported field"),
    ],
)
def test_template_schema_errors_have_context_without_contents(tmp_path, content, field):
    (tmp_path / "bad.yaml").write_text(content, encoding="UTF-8")
    with pytest.raises(TemplateError) as error:
        load_template("bad", tmp_path)
    assert "bad" in str(error.value)
    assert field in str(error.value)
    assert "PRIVATE" not in str(error.value)


def test_empty_and_null_yaml_and_missing_directory(tmp_path):
    directory = tmp_path / "absent"
    assert templates.list_templates(directory) == []
    with pytest.raises(TemplateError):
        load_template("missing", directory)
    assert not directory.exists()
    (tmp_path / "empty.yaml").write_text("", encoding="UTF-8")
    (tmp_path / "null.yaml").write_text(
        "prompt: null\ntext: null\nsystem: null\nmodel: null\ntemperature: null\nreasoning_effort: null\n",
        encoding="UTF-8",
    )
    assert load_template("empty", tmp_path) == load_template("null", tmp_path) == Template()


def test_request_options_use_shared_reserved_validation():
    with pytest.raises(TemplateError, match="reserved"):
        Template(request_options={"extra_body": {"messages": []}})
    with pytest.raises(RequestResolutionError, match="reserved"):
        resolve_request(request_options={"model": "4o"})


def test_template_management_refuses_overwrite_and_renames_basename(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_template_directory", lambda ctx=None: tmp_path)
    monkeypatch.setattr(cli.click, "edit", lambda **kwargs: pytest.fail("existing template edited"))
    old = tmp_path / "old.yaml"
    old.write_text("prompt: Keep\n", encoding="UTF-8")
    occupied = tmp_path / "occupied.yaml"
    occupied.write_text("prompt: Existing\n", encoding="UTF-8")
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["templates", "add", "old"])
    assert result.exit_code == 2
    result = runner.invoke(cli.lmt, ["templates", "rename", "old"], input="occupied\n")
    assert result.exit_code == 2
    assert old.read_text() == "prompt: Keep\n"
    assert occupied.read_text() == "prompt: Existing\n"
    result = runner.invoke(cli.lmt, ["templates", "rename", "old"], input="new\n")
    assert result.exit_code == 0, result.output
    assert not old.exists()
    assert (tmp_path / "new.yaml").read_text() == "prompt: Keep\n"
    (tmp_path / "ignore.txt").touch()
    (tmp_path / "directory.yaml").mkdir()
    assert templates.list_templates(tmp_path) == ["new", "occupied"]
    assert cli.complete_template(None, None, "n") == ["new"]
    with pytest.raises(TemplateError):
        templates.template_path("../outside", tmp_path)
    with pytest.raises(TemplateError):
        templates.template_path("new.yaml", tmp_path)


@pytest.mark.parametrize("group_name", ["template", "templates"])
def test_template_management_alias_routes_list_and_view(
    no_provider_or_download, monkeypatch, tmp_path, group_name
):
    monkeypatch.setattr(cli, "_template_directory", lambda: tmp_path)
    monkeypatch.setattr(
        cli,
        "execute_request",
        lambda *args, **kwargs: pytest.fail("Template management must not generate a response"),
    )
    (tmp_path / "translate.yaml").write_text("prompt: Translate\n", encoding="UTF-8")
    (tmp_path / "transcribe.yaml").write_text("prompt: Transcribe\n", encoding="UTF-8")
    (tmp_path / "ignore.txt").touch()
    runner = CliRunner()
    listed = runner.invoke(cli.lmt, [group_name, "list"])
    assert listed.exit_code == 0, listed.output
    assert listed.output == "transcribe\ntranslate\n"
    viewed = runner.invoke(cli.lmt, [group_name, "view", "translate"])
    assert viewed.exit_code == 0, viewed.output
    assert viewed.output.strip() == "prompt: Translate"


@pytest.mark.parametrize("group_name", ["template", "templates"])
@pytest.mark.parametrize("args", [["--help"], [], ["invalid-command"]])
def test_template_management_alias_help_and_errors(no_provider_or_download, group_name, args):
    result = CliRunner().invoke(cli.lmt, [group_name, *args])
    assert result.exit_code == (0 if args == ["--help"] else 2), result.output
    assert f"Usage: lmt {group_name}" in result.output
    if args == ["invalid-command"]:
        assert "No such command 'invalid-command'" in result.output
    else:
        assert "Manage the templates." in result.output
        for command in ("add", "delete", "edit", "list", "rename", "view"):
            assert f"  {command} " in result.output


def test_template_management_alias_in_root_help(no_provider_or_download):
    result = CliRunner().invoke(cli.lmt, ["--help"])
    assert result.exit_code == 0
    for group_name in ("template", "templates"):
        assert f"  {group_name} " in result.output


def test_explicit_prompt_accepts_template_command_word(no_provider_or_download, monkeypatch):
    requests = []
    monkeypatch.setattr(cli, "execute_request", lambda request, **kwargs: requests.append(request))
    result = CliRunner().invoke(cli.lmt, ["prompt", "template", "list"])
    assert result.exit_code == 0, result.output
    assert len(requests) == 1
    assert requests[0].messages[-1]["content"] == "template list"


@pytest.mark.parametrize("group_name", ["template", "templates"])
def test_template_management_alias_bash_completion(no_provider_or_download, tmp_path, group_name):
    directory = tmp_path / ".config" / "lmt" / "templates"
    directory.mkdir(parents=True)
    for name in ("translate.yaml", "transcribe.yaml", "summarize.yaml", "trans.txt"):
        (directory / name).write_text("prompt: Task\n", encoding="UTF-8")
    (directory / "trans-directory.yaml").mkdir()
    runner = CliRunner()
    for words, expected in [
        (f"lmt {group_name} ", ["add", "delete", "edit", "list", "rename", "view"]),
        *[
            (f"lmt {group_name} {command} trans", ["transcribe", "translate"])
            for command in ("view", "edit", "delete", "rename")
        ],
    ]:
        result = runner.invoke(
            cli.lmt,
            [],
            prog_name="lmt",
            env={
                "_LMT_COMPLETE": "bash_complete",
                "COMP_WORDS": words,
                "COMP_CWORD": "2" if words.endswith(" ") else "3",
            },
        )
        assert result.exit_code == 0, result.output
        assert result.output.splitlines() == [f"plain,{value}" for value in expected]


@pytest.mark.parametrize("command", ["lmt", "lmt prompt"])
def test_prompt_template_bash_completion(no_provider_or_download, tmp_path, command):
    directory = tmp_path / ".config" / "lmt" / "templates"
    directory.mkdir(parents=True)
    (directory / "translate.yaml").write_text("prompt: Task\n", encoding="UTF-8")
    result = CliRunner().invoke(
        cli.lmt,
        [],
        prog_name="lmt",
        env={
            "_LMT_COMPLETE": "bash_complete",
            "COMP_WORDS": f"{command} -t trans",
            "COMP_CWORD": str(len(command.split()) + 1),
        },
    )
    assert result.exit_code == 0, result.output
    assert result.output == "plain,translate\n"


@pytest.mark.parametrize("command_name", ["view", "edit", "delete", "rename"])
def test_template_management_argument_completion(no_provider_or_download, tmp_path, command_name):
    directory = tmp_path / ".config" / "lmt" / "templates"
    directory.mkdir(parents=True)
    for name in ("translate.yaml", "transcribe.yaml", "summarize.yaml", "trans.txt", "trans.yml"):
        (directory / name).write_text("prompt: Task\n", encoding="UTF-8")
    (directory / "trans-directory.yaml").mkdir()
    command = cli.templates.commands[command_name]
    argument = next(param for param in command.params if param.name == "template")
    ctx = cli.click.Context(command)
    assert [item.value for item in argument.shell_complete(ctx, "trans")] == [
        "transcribe",
        "translate",
    ]


def test_malformed_yaml_traceback_does_not_reveal_content(tmp_path):
    (tmp_path / "bad.yaml").write_text("prompt: [PRIVATE\n", encoding="UTF-8")
    with pytest.raises(TemplateError) as error:
        load_template("bad", tmp_path)
    assert "PRIVATE" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("saved", [True, False], ids=["saved", "unchanged"])
def test_template_add_saved_or_unchanged(no_provider_or_download, monkeypatch, tmp_path, saved):
    directory = tmp_path / ".config" / "lmt" / "templates"
    monkeypatch.setattr(cli, "_template_directory", lambda: directory)
    assert not directory.exists()
    content = "prompt: Translate into French.\n"

    def edit_draft(*, filename):
        if saved:
            Path(filename).write_text(content, encoding="UTF-8")

    monkeypatch.setattr(cli.click, "edit", edit_draft)
    result = CliRunner().invoke(cli.lmt, ["templates", "add", "translate"])

    assert result.exit_code == 0, result.output
    template_file = directory / "translate.yaml"
    if saved:
        assert "created" in result.output
        assert template_file.read_bytes() == content.encode()
        assert load_template("translate", directory).prompt == "Translate into French."
    else:
        assert "no changes were made" in result.output
        assert not template_file.exists()


@pytest.mark.parametrize("saved", [True, False], ids=["saved", "unchanged"])
def test_template_edit_saved_or_unchanged(no_provider_or_download, monkeypatch, tmp_path, saved):
    directory = tmp_path / ".config" / "lmt" / "templates"
    directory.mkdir(parents=True)
    monkeypatch.setattr(cli, "_template_directory", lambda: directory)
    template_file = directory / "translate.yaml"
    original = "# Café\r\nprompt: Translate into English.\r\n".encode()
    template_file.write_bytes(original)
    other = directory / "other.yaml"
    other_content = b"prompt: Keep this template.\n"
    other.write_bytes(other_content)
    content = "prompt: Translate into French.\n"

    def edit_template(*, filename):
        if saved:
            Path(filename).write_text(content, encoding="UTF-8")

    monkeypatch.setattr(cli.click, "edit", edit_template)
    result = CliRunner().invoke(cli.lmt, ["templates", "edit", "translate"])

    assert result.exit_code == 0, result.output
    assert other.read_bytes() == other_content
    if saved:
        assert "was updated" in result.output
        assert template_file.read_bytes() == content.encode()
        assert load_template("translate", directory).prompt == "Translate into French."
    else:
        assert "No changes were made" in result.output
        assert template_file.read_bytes() == original
        assert load_template("translate", directory).prompt == "Translate into English."


@pytest.mark.parametrize("confirmed", [True, False], ids=["confirmed", "declined"])
def test_template_delete_confirmation(no_provider_or_download, monkeypatch, tmp_path, confirmed):
    directory = tmp_path / ".config" / "lmt" / "templates"
    directory.mkdir(parents=True)
    monkeypatch.setattr(cli, "_template_directory", lambda: directory)
    monkeypatch.setattr(cli.click, "edit", lambda **kwargs: pytest.fail("unexpected editor"))
    template_file = directory / "translate.yaml"
    original = "# Café\r\nprompt: Translate into English.\r\n".encode()
    template_file.write_bytes(original)
    other = directory / "other.yaml"
    other_content = b"prompt: Keep this template.\n"
    other.write_bytes(other_content)

    result = CliRunner().invoke(
        cli.lmt, ["templates", "delete", "translate"], input="y\n" if confirmed else "n\n"
    )

    assert other.read_bytes() == other_content
    if confirmed:
        assert result.exit_code == 0, result.output
        assert "deleted" in result.output
        assert not template_file.exists()
    else:
        assert result.exit_code == 1, result.output
        assert "Aborted" in result.output
        assert template_file.read_bytes() == original
        assert load_template("translate", directory).prompt == "Translate into English."


@pytest.mark.parametrize("timestamp", ["2026-13-01", "2026-10-07T25:00:00Z"])
def test_execution_yaml_constructor_errors_are_normalized(monkeypatch, tmp_path, timestamp):
    (tmp_path / "bad.yaml").write_text(f"prompt: {timestamp}\ntext: SYNTHETIC\n")
    monkeypatch.setattr(cli, "_template_directory", lambda: tmp_path)
    with pytest.raises(TemplateError, match="contains invalid YAML"):
        load_template("bad", tmp_path)
    with CliRunner().isolation():
        monkeypatch.setattr(sys, "stdin", UnreadableStdin())
        with pytest.raises(cli.click.UsageError, match="contains invalid YAML") as error:
            cli.lmt.main(["-t", "bad"], standalone_mode=False)
    assert "SYNTHETIC" not in str(error.value)


@pytest.mark.parametrize("model", ["o1", "o1-2024-12-17", "o3"])
def test_composed_selected_model_message_policy_leaves_raw_roles_untouched(model):
    from lmterminal.request_options import prepare_request

    composed = resolve_request(model=model, system="System", prompt="Task", emoji=True)
    raw_messages = [{"role": "system", "content": "System"}, {"role": "user", "content": "Task"}]
    raw = prepare_request(model, raw_messages)
    assert raw.messages == raw_messages
    if model.startswith("o1"):
        assert composed.messages == [{"role": "user", "content": "Task"}]
    else:
        assert composed.messages[0]["content"].startswith("System. Add plenty")
        assert composed.messages[1] == raw_messages[1]


def test_resolution_import_and_execution_need_no_application_resources():
    import subprocess

    source = """
import importlib.abc
import pathlib
import socket
import sys

def forbidden(*args, **kwargs):
    raise AssertionError("No resources during resolution")

class BoundaryGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        assert fullname.split('.')[0] not in {'click', 'rich', 'openai'}
        assert fullname not in {'lmterminal.cli', 'lmterminal.cli_output', 'lmterminal.gpt_integration', 'lmterminal.storage'}

sys.meta_path.insert(0, BoundaryGuard())
pathlib.Path.home = forbidden
pathlib.Path.read_text = forbidden
socket.socket.connect = forbidden
from lmterminal.resolution import resolve_request
from lmterminal.templates import Template
assert resolve_request(template=Template(prompt="Task"), text="Content").messages[-1]['content'] == "Content\\n___\\nTask"
"""
    result = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
