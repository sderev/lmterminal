import io
import json
import sys
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from pygments.token import Keyword
from rich.console import Console

from lmterminal import cli, templates
from lmterminal.templates import TemplateError, load_template


@pytest.fixture(autouse=True)
def isolated_view(no_provider_or_download, monkeypatch, tmp_path):
    monkeypatch.setattr(templates, "TEMPLATES_DIR", tmp_path)


def test_saved_mapping_output_preserves_values_and_order(monkeypatch, tmp_path):
    source = """# SYNTHETIC_HEADER
text: |2-
   indented [bold]literal[/bold] `code` # data

  Bonjour, Sébastien.
prompt: |+
  Task

system: >-
  Folded
  instructions.
model: 4o # SYNTHETIC_INLINE
temperature: 0
request_options:
  metadata: {flag: false, count: 0, quoted: "true"}
  stops: [one, two]
"""
    path = tmp_path / "demo.yaml"
    path.write_text(source, encoding="UTF-8")
    monkeypatch.setattr(
        cli, "get_markdown_code_block_theme", lambda: pytest.fail("plain output read theme")
    )
    result = CliRunner().invoke(cli.lmt, ["templates", "view", "demo"], terminal_width=20)
    assert result.exit_code == 0, result.output
    expected = yaml.safe_load(source)
    actual = yaml.safe_load(result.stdout)
    assert actual == expected
    assert list(actual) == list(expected)
    assert actual["request_options"]["metadata"] == {"flag": False, "count": 0, "quoted": "true"}
    assert actual["request_options"]["metadata"]["flag"] is False
    assert type(actual["request_options"]["metadata"]["count"]) is int
    assert type(actual["temperature"]) is int
    assert "SYNTHETIC" not in result.stdout
    assert "\x1b" not in result.stdout
    assert "text: |2-\n   indented [bold]literal[/bold] `code` # data\n" in result.stdout
    assert "system: Folded instructions.\n" in result.stdout
    assert path.read_text(encoding="UTF-8") == source
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "source,expected",
    [
        ('prompt: "A\\u0085B"\n', "A\x85B"),
        ('prompt: "A\\u0085B\\n"\n', "A\x85B\n"),
    ],
)
def test_next_line_character_preserves_parsed_value(tmp_path, source, expected):
    (tmp_path / "demo.yaml").write_text(source, encoding="UTF-8")
    result = CliRunner().invoke(cli.lmt, ["templates", "view", "demo"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout) == {"prompt": expected}
    assert "\x85" not in result.stdout


@pytest.mark.parametrize(
    "source,expected",
    [
        ("prompt: Task\n", {"prompt": "Task"}),
        (
            "prompt: null\ntext: ''\nrequest_options: {}\n",
            {"prompt": None, "text": "", "request_options": {}},
        ),
        ("{}\n", {}),
        ("user: old\nprompt: 3\nunknown: false\n", {"user": "old", "prompt": 3, "unknown": False}),
        ('prompt: " leading \\n\\n trailing \\n"\n', {"prompt": " leading \n\n trailing \n"}),
    ],
)
def test_sparse_null_and_invalid_schema_mappings(tmp_path, source, expected):
    (tmp_path / "demo.yaml").write_text(source, encoding="UTF-8")
    result = CliRunner().invoke(cli.lmt, ["templates", "view", "demo"])
    assert result.exit_code == 0, result.output
    assert yaml.safe_load(result.stdout) == expected
    assert list(yaml.safe_load(result.stdout)) == list(expected)
    if "user" in expected:
        with pytest.raises(TemplateError, match="replace `user`"):
            load_template("demo")


@pytest.mark.parametrize(
    "source,error",
    [
        (None, "Cannot read"),
        (b"\xffSYNTHETIC", "Cannot read"),
        (b"prompt: [SYNTHETIC", "invalid YAML"),
        (b"[SYNTHETIC]", "mapping"),
        (b"SYNTHETIC", "mapping"),
        (b"null", "mapping"),
        (b"", "mapping"),
    ],
)
def test_view_errors_without_partial_output_or_contents(tmp_path, source, error):
    if source is not None:
        (tmp_path / "demo.yaml").write_bytes(source)
    result = CliRunner().invoke(cli.lmt, ["templates", "view", "demo"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert error in result.stderr
    assert "SYNTHETIC" not in result.stderr


@pytest.mark.parametrize("timestamp", ["2026-13-01", "2026-10-07T25:00:00Z"])
def test_invalid_timestamp_returns_content_free_cli_error(tmp_path, timestamp):
    (tmp_path / "demo.yaml").write_text(f"prompt: {timestamp}\ntext: SYNTHETIC\n", encoding="UTF-8")
    result = CliRunner().invoke(cli.lmt, ["templates", "view", "demo"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: Template `demo` contains invalid YAML.\n"


def test_unreadable_file_error(monkeypatch, tmp_path):
    def unreadable(*args, **kwargs):
        raise PermissionError("SYNTHETIC")

    monkeypatch.setattr(Path, "read_text", unreadable)
    result = CliRunner().invoke(cli.lmt, ["templates", "view", "demo"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr == "Error: Cannot read template `demo`.\n"


@pytest.mark.parametrize("theme_name", ["alabaster", "missing-lmt-style"])
def test_terminal_theme_and_plain_bypass(monkeypatch, tmp_path, theme_name):
    (tmp_path / "demo.yaml").write_text("prompt: '[bold]literal[/bold]'\n", encoding="UTF-8")
    config = tmp_path / ".config" / "lmt" / "config.json"
    config.parent.mkdir(parents=True)
    config_source = json.dumps({"code_block_theme": theme_name})
    config.write_text(config_source, encoding="UTF-8")
    captured = []
    resolve = cli.resolve_code_theme

    def capture_theme(name):
        theme = resolve(name)
        captured.append((name, theme))
        return theme

    monkeypatch.setattr(cli, "resolve_code_theme", capture_theme)
    output = io.StringIO()
    console = Console(file=output, force_terminal=True, color_system="truecolor", width=80)
    monkeypatch.setattr(cli, "Console", lambda: console)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    if theme_name == "alabaster":
        cli.view_template.callback("demo")
        assert captured[0][0] == "alabaster"
        assert captured[0][1].get_style_for_token(Keyword).color.triplet == (122, 62, 157)
        rendered = output.getvalue()
        assert "38;2;0;122;204" in rendered  # YAML key uses the Alabaster Name.Tag color.
        assert "[bold]literal[/bold]" in cli.click.unstyle(rendered)
        assert cli.click.unstyle(rendered).splitlines()[0].startswith("prompt:")
    else:
        with pytest.raises(cli.click.ClickException, match="theme is unavailable") as error:
            cli.view_template.callback("demo")
        assert "redirect stdout" in str(error.value)
        assert output.getvalue() == ""
    captured.clear()
    result = CliRunner().invoke(cli.lmt, ["templates", "view", "demo"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "prompt: '[bold]literal[/bold]'\n"
    assert captured == []
    assert config.read_text(encoding="UTF-8") == config_source
