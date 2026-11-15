import os
import subprocess
import sys
from pathlib import Path

import pytest

# A fresh interpreter matters: the ordinary test guards import and patch the SDK.
GUARDED_CLI = """
import importlib.abc
import pathlib
import socket
import sys

sys.path.insert(0, sys.argv.pop(1))

class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] == 'openai':
            raise AssertionError('Provider-free commands must not import the SDK')

def forbidden(*args, **kwargs):
    raise AssertionError('Provider-free commands must not read keys or use the network')

sys.meta_path.insert(0, NoSDK())
socket.socket.connect = forbidden
socket.create_connection = forbidden
from lmterminal import cli, cli_output
cli.read_api_key = cli_output.read_api_key = forbidden
try:
    cli.lmt.main(prog_name='lmt')
finally:
    assert 'openai' not in sys.modules
"""


def run_local_cli(tmp_path, arguments, **environment):
    directory = tmp_path / ".config/lmt/templates"
    directory.mkdir(parents=True)
    (directory / "translate.yaml").write_text("prompt: Translate.\n", encoding="UTF-8")
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            GUARDED_CLI,
            str(Path(__file__).resolve().parents[1]),
            *arguments,
        ],
        env={
            "HOME": str(tmp_path),
            "PATH": os.defpath,
            "TERM": "dumb",
            "EDITOR": "true",
            **environment,
        },
        input=b"y\n",
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["--help"], b"Usage: lmt"),
        (["--version"], b"lmt, version"),
        (["models"], b"gpt-4o"),
        (["config", "path"], b"/.config/lmt/config.json\n"),
        (["key", "path"], b"/.config/lmt/keys.json\n"),
        (["templates", "path"], b"/.config/lmt/templates\n"),
        (["template", "list"], b"translate"),
        (["templates", "view", "translate"], b"prompt: Translate.\n"),
        (["template", "path"], b"/.config/lmt/templates\n"),
        (["templates", "add", "draft"], b"no changes were made"),
        (["templates", "edit", "translate"], b"No changes were made."),
        (["templates", "rename", "translate"], b"Success!"),
        (["templates", "delete", "translate"], b"Success!"),
    ],
)
def test_local_commands_do_not_import_sdk_or_read_keys(tmp_path, arguments, expected):
    result = run_local_cli(tmp_path, arguments)
    assert result.returncode == 0, result.stderr.decode()
    assert expected in result.stdout
    assert result.stderr == b""


@pytest.mark.parametrize("shell", ["bash", "zsh"])
@pytest.mark.parametrize("protocol", ["source", "complete"])
def test_completion_does_not_import_sdk_or_read_keys(tmp_path, shell, protocol):
    result = run_local_cli(
        tmp_path,
        [],
        _LMT_COMPLETE=f"{shell}_{protocol}",
        COMP_WORDS="lmt --template=trans",
        COMP_CWORD="1",
    )
    assert result.returncode == 0, result.stderr.decode()
    if protocol == "source":
        assert b"_lmt_completion" in result.stdout
    else:
        expected = b"plain,translate\n"
        if shell == "zsh":
            expected = b"plain\n--template=translate\n_\n"
        assert result.stdout == expected
    assert result.stderr == b""


def test_token_estimate_stays_key_free_without_sdk(tmp_path):
    result = run_local_cli(tmp_path, ["--tokens", "--model", "gpt-6-luna", "hello"])
    # This model has no upstream tokenizer mapping; no vocabulary download is needed.
    assert result.returncode == 1, result.stderr.decode()
    assert b"Request input tokens and cost: unavailable" in result.stdout
    assert result.stderr == b""
