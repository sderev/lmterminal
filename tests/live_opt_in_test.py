"""Exercise pytest's live boundary in subprocesses with synthetic state and clients."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = """
import json
import os
import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import openai
import pytest

def forbidden(*args, **kwargs):
    raise RuntimeError("Subprocess sentinel forbids network access")

socket.socket.connect = forbidden
socket.create_connection = forbidden

def create(**kwargs):
    with open(os.environ['LMT_CALL_LOG'], 'a') as log:
        log.write(json.dumps(kwargs['model']) + '\\n')
    return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='pong'))])]

class FakeClient:
    chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    def __init__(self, *, api_key):
        if (
            api_key != 'synthetic-key'
            or str(Path.home()) != os.environ['LMT_SENTINEL_HOME']
            or os.environ.get('OPENAI_API_KEY') != 'synthetic-env-key'
            or os.environ['XDG_CONFIG_HOME'] != str(Path.home() / 'xdg-config')
        ):
            raise RuntimeError('Fake client used outside the selected live context')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

openai.OpenAI = FakeClient

def audit(event, args):
    key_path = str(Path(os.environ['LMT_SENTINEL_HOME']) / '.config/lmt/keys.json')
    if event == 'open' and args[0] == key_path:
        with open(Path(os.environ['LMT_CALL_LOG']).with_suffix('.credentials'), 'a') as log:
            log.write('synthetic key read\\n')

sys.addaudithook(audit)
sys.exit(pytest.main(sys.argv[1:]))
"""
GUARDS = """
import os
import socket
from pathlib import Path

import openai
import pytest
import tiktoken.load

def assert_guarded():
    assert str(Path.home()) != os.environ['LMT_SENTINEL_HOME']
    assert os.environ['XDG_CONFIG_HOME'].startswith(str(Path.home()))
    for name in ('OPENAI_API_KEY', 'AZURE_API_KEY', 'ACCESS_TOKEN'):
        assert name not in os.environ
    with pytest.raises(AssertionError):
        openai.OpenAI(api_key='synthetic-key')
    with pytest.raises(AssertionError):
        socket.create_connection(('example.invalid', 443))
    with socket.socket() as connection, pytest.raises(AssertionError):
        connection.connect(('127.0.0.1', 443))
    with pytest.raises(AssertionError):
        tiktoken.load.read_file('https://example.invalid/tokenizer')

assert_guarded()  # Collection/imports must also run inside the offline boundary.

@pytest.mark.integration
def test_marker_stays_guarded():
    assert_guarded()

def test_after_live_stays_guarded():
    assert_guarded()
"""


@pytest.fixture
def isolated_pytest(tmp_path):
    suite = tmp_path / "suite"
    tests = suite / "tests"
    tests.mkdir(parents=True)
    for name in ("conftest.py", "live_test.py"):
        shutil.copyfile(ROOT / "tests" / name, tests / name)
    (tests / "guard_test.py").write_text(GUARDS)
    home = tmp_path / "ambient-home"
    configuration = home / ".config/lmt"
    configuration.mkdir(parents=True)
    (configuration / "keys.json").write_text('{"openai": "synthetic-key"}')
    calls = tmp_path / "calls.jsonl"
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / "xdg-config"),
        "OPENAI_API_KEY": "synthetic-env-key",
        "AZURE_API_KEY": "synthetic-env-key",
        "ACCESS_TOKEN": "synthetic-token",
        "LMT_SENTINEL_HOME": str(home),
        "LMT_CALL_LOG": str(calls),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }

    def run(*arguments):
        result = subprocess.run(
            [sys.executable, "-c", RUNNER, "-q", *arguments],
            cwd=suite,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        return result, calls

    return run


def test_live_selection_preserves_guards_for_other_tests(isolated_pytest):
    result, calls = isolated_pytest(
        "tests/guard_test.py::test_marker_stays_guarded",
        "tests/live_test.py",
        "tests/guard_test.py::test_after_live_stays_guarded",
        "--run-live",
        "--live-model",
        "4o-mini",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout
    assert calls.read_text().splitlines() == [json.dumps("gpt-4o-mini")]
    assert calls.with_suffix(".credentials").exists()


def test_live_collection_is_one_case_and_default_run_skips_it(isolated_pytest):
    result, calls = isolated_pytest(
        "--collect-only", "tests/live_test.py", "--run-live", "--live-model", "4o-mini"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout
    assert not calls.exists()
    result, calls = isolated_pytest("tests/live_test.py")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout
    assert not calls.exists()


@pytest.mark.parametrize(
    "arguments, message",
    [
        (("--run-live",), "require both --run-live and --live-model MODEL"),
        (("--live-model", "4o-mini"), "require both --run-live and --live-model MODEL"),
        (("--run-live", "--live-model", "no-such-model"), "Unknown --live-model"),
    ],
)
def test_live_selector_errors_are_local(isolated_pytest, arguments, message):
    result, calls = isolated_pytest("tests/live_test.py", *arguments)
    assert result.returncode == pytest.ExitCode.USAGE_ERROR, result.stdout + result.stderr
    assert message in result.stderr
    assert not calls.exists()
    assert not calls.with_suffix(".credentials").exists()


def test_gate_refuses_live_and_enforces_offline_selection(tmp_path):
    executable = tmp_path / "bin"
    executable.mkdir()
    log = tmp_path / "commands.jsonl"
    for name in ("uv", "uvx"):
        stub = executable / name
        stub.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "with open(os.environ['LMT_GATE_LOG'], 'a') as log:\n"
            "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        )
        stub.chmod(0o700)
    environment = {
        "PATH": str(executable) + os.pathsep + os.environ["PATH"],
        "LMT_GATE_LOG": str(log),
    }
    for option_source in ("GATE_PYTEST_ARGS", "PYTEST_ADDOPTS"):
        result = subprocess.run(
            ["bash", ".ci/gate"],
            cwd=ROOT,
            env=environment | {option_source: "--run-live --live-model 4o-mini"},
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 2, result.stdout + result.stderr
        assert "gate is offline" in result.stderr
        assert "directly with pytest" in result.stderr
        assert not log.exists()
    result = subprocess.run(
        ["bash", ".ci/gate"],
        cwd=ROOT,
        env=environment | {"GATE_PYTEST_ARGS": "-m integration", "GATE_REAL": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    test_runs = [arguments for arguments in commands if "pytest" in arguments]
    assert [arguments[2] for arguments in test_runs] == ["3.10", "3.11", "3.12", "3.13", "3.14"]
    assert all(arguments[-2:] == ["-m", "not integration"] for arguments in test_runs)
