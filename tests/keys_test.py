import json
import os
import socket
import stat

import pytest
from click.testing import CliRunner

from lmterminal import cli, cli_output, storage
from lmterminal.storage import StorageError


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    def forbidden(*args, **kwargs):
        pytest.fail("Key tests must not contact a provider")

    monkeypatch.setattr(cli_output.openai, "OpenAI", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture
def key_path(tmp_path):
    path = tmp_path / ".config/lmt/keys.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="UTF-8")
    return path


def test_new_key_file_is_private_under_permissive_umask(tmp_path, key_path):
    previous_umask = os.umask(0)
    try:
        path = key_path
        assert path == tmp_path / ".config" / "lmt" / "keys.json"
        path.unlink()
        assert storage.read_api_key(key_path) == ""
        assert not path.exists()

        key = '  synthetic-"key"-\\-é\n'
        storage.write_key(key_path, key)
        assert json.loads(path.read_text(encoding="UTF-8")) == {"openai": key}
        assert storage.read_api_key(key_path) == key.strip()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        os.umask(previous_umask)


def test_overwrite_restricts_permissions_before_replacing_key(monkeypatch, key_path):
    path = key_path
    original_keys = {"openai": "synthetic-original-key", "future": "synthetic-other-key"}
    path.write_text(json.dumps(original_keys), encoding="UTF-8")
    original_bytes = path.read_bytes()
    path.chmod(0o666)
    assert storage.read_api_key(key_path) == "synthetic-original-key"
    assert stat.S_IMODE(path.stat().st_mode) == 0o666
    real_fchmod = os.fchmod
    calls = []

    def restrict(descriptor, mode):
        assert path.read_bytes() == original_bytes
        real_fchmod(descriptor, mode)
        assert stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600
        calls.append(mode)

    monkeypatch.setattr(storage.os, "fchmod", restrict)
    storage.write_key(key_path, "dummy-new")

    assert calls == [0o600]
    assert json.loads(path.read_text(encoding="UTF-8")) == {
        "openai": "dummy-new",
        "future": "synthetic-other-key",
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_permission_failure_leaves_stored_key_intact(monkeypatch, key_path):
    path = key_path
    path.write_text('{"openai": "synthetic-original-key"}', encoding="UTF-8")
    original_bytes = path.read_bytes()
    path.chmod(0o666)

    def fail(*args):
        raise PermissionError("synthetic chmod failure")

    monkeypatch.setattr(storage.os, "fchmod", fail)
    with pytest.raises(PermissionError, match="synthetic chmod failure"):
        storage.write_key(key_path, "dummy-new")
    assert path.read_bytes() == original_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o666


@pytest.mark.parametrize(
    "contents",
    [
        b'{"openai": "synthetic-secret-sentinel",',
        b'{"openai": "synthetic-secret-sentinel\xff"}',
        b'["synthetic-secret-sentinel"]',
        b'"synthetic-secret-sentinel"',
        b"null",
        b'{"openai": null}',
        b'{"openai": false}',
        b'{"openai": {"nested": "synthetic-secret-sentinel"}}',
        b'{"openai": "dummy", "future": 42}',
    ],
)
def test_invalid_storage_fails_without_exposing_or_replacing_contents(contents, key_path):
    path = key_path
    path.write_bytes(contents)
    path.chmod(0o666)

    for operation in (
        lambda: storage.read_api_key(key_path),
        lambda: storage.write_key(key_path, "dummy-new"),
    ):
        with pytest.raises(StorageError, match="keys.json must contain") as error:
            operation()
        assert "synthetic-secret-sentinel" not in str(error.value)
        assert error.value.__cause__ is None
        assert path.read_bytes() == contents
        assert stat.S_IMODE(path.stat().st_mode) == 0o666


def test_write_rejects_non_string_key_without_replacing_storage(key_path):
    path = key_path
    original_bytes = path.read_bytes()
    with pytest.raises(StorageError, match="API key must be a string"):
        storage.write_key(key_path, None)
    assert path.read_bytes() == original_bytes


@pytest.mark.parametrize("command", [["key", "set"], ["key", "edit"], ["--raw", "dummy"]])
@pytest.mark.parametrize(
    "contents, message",
    [
        ('{"openai": "synthetic-secret-sentinel",', "valid UTF-8 JSON"),
        ('{"openai": ["synthetic-secret-sentinel"]}', "object with string key values"),
    ],
)
def test_cli_reports_invalid_storage_without_secret_contents(command, contents, message, key_path):
    path = key_path
    path.write_text(contents, encoding="UTF-8")
    result = CliRunner().invoke(cli.lmt, command)
    assert result.exit_code == 1, result.output
    assert message in result.output
    assert "synthetic-secret-sentinel" not in result.output
    assert "Traceback" not in result.output
    assert path.read_text(encoding="UTF-8") == contents


@pytest.mark.parametrize("command", ["set", "edit"])
@pytest.mark.parametrize("openai_entry", [{}, {"openai": ""}])
def test_key_commands_add_missing_openai_key_and_preserve_other_provider(
    command, openai_entry, key_path
):
    path = key_path
    path.write_text(json.dumps({"future": "dummy-other", **openai_entry}), encoding="UTF-8")
    result = CliRunner().invoke(cli.lmt, ["key", command], input="dummy-first\n")
    assert result.exit_code == 0, result.output
    assert "API key added." in result.output
    assert "dummy-first" not in result.output
    assert json.loads(path.read_text(encoding="UTF-8")) == {
        "future": "dummy-other",
        "openai": "dummy-first",
    }


def test_key_commands_keep_hidden_input_and_unchanged_key_behavior(key_path):
    runner = CliRunner()
    result = runner.invoke(cli.lmt, ["key", "set"], input="dummy-first\n")
    assert result.exit_code == 0, result.output
    assert "API key added." in result.output
    assert "dummy-first" not in result.output
    path = key_path
    path.write_text('{"openai": "dummy-first", "future": "dummy-other"}', encoding="UTF-8")
    original_bytes = path.read_bytes()
    path.chmod(0o666)

    result = runner.invoke(cli.lmt, ["key", "set"])
    assert result.exit_code == 0, result.output
    assert "API key already exists." in result.output
    result = runner.invoke(cli.lmt, ["key", "edit"], input="dummy-first\n")
    assert result.exit_code == 0, result.output
    assert "No changes were made." in result.output
    assert "dummy-first" not in result.output
    assert path.read_bytes() == original_bytes
    assert stat.S_IMODE(path.stat().st_mode) == 0o666

    result = runner.invoke(cli.lmt, ["key", "edit"], input="dummy-second\n")
    assert result.exit_code == 0, result.output
    assert "API key was updated." in result.output
    assert "dummy-second" not in result.output
    assert storage.read_api_key(key_path) == "dummy-second"
    assert json.loads(path.read_text(encoding="UTF-8")) == {
        "openai": "dummy-second",
        "future": "dummy-other",
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_missing_key_cli_is_stderr_only_and_creates_nothing(tmp_path):
    result = CliRunner().invoke(cli.lmt, ["--raw", "fixture"], input="")
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "lmt key set" in result.stderr
    assert not (tmp_path / ".config/lmt").exists()
