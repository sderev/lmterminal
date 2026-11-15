import hashlib
import os
import shutil
import socket
import tempfile
from pathlib import Path

import pytest
import tiktoken.load
import tiktoken.registry
from click.testing import CliRunner

# Guard imports as well as ordinary tests, even when a live case is selected.
_home_variables = {name: os.environ.get(name) for name in ("HOME", "XDG_CONFIG_HOME")}
_private_environment = {
    name: value
    for name, value in os.environ.items()
    if any(part in name for part in ("API_KEY", "TOKEN", "OPENAI", "AZURE"))
    and name not in {"LMT_TEST_TIKTOKEN_CACHE", "TIKTOKEN_CACHE_DIR"}
}
_home = tempfile.TemporaryDirectory(prefix="lmt-tests-home-")
os.environ["HOME"] = _home.name
os.environ["XDG_CONFIG_HOME"] = _home.name + "/config"
os.environ["TIKTOKEN_CACHE_DIR"] = _home.name + "/tokenizer"
for _key in _private_environment:
    os.environ.pop(_key)


def _forbidden(*args, **kwargs):
    raise AssertionError("Offline tests must not construct provider clients or access networks")


_socket_connect = socket.socket.connect
_create_connection = socket.create_connection
socket.socket.connect = _forbidden
socket.create_connection = _forbidden
import openai

_provider_client = openai.OpenAI
openai.OpenAI = _forbidden
tiktoken.load.read_file = _forbidden

from lmterminal import cli, cli_output
from lmterminal.model_registry import resolve_model_name


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Opt in to one paid provider test; also requires --live-model MODEL",
    )
    parser.addoption(
        "--live-model",
        metavar="MODEL",
        help="Registered model or alias for the single paid live test; no default; requires --run-live",
    )


def pytest_configure(config):
    enabled = config.getoption("--run-live")
    selected = config.getoption("--live-model")
    if enabled != (selected is not None):
        raise pytest.UsageError("Paid live tests require both --run-live and --live-model MODEL")
    if selected is not None and resolve_model_name(selected) is None:
        raise pytest.UsageError(
            f"Unknown --live-model {selected!r}; choose a registered model or alias from `lmt models` "
            "(offline catalog; account access is not checked)"
        )


@pytest.fixture
def live_model(request, monkeypatch):
    selected = request.config.getoption("--live-model")
    if not request.config.getoption("--run-live") or selected is None:
        pytest.skip("Paid live test requires --run-live and --live-model MODEL")
    # A marker or a request for this fixture cannot grant another test live access.
    if (
        request.node.path != Path(__file__).with_name("live_test.py")
        or request.node.name != "test_live_call"
    ):
        raise pytest.UsageError("Only tests/live_test.py::test_live_call may use live access")
    with monkeypatch.context() as live:
        for name, value in (_home_variables | _private_environment).items():
            if value is None:
                live.delenv(name, raising=False)
            else:
                live.setenv(name, value)
        live.setattr(socket.socket, "connect", _socket_connect)
        live.setattr(socket, "create_connection", _create_connection)
        live.setattr(openai, "OpenAI", _provider_client)
        yield resolve_model_name(selected)


@pytest.fixture
def no_provider_or_download(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Tokenizer tests must not access credentials, providers or downloads")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(cli_output, "read_api_key", forbidden)
    monkeypatch.setattr("openai.OpenAI", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(tiktoken.load, "read_file", forbidden)
    monkeypatch.setattr(tiktoken.registry, "ENCODINGS", {})


@pytest.fixture
def tokenizer_cache(no_provider_or_download, monkeypatch, tmp_path):
    # Copy only explicitly provisioned public assets; never use an ambient cache.
    source = os.environ.get("LMT_TEST_TIKTOKEN_CACHE")
    if not source:
        message = "Set LMT_TEST_TIKTOKEN_CACHE to a provisioned cl100k_base/o200k_base cache"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    target = tmp_path / "oracle-cache"
    target.mkdir()
    for name in ("cl100k_base", "o200k_base"):
        url = f"https://openaipublic.blob.core.windows.net/encodings/{name}.tiktoken"
        filename = hashlib.sha1(url.encode()).hexdigest()
        asset = Path(source) / filename
        if not asset.is_file():
            pytest.fail(f"Provisioned tokenizer cache is missing {name}: {asset}")
        shutil.copyfile(asset, target / filename)
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(target))
    # Upstream loaders verify their expected SHA-256 before constructing encoders.
    return target


@pytest.fixture(params=["bash", "zsh"])
def completion_shell(request):
    return request.param


@pytest.fixture
def shell_completion(completion_shell, no_provider_or_download):
    """Read candidates through Click's shell protocol, with keys/network forbidden."""

    def complete(words, word_index):
        result = CliRunner().invoke(
            cli.lmt,
            [],
            prog_name="lmt",
            env={
                "_LMT_COMPLETE": f"{completion_shell}_complete",
                "COMP_WORDS": words,
                "COMP_CWORD": str(word_index),
            },
        )
        assert result.exit_code == 0, result.output
        lines = result.output.rstrip("\n").splitlines()
        if completion_shell == "bash":
            assert all(line.startswith("plain,") for line in lines), result.output
            return [line.partition(",")[2] for line in lines]
        assert len(lines) % 3 == 0, result.output
        assert all(kind == "plain" for kind in lines[::3]), result.output
        return lines[1::3]

    return complete
