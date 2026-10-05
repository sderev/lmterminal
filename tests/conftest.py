import hashlib
import os
import shutil
import socket
from pathlib import Path

import pytest
import tiktoken.load
import tiktoken.registry

from lmterminal import gpt_integration, lib


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests that hit the real API endpoint",
    )


@pytest.fixture
def no_provider_or_download(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Tokenizer tests must not access credentials, providers or downloads")

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "empty-cache"))
    monkeypatch.setattr(lib, "get_api_key", forbidden)
    monkeypatch.setattr(gpt_integration, "_get_client", forbidden)
    monkeypatch.setattr(gpt_integration.openai, "OpenAI", forbidden)
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
