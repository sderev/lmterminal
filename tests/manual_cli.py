"""Offline CLI walkthrough: uv run python tests/manual_cli.py [--fixture MODE] [lmt args]."""

import argparse
import json
import os
import socket
import tempfile
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture", choices=("echo", "markdown", "partial-failure", "missing-key"), default="echo"
    )
    options, arguments = parser.parse_known_args()

    def forbidden(*args, **kwargs):
        raise AssertionError("Offline walkthrough forbids real clients and network access")

    socket.socket.connect = forbidden
    socket.create_connection = forbidden
    with tempfile.TemporaryDirectory(prefix="lmt-walkthrough-") as home:
        os.environ["HOME"] = home
        os.environ["XDG_CONFIG_HOME"] = home + "/config"
        os.environ["TIKTOKEN_CACHE_DIR"] = home + "/empty-tokenizers"
        for name in list(os.environ):
            if any(part in name for part in ("API_KEY", "TOKEN", "OPENAI", "AZURE")):
                os.environ.pop(name)
        os.environ["TIKTOKEN_CACHE_DIR"] = home + "/empty-tokenizers"

        import openai

        openai.OpenAI = forbidden
        from lmterminal.cli import lmt

        directory = Path(home) / ".config/lmt"
        templates = directory / "templates"
        templates.mkdir(parents=True)
        (templates / "translate.yaml").write_text("prompt: Translate into English.\nmodel: 4o\n")
        (templates / "sparse.yaml").write_text("prompt: null\nunknown: false\n")
        (directory / "config.json").write_text(
            json.dumps({"code_block_theme": "alabaster", "inline_code_theme": "#325cc0 on #f0f0f0"})
        )
        if options.fixture != "missing-key":
            (directory / "keys.json").write_text('{"openai": "synthetic-key"}\n')
            (directory / "keys.json").chmod(0o600)

        def create(**kwargs):
            text = kwargs["messages"][-1]["content"]
            if options.fixture == "markdown":
                text = 'Inline `fib`.\n\n```python\ndef fib():\n    return "hello"\n```\n'

            def events():
                if options.fixture == "partial-failure":
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))]
                    )
                    raise RuntimeError("Synthetic stream interruption")
                for part in (text[:3], text[3:]):
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
                    )

            if kwargs["stream"]:
                return events()
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])

        class FixtureClient:
            chat = SimpleNamespace(completions=SimpleNamespace(create=create))

            def __init__(self, *, api_key):
                assert api_key == "synthetic-key"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        openai.OpenAI = FixtureClient
        try:
            lmt.main(args=arguments, prog_name="lmt")
        finally:
            if options.fixture == "missing-key":
                assert not (directory / "keys.json").exists()


if __name__ == "__main__":
    main()
