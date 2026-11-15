"""Offline CLI walkthrough with a fake provider and temporary configuration.

uv run python tests/manual_cli.py [--saved-model NAME] [--fixture MODE] [lmt args]
"""

import argparse
import json
import os
import socket
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace


def response_parts(fixture, text):
    if fixture in {"long-code", "scrollback-failure", "scrollback-interrupt"}:
        return (
            ["Intro `fib`.\n\n```python\n"]
            + [f'def item_{n}(): return "TRACE-{n:03d}"  # café 漢字\n' for n in range(60)]
            + ["```\n\nDONE-CODE\n"]
        )
    if fixture == "long-paragraph":
        return [f"TRACE-{n:03d} **bold** `inline` café 漢字 " for n in range(100)] + ["END\n"]
    if fixture == "list-table":
        return (
            [f"{n + 1}. TRACE-{n:03d} List **bold** `inline`\n" for n in range(30)]
            + ["\n| Item | Value |\n| --- | --- |\n"]
            + [f"| TRACE-{n:03d} | **row** {n} `inline` |\n" for n in range(30, 60)]
            + ["| A later wide column | a later wide value |\n\nDONE-TABLE\n"]
        )
    if fixture == "markdown":
        return ['Inline `fib`.\n\n```python\ndef fib():\n    return "hello"\n```\n']
    return [text[:3], text[3:]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        choices=(
            "echo",
            "markdown",
            "partial-failure",
            "missing-key",
            "auth",
            "pipe",
            "long-code",
            "long-paragraph",
            "list-table",
            "scrollback-failure",
            "scrollback-interrupt",
        ),
        default="echo",
    )
    parser.add_argument(
        "--delay", type=float, default=0.08, help="Seconds between long-fixture chunks."
    )
    parser.add_argument(
        "--hold", type=float, default=3.0, help="Seconds to hold a long fixture open."
    )
    parser.add_argument(
        "--code-theme", default="alabaster", help="Fixture code_block_theme preference."
    )
    parser.add_argument("--saved-model", help="Save a model preference in the temporary config.")
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
        config = {
            "code_block_theme": options.code_theme,
            "inline_code_theme": "#325cc0 on #f0f0f0",
        }
        if options.saved_model is not None:
            config["default_model"] = options.saved_model
        (directory / "config.json").write_text(json.dumps(config))
        if options.fixture != "missing-key":
            (directory / "keys.json").write_text('{"openai": "synthetic-key"}\n')
            (directory / "keys.json").chmod(0o600)

        def create(**kwargs):
            if options.fixture == "auth":
                import httpx

                raise openai.AuthenticationError(
                    "PRIVATE",
                    response=httpx.Response(
                        401, request=httpx.Request("POST", "https://example.invalid")
                    ),
                    body={"message": "Invalid PRIVATE key"},
                )
            text = kwargs["messages"][-1]["content"]
            parts = response_parts(options.fixture, text)
            paced = options.fixture in {
                "long-code",
                "long-paragraph",
                "list-table",
                "scrollback-failure",
                "scrollback-interrupt",
            }

            def events():
                if options.fixture == "pipe":
                    for _ in range(100000):
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="x" * 16))]
                        )
                    return
                if options.fixture == "partial-failure":
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="partial"))]
                    )
                    raise RuntimeError("Synthetic stream interruption")
                for part in parts:
                    if paced:
                        time.sleep(options.delay)
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content=part))]
                    )
                if options.fixture in {"scrollback-failure", "scrollback-interrupt"}:
                    # Deliver one last chunk without a pause, under the refresh cadence.
                    yield SimpleNamespace(
                        choices=[SimpleNamespace(delta=SimpleNamespace(content="COALESCED-TAIL"))]
                    )
                    if options.fixture == "scrollback-failure":
                        raise RuntimeError("Synthetic stream interruption")
                if paced:
                    time.sleep(options.hold)

            if kwargs["stream"]:
                return events()
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="".join(parts)))]
            )

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
