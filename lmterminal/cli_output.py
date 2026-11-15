"""CLI presentation and execution policy; not a library generation API."""

import sys
from collections.abc import Mapping
from contextlib import nullcontext
from time import monotonic

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.theme import Theme

from . import gpt_integration
from .code_themes import resolve_code_theme
from .estimation import estimate_request
from .storage import StorageError, load_config, read_api_key

_MARKDOWN_REFRESH_INTERVAL = 1 / 20


class _CliError(click.ClickException):
    def show(self, file=None):
        click.echo(
            f"{click.style('Error:', fg='red')} {self.format_message()}",
            file=file,
            err=True,
        )


def _status_message(error):
    body = error.body
    message = body.get("message") if isinstance(body, Mapping) else None
    if isinstance(message, str) and message.strip():
        return message.strip()
    return f"OpenAI request failed (HTTP {error.status_code})."


def code_block_theme(config):
    value = config.get("code_block_theme")
    return value if isinstance(value, str) else "monokai"


def inline_code_theme(config):
    value = config.get("inline_code_theme")
    return value if isinstance(value, str) else "blue on black"


def execute_request(request, *, config_path, key_path, raw=False, stream=True, diagnostics=None):
    console = Console()
    use_live_markdown = (
        stream
        and not raw
        and getattr(sys.stdout, "isatty", lambda: False)()
        and console.is_terminal
        and console.is_interactive
        and not console.is_dumb_terminal
    )
    selected_theme = None
    if use_live_markdown:
        config = load_config(config_path)
        try:
            selected_theme = resolve_code_theme(code_block_theme(config))
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        console.push_theme(Theme({"markdown.code": inline_code_theme(config)}))

    try:
        api_key = read_api_key(key_path)
    except StorageError as error:
        raise click.ClickException(str(error)) from error
    if not api_key:
        raise click.ClickException("You need to set your OpenAI API key. Run `lmt key set`.")

    import openai

    markdown_stream = ""
    markdown_pending = False
    last_markdown_refresh = None
    live_context = (
        Live("", console=console, auto_refresh=False) if use_live_markdown else nullcontext()
    )
    if diagnostics:
        diagnostics.mark(
            "output ready",
            level=2,
            detail="mode=Markdown" if use_live_markdown else "mode=plain",
        )
    with live_context as live:

        def update_markdown_stream(chunk: str) -> None:
            nonlocal markdown_stream, markdown_pending, last_markdown_refresh
            if not chunk:
                return
            markdown_stream += chunk
            markdown_pending = True
            if (
                last_markdown_refresh is not None
                and monotonic() - last_markdown_refresh < _MARKDOWN_REFRESH_INTERVAL
            ):
                return
            if diagnostics:
                diagnostics.first("first text submitted to Markdown", level=2)
            live.update(
                Markdown(markdown_stream, code_theme=selected_theme),
                refresh=True,
            )
            if diagnostics:
                diagnostics.first("first Markdown refresh returned")
            last_markdown_refresh = monotonic()
            markdown_pending = False

        def emit_text(chunk):
            if use_live_markdown:
                update_markdown_stream(chunk)
            else:
                if diagnostics:
                    diagnostics.first("first text submitted to output", level=2)
                print(chunk, end="", flush=True)
                if diagnostics:
                    diagnostics.first("first text flushed")

        try:
            with openai.OpenAI(api_key=api_key) as client:
                if diagnostics:
                    diagnostics.mark("client ready", level=2)
                content, response_time, response = gpt_integration.send_prepared_request(
                    client=client,
                    request=request,
                    stream=stream,
                    on_text=emit_text,
                    diagnostics=diagnostics,
                )

            has_text = bool(content)
            if not content.endswith("\n"):
                content += "\n"
                if stream and not getattr(sys.stdout, "isatty", lambda: False)():
                    print("\n", end="", flush=True)

            if not stream:
                if diagnostics and has_text:
                    diagnostics.first("first text submitted to output", level=2)
                print(content, end="", flush=True)
                if diagnostics and has_text:
                    diagnostics.first("first text flushed")

        except openai.RateLimitError as error:
            raise _CliError(
                f"{_status_message(error)}\nWait before retrying, or reduce your request rate."
            ) from error

        except openai.AuthenticationError:
            raise _CliError(
                "Your OpenAI API key is invalid, expired, or revoked. "
                f"Check your key and run `{click.style('lmt key edit', fg='blue')}` to replace it."
            ) from None

        except openai.APIConnectionError as error:
            raise _CliError(
                "Could not connect to OpenAI. Check your connection and retry."
            ) from error

        except openai.APIStatusError as error:
            raise _CliError(_status_message(error)) from error

        except BrokenPipeError:
            # Click owns quiet EPIPE handling, including interpreter-shutdown flushing.
            raise

        # Preserve the CLI's error-to-exit contract for unexpected failures.
        except Exception as error:
            raise _CliError(str(error)) from error

        finally:
            # Live refreshes on exit, including errors and Ctrl+C. Give it any
            # text coalesced since the last refresh before that final render.
            if use_live_markdown and markdown_pending:
                live.update(Markdown(markdown_stream, code_theme=selected_theme))

    if diagnostics:
        diagnostics.output_complete()
    return content, response_time, response


def display_debug_information(prompt, model, temperature):
    """
    Displays debug information.
    """
    click.echo("---", err=True)
    click.secho("Debug information:", fg="yellow", err=True)
    click.echo(err=True)

    click.secho("Prompt:", fg="red", nl=False, err=True)
    for role in prompt:
        click.echo(err=True)
        click.secho(f"{role['role']}:", fg="blue", err=True)
        click.echo(f"{role}", err=True)
    click.echo(err=True)

    click.secho("Model:", fg="red", err=True)
    click.echo(f"{model=}", err=True)
    click.echo(err=True)

    click.secho("Temperature:", fg="red", err=True)
    click.echo(f"{temperature=}", err=True)
    click.echo(err=True)

    click.secho("End of debug information.", fg="yellow", err=True)
    click.echo("---\n", err=True)


def display_tokens_count_and_cost(request):
    """Render the local estimate without implying provider usage or a total bill."""
    estimate = estimate_request(request)
    click.echo(f"Model: {click.style(estimate.model, fg='blue')}")
    if estimate.input_tokens is not None:
        click.echo(
            f"Estimated input tokens: {click.style(f'~{estimate.input_tokens}', fg='yellow')}"
        )
    elif estimate.message_tokens is not None:
        click.echo(
            f"Message-only token estimate: {click.style(f'~{estimate.message_tokens}', fg='yellow')}"
        )
        click.echo("Request input tokens and cost: unavailable")
    else:
        click.echo("Request input tokens and cost: unavailable")
    if estimate.input_cost_usd is not None:
        click.echo(
            "Standard uncached input cost estimate:"
            f" {click.style(f'USD {estimate.input_cost_usd:f}', fg='yellow')}"
        )
        click.echo(
            f"Input rate: {click.style(f'USD {estimate.input_rate_usd_per_million:f}', fg='yellow')}"
            " / 1M tokens"
        )
    elif estimate.input_tokens is not None:
        click.echo("Input cost: unavailable")
    if estimate.pricing_context:
        click.echo(
            f"Pricing tier: {click.style(estimate.pricing_context, fg='yellow')} context,"
            " based on estimated tokens."
        )
    for reason in estimate.warnings:
        click.echo(f"Note: {reason}")
    click.echo("Local message framing is heuristic; provider usage may differ.")
    click.echo("Cache reads and writes are excluded; actual input charges may be lower or higher.")
    click.echo("Excludes output/reasoning, tool fees and service-tier adjustments; not a bill.")
    return 1 if estimate.message_tokens is None else 0
