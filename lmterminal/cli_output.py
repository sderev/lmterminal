"""CLI presentation and execution policy; not a library generation API."""

import sys
from contextlib import nullcontext

import click
import openai
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.theme import Theme

from . import gpt_integration
from .code_themes import resolve_code_theme
from .estimation import estimate_request
from .storage import StorageError, load_config, read_api_key

BLUE = "\x1b[34m"
RED = "\x1b[91m"
RESET = "\x1b[0m"


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

    markdown_stream = ""
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
            nonlocal markdown_stream
            if not chunk:
                return
            markdown_stream += chunk
            if diagnostics:
                diagnostics.first("first text submitted to Markdown", level=2)
            live.update(
                Markdown(markdown_stream, code_theme=selected_theme),
                refresh=True,
            )
            if diagnostics:
                diagnostics.first("first Markdown refresh returned")

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
            click.echo(f"{RED}Error:{RESET} {error}", err=True)
            handle_rate_limit_error()
            sys.exit(1)

        except openai.AuthenticationError:
            handle_authentication_error()
            sys.stderr.write("\nYou can set your API key by running: ")
            sys.stderr.write(f"{BLUE}lmt key set{RESET}\n")
            sys.exit(1)

        except openai.APIConnectionError as error:
            click.echo(f"{RED}Error:{RESET} {error}", err=True)
            sys.exit(1)

        # Preserve the CLI's error-to-exit contract for unexpected failures.
        except Exception as error:  # noqa: BLE001
            click.echo(f"{RED}Error:{RESET} {error}", err=True)
            sys.exit(1)

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


def handle_rate_limit_error():
    """
    Provides guidance on how to handle a rate limit error.
    """
    sys.stderr.write("\n")
    sys.stderr.write(
        BLUE
        + "You might not have set a usage rate limit in your OpenAI account settings. "
        + RESET
        + "\n"
    )
    sys.stderr.write(
        "If that's the case, you can set it"
        " here:\nhttps://platform.openai.com/account/billing/limits" + "\n"
    )

    sys.stderr.write("\n")
    sys.stderr.write(
        BLUE + "If you have set a usage rate limit, please try the following steps:" + RESET + "\n"
    )
    sys.stderr.write("- Wait a few seconds before trying again.\n")
    sys.stderr.write("\n")
    sys.stderr.write(
        "- Reduce your request rate or batch tokens. You can read the"
        " OpenAI rate limits"
        " here:\nhttps://platform.openai.com/account/rate-limits" + "\n"
    )
    sys.stderr.write("\n")
    sys.stderr.write(
        "- If you are using the free plan, you can upgrade to the paid"
        " plan"
        " here:\nhttps://platform.openai.com/account/billing/overview" + "\n"
    )
    sys.stderr.write("\n")
    sys.stderr.write(
        "- If you are using the paid plan, you can increase your usage"
        " rate limit"
        " here:\nhttps://platform.openai.com/account/billing/limits" + "\n"
    )


def handle_authentication_error():
    """
    Provides guidance on how to handle an authentication error.
    """
    sys.stderr.write(
        f"{RED}Error:{RESET} Your API key or token is invalid, expired, or"
        " revoked. Check your API key or token and make sure it is correct"
        " and active.\n"
    )
    sys.stderr.write(
        "\nYou may need to generate a new API key from your account"
        " dashboard: https://platform.openai.com/account/api-keys\n"
    )
