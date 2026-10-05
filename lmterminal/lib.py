import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import click
import openai
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.theme import Theme

from . import gpt_integration as openai_utils
from .code_themes import resolve_code_theme
from .estimation import estimate_request
from .model_registry import resolve_model_name
from .request_options import DEFAULT_MODEL as DEFAULT_MODEL  # noqa: PLC0414 -- public constant
from .request_options import UNSET, PreparedRequest, prepare_request, validate_request_options
from .templates import Template, load_template

BLUE = "\x1b[34m"
RED = "\x1b[91m"
RESET = "\x1b[0m"

DEFAULT_CODE_BLOCK_THEME = "monokai"
DEFAULT_INLINE_CODE_THEME = "blue on black"


class RequestResolutionError(ValueError):
    """Invocation inputs or effective request settings cannot be resolved."""


class SystemTemplateConflict(RequestResolutionError):
    """A supplied system argument cannot accompany a template."""


def _join_inputs(stored, supplied):
    return stored + "\n\n" + supplied if stored and supplied else stored or supplied


def resolve_request(
    *,
    template: Template | str | None = None,
    system=UNSET,
    prompt: str = "",
    text: str = "",
    model=UNSET,
    temperature=UNSET,
    reasoning_effort=UNSET,
    request_options=None,
) -> PreparedRequest:
    """Resolve inputs offline into a PreparedRequest shared by CLI and library.

    Explicit scalars override template settings, then package defaults. Explicit
    None omits a control. Options merge by top-level key. System argument presence
    conflicts with any template, even when the argument is empty or None.
    """
    if template is not None and system is not UNSET:
        raise SystemTemplateConflict("You cannot use both `--template` and `--system`.")
    if isinstance(template, str):
        template = load_template(template)
    if template is not None and not isinstance(template, Template):
        raise RequestResolutionError("Template must be a Template, a name or None.")
    for name, value in (("prompt", prompt), ("text", text)):
        if not isinstance(value, str):
            raise RequestResolutionError(f"Argument `{name}` must be text.")
    if system is not UNSET and system is not None and not isinstance(system, str):
        raise RequestResolutionError("Argument `system` must be text or None.")
    stored = template if template is not None else Template()
    system = stored.system if system is UNSET else system or ""
    task = _join_inputs(stored.prompt, prompt)
    content = _join_inputs(stored.text, text)
    user = content + "\n___\n" + task if content and task else content or task
    if model is UNSET and stored.model is not None:
        model = stored.model
    if model is not UNSET:
        canonical = resolve_model_name(model) if isinstance(model, str) else None
        if canonical is None:
            raise RequestResolutionError("Argument/field `model` must be a registered model name.")
        model = canonical
    if temperature is UNSET and stored.temperature is not None:
        temperature = stored.temperature
    if (
        temperature is not UNSET
        and temperature is not None
        and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        )
    ):
        raise RequestResolutionError("Argument `temperature` must be between 0 and 2 or None.")
    if reasoning_effort is UNSET and stored.reasoning_effort is not None:
        reasoning_effort = stored.reasoning_effort
    try:
        validate_request_options(request_options)
        options = {**stored.request_options, **(request_options or {})}
        messages = openai_utils.format_prompt(system, user)
        # Retain the existing o1 message policy in both entry points.
        if isinstance(model, str) and "o1" in model:
            messages = [{"role": "user", "content": user}]
        return prepare_request(model, messages, temperature, reasoning_effort, options)
    except (TypeError, ValueError) as error:
        raise RequestResolutionError(str(error)) from error


def prepare_and_generate_response(
    request,
    *,
    emoji=False,
    tokens=False,
    no_stream=False,
    raw=False,
    debug=False,
    diagnostics=None,
):
    """Render, estimate or send an already resolved request."""
    if emoji:
        messages = [dict(message) for message in request.messages]
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = add_emoji(messages[0]["content"])
        request = PreparedRequest(request.model, messages, request.controls)
    if diagnostics:
        diagnostics.request_prepared(request, not no_stream)
    if debug:
        display_debug_information(
            request.messages, request.model, request.controls.get("temperature")
        )
    if tokens:
        display_tokens_count_and_cost(request)
    return _generate_prepared_response(request, raw, not no_stream, diagnostics=diagnostics)


def _prepare_request(model, prompt, temperature, reasoning_effort, request_options):
    try:
        return prepare_request(model, prompt, temperature, reasoning_effort, request_options)
    except (TypeError, ValueError) as error:
        raise click.BadParameter(str(error)) from error


def add_emoji(system: str) -> str:
    """
    Adds an emoji to the system message.
    """
    emoji_message = (
        "Add plenty of emojis as a colorful way to convey emotions. However, don't mention it."
    )
    system = system.rstrip()

    if system == "":
        return emoji_message

    if not system.endswith("."):
        system += "."
    return system + " " + emoji_message


def get_config_path() -> Path:
    """
    Gets the path to the config file.
    """
    config_path = Path.home() / ".config/lmt/config.json"
    return config_path


def load_config() -> dict:
    """
    Reads the config file without creating or changing it.
    """
    config_path = get_config_path()

    try:
        with open(config_path, "r", encoding="UTF-8") as file:
            config = json.load(file)
    except (json.decoder.JSONDecodeError, OSError):
        return {}

    if not isinstance(config, dict):
        return {}

    return config


def save_config(config: dict) -> None:
    """
    Saves the config file.
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="UTF-8") as config_file:
        json.dump(config, config_file, indent=4)


def get_markdown_code_block_theme() -> str:
    """
    Gets the markdown code block theme from the config file.
    """
    config = load_config()
    code_block_theme = config.get("code_block_theme")
    if isinstance(code_block_theme, str):
        return code_block_theme
    return DEFAULT_CODE_BLOCK_THEME


def get_markdown_inline_code_theme() -> str:
    """
    Gets the markdown inline code theme from the config file.
    """
    config = load_config()
    inline_code_theme = config.get("inline_code_theme")
    if isinstance(inline_code_theme, str):
        return inline_code_theme
    return DEFAULT_INLINE_CODE_THEME


def generate_response(
    model=UNSET,
    prompt: str | None = None,
    raw: bool = False,
    stream: bool = True,
    temperature=UNSET,
    *,
    reasoning_effort=UNSET,
    request_options: dict | None = None,
    diagnostics=None,
):
    """
    Generate a response; omitted model/effort use Luna/none.

    Explicit models retain their provider effort default. Explicit None omits the control.
    """
    if diagnostics:
        diagnostics.mark("request preparation started", level=2)
    request = _prepare_request(model, prompt, temperature, reasoning_effort, request_options)
    if diagnostics:
        diagnostics.request_prepared(request, stream)
    return _generate_prepared_response(request, raw, stream, diagnostics=diagnostics)


def _generate_prepared_response(request, raw, stream, *, diagnostics=None):
    console = Console()
    use_live_markdown = (
        stream
        and not raw
        and getattr(sys.stdout, "isatty", lambda: False)()
        and console.is_terminal
        and console.is_interactive
        and not console.is_dumb_terminal
    )
    code_block_theme = None
    if use_live_markdown:
        try:
            code_block_theme = resolve_code_theme(get_markdown_code_block_theme())
        except ValueError as error:
            raise click.ClickException(str(error)) from error
        console.push_theme(Theme({"markdown.code": get_markdown_inline_code_theme()}))

    api_key = get_api_key()

    if not api_key:
        click.secho("Error:", fg="red", nl=False)
        click.echo(" You need to set your OpenAI API key.")
        click.echo("You can do so by running:", nl=False)
        click.echo(f"  {click.style('lmt key set', fg='blue')}\n")
        sys.exit(1)

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
                Markdown(markdown_stream, code_theme=code_block_theme),
                refresh=True,
            )
            if diagnostics:
                diagnostics.first("first Markdown refresh returned")

        try:
            content, response_time, response = openai_utils.send_prepared_request(
                api_key=api_key,
                request=request,
                stream=stream,
                update_markdown_stream=update_markdown_stream if use_live_markdown else None,
                diagnostics=diagnostics,
            )

            has_text = bool(content)
            # This is temporary to ensure that the last line always ends with a newline
            # This will be removed when refactored
            if not content.endswith("\n"):
                content += "\n"
            #############################

            if not stream:
                if diagnostics and has_text:
                    diagnostics.first("first text submitted to output", level=2)
                print(content, end="", flush=True)
                if diagnostics and has_text:
                    diagnostics.first("first text flushed")

        except openai.RateLimitError as error:
            click.echo(f"{RED}Error:{RESET} {error}", err=True)
            openai_utils.handle_rate_limit_error()
            sys.exit(1)

        except openai.AuthenticationError:
            openai_utils.handle_authentication_error()
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
    sys.exit(1 if estimate.message_tokens is None else 0)


def get_api_key() -> str:
    """
    Return the OpenAI API key.
    """
    key_file_path = get_api_key_path()
    return _read_keys(key_file_path).get("openai", "").strip()


def _read_keys(key_file_path: Path) -> dict[str, str]:
    """Read the provider-key mapping without exposing invalid contents in errors."""
    with open(key_file_path, "r", encoding="UTF-8") as key_file:
        try:
            keys = json.load(key_file)
        except (json.JSONDecodeError, UnicodeError):
            raise click.ClickException("keys.json must contain valid UTF-8 JSON.") from None
    if not isinstance(keys, dict) or any(not isinstance(value, str) for value in keys.values()):
        raise click.ClickException("keys.json must contain an object with string key values.")
    return keys


def get_api_key_path() -> Path:
    """
    Return the path to the keys file.
    """
    key_file_path = Path.home() / ".config" / "lmt" / "keys.json"
    if not key_file_path.exists():
        key_file_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(key_file_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="UTF-8") as key_file:
            key_file.write("{}\n")
    return key_file_path


def write_key(key: str) -> None:
    """
    Write the OpenAI API key, preserving other provider entries.
    """
    if not isinstance(key, str):
        raise click.ClickException("API key must be a string.")
    key_file_path = get_api_key_path()
    keys = _read_keys(key_file_path)
    keys["openai"] = key
    descriptor = os.open(key_file_path, os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "w", encoding="UTF-8") as key_file:
        # Restrict the opened file before replacing any stored key.
        os.fchmod(key_file.fileno(), 0o600)
        key_file.truncate(0)
        json.dump(keys, key_file, indent=4)
        key_file.write("\n")


def set_key() -> None:
    """
    Add the OpenAI API key.
    """
    key_path = get_api_key_path()
    key = get_api_key()
    if key:
        click.secho("Error: ", fg="red", nl=False)
        click.echo("API key already exists.")
        click.echo(f"Use `{click.style('lmt key edit', fg='blue')}` to edit it.")
        return

    key = click.prompt("Your OpenAI API key", hide_input=True)
    write_key(key)
    click.secho("Success!", fg="green", nl=False)
    click.echo(" API key added.")
    click.echo(f"\nThe API key is stored in {key_path}.")


def edit_key() -> None:
    """
    Edit the OpenAI API key.
    """
    key_file_path = get_api_key_path()
    key = get_api_key()
    if not key:
        click.secho("Error: ", fg="red", nl=False)
        click.echo("API key does not exist.")
        click.echo("You will now be prompted to add it.\n")
        set_key()
        return

    original_key = key
    new_key = click.prompt("Your OpenAI API key", hide_input=True)
    if original_key != new_key:
        write_key(new_key)
        click.secho("Success!", fg="green", nl=False)
        click.echo(" API key was updated.")
    else:
        click.echo("No changes were made.")
    click.echo(f"\nThe API key is stored in {key_file_path}.")
