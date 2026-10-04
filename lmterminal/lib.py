import json
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
from .estimation import estimate_request
from .model_registry import resolve_model_name
from .request_options import prepare_request
from .templates import handle_template

BLUE = "\x1b[34m"
RED = "\x1b[91m"
RESET = "\x1b[0m"

DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_CODE_BLOCK_THEME = "monokai"
DEFAULT_INLINE_CODE_THEME = "blue on black"


def prepare_and_generate_response(
    system: str,
    template: str,
    model: str,
    emoji: bool,
    prompt_input: str,
    temperature: float,
    tokens: bool,
    no_stream: bool,
    raw: bool,
    debug: bool,
    *,
    reasoning_effort: str | None = None,
    request_options: dict | None = None,
):
    """
    Handles the parameters.
    """
    if not system:
        system = ""

    if template:
        system, prompt_input, template_model = handle_template(
            template, system, prompt_input, model
        )
        # If a model name is given in the options,
        # it will bypass the model name in the template.
        if model == DEFAULT_MODEL:
            model = resolve_model_name(template_model) if isinstance(template_model, str) else None
            if model is None:
                raise click.BadParameter(f"Invalid template model name: {template_model!r}")

    if emoji:
        system = add_emoji(system)

    prompt = openai_utils.format_prompt(system, prompt_input)

    # Temporary reformatting of the prompt for `o1` models
    # as they don't support system messages yet.
    if "o1" in model:
        prompt = [
            {
                "role": "user",
                "content": prompt_input,
            },
        ]

    request = _prepare_request(model, prompt, temperature, reasoning_effort, request_options)
    if debug:
        display_debug_information(request.messages, request.model, temperature)

    if tokens:
        display_tokens_count_and_cost(request)

    return _generate_prepared_response(request, raw, not no_stream)


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
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    raw: bool = False,
    stream: bool = True,
    temperature: float = 1,
    *,
    reasoning_effort: str | None = None,
    request_options: dict | None = None,
):
    """
    Generates a response from a ChatGPT.
    """
    request = _prepare_request(model, prompt, temperature, reasoning_effort, request_options)
    return _generate_prepared_response(request, raw, stream)


def _generate_prepared_response(request, raw, stream):
    api_key = get_api_key()

    if not api_key:
        click.secho("Error:", fg="red", nl=False)
        click.echo(" You need to set your OpenAI API key.")
        click.echo("You can do so by running:", nl=False)
        click.echo(f"  {click.style('lmt key set', fg='blue')}\n")
        sys.exit(1)

    # Theming for Rich Markdown
    code_block_theme = get_markdown_code_block_theme()
    inline_code_theme = get_markdown_inline_code_theme()
    custom_theme = Theme({"markdown.code": inline_code_theme})

    console = Console(theme=custom_theme)
    markdown_stream = ""
    use_live_markdown = (
        stream
        and not raw
        and getattr(sys.stdout, "isatty", lambda: False)()
        and console.is_terminal
        and console.is_interactive
        and not console.is_dumb_terminal
    )
    live_context = (
        Live("", console=console, auto_refresh=False) if use_live_markdown else nullcontext()
    )
    with live_context as live:

        def update_markdown_stream(chunk: str) -> None:
            nonlocal markdown_stream
            if not chunk:
                return
            markdown_stream += chunk
            live.update(
                Markdown(markdown_stream, code_theme=code_block_theme),
                refresh=True,
            )

        try:
            content, response_time, response = openai_utils.send_prepared_request(
                api_key=api_key,
                request=request,
                stream=stream,
                update_markdown_stream=update_markdown_stream if use_live_markdown else None,
            )

            # This is temporary to ensure that the last line always ends with a newline
            # This will be removed when refactored
            if not content.endswith("\n"):
                content += "\n"
            #############################

            if not stream:
                print(content, end="")

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

        else:
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
    click.echo(f"Model: {estimate.model}")
    if estimate.input_tokens is not None:
        click.echo(f"Estimated input tokens: ~{estimate.input_tokens}")
    elif estimate.message_tokens is not None:
        click.echo(f"Message-only token estimate: ~{estimate.message_tokens}")
        click.echo("Request input tokens and cost: unavailable")
    else:
        click.echo("Request input tokens and cost: unavailable")
    if estimate.input_cost_usd is not None:
        click.echo(f"Standard uncached input cost estimate: USD {estimate.input_cost_usd:f}")
        click.echo(f"Input rate: USD {estimate.input_rate_usd_per_million:f} / 1M tokens")
    elif estimate.input_tokens is not None:
        click.echo("Input cost: unavailable")
    if estimate.pricing_context:
        click.echo(f"Pricing tier: {estimate.pricing_context} context, based on estimated tokens.")
    for reason in estimate.warnings:
        click.echo(f"Note: {reason}")
    click.echo("Local message framing is heuristic; provider usage may differ.")
    click.echo("Actual input may cost less with cached tokens; cache hits are not predicted.")
    click.echo("Excludes output/reasoning, tool fees and service-tier adjustments; not a bill.")
    sys.exit(1 if estimate.message_tokens is None else 0)


def get_api_key() -> str:
    """
    Return the OpenAI API key.
    """
    key_file_path = get_api_key_path()
    with open(key_file_path, "r", encoding="UTF-8") as key_file:
        return key_file.read().strip()


def get_api_key_path() -> Path:
    """
    Return the path to the keys file.
    """
    key_file_path = Path.home() / ".config" / "lmt" / "key.env"
    if not key_file_path.exists():
        key_file_path.parent.mkdir(parents=True, exist_ok=True)
        key_file_path.touch()
    return key_file_path


def write_key(key: str) -> None:
    """
    Write the OpenAI API key to the key file.
    """
    key_file_path = get_api_key_path()
    with open(key_file_path, "w", encoding="UTF-8") as key_file:
        key_file.write(key)


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
