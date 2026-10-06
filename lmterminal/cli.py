import json
import os
import sys

import click
from click.core import ParameterSource
from click_default_group import DefaultGroup

from .diagnostics import RequestDiagnostics
from .lib import (
    DEFAULT_MODEL,
    RequestResolutionError,
    edit_key,
    prepare_and_generate_response,
    resolve_request,
    set_key,
)
from .model_registry import REASONING_EFFORTS, get_valid_models, resolve_model_name
from .request_options import UNSET, validate_request_options
from .templates import (
    DEFAULT_TEMPLATE_CONTENT,
    TemplateError,
    list_templates,
    load_template,
    template_path,
)


def complete_template(ctx, param, incomplete):
    return [name for name in list_templates() if name.startswith(incomplete)]


def _template_path(name):
    try:
        return template_path(name)
    except TemplateError as error:
        raise click.UsageError(str(error)) from error


# The first two parameters are required by Click for a callback.
def validate_model_name(ctx, param, value):
    """
    Validates the model name parameter.
    """
    canonical_model_name = resolve_model_name(value)
    if canonical_model_name is not None:
        return canonical_model_name

    error_message = (
        f"{click.style('Invalid model name.', fg='red')}\n"
        f"{click.style('To see the model names and their aliases, use:', fg='blue')} lmt models"
    )

    raise click.BadParameter(error_message)


# The first two parameters are required by Click for a callback.
def validate_temperature(ctx, param, value):
    """
    Validates the temperature parameter.
    """
    if value is None or 0 <= value <= 2:
        return value

    raise click.BadParameter("Temperature must be between 0 and 2.")


def parse_request_option_value(raw_value):
    """Parses a request option value from the CLI."""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def add_request_option(options, key, value):
    """Adds a parsed request option to a nested mapping."""
    key_parts = key.split(".")
    if any(not key_part for key_part in key_parts):
        raise click.BadParameter("Option keys cannot contain empty path segments.")

    current_level = options
    for key_part in key_parts[:-1]:
        if key_part not in current_level:
            current_level[key_part] = {}
        existing_value = current_level[key_part]
        if not isinstance(existing_value, dict):
            raise click.BadParameter(
                f"Option `{key}` conflicts with an existing non-object option."
            )
        current_level = existing_value

    leaf_key = key_parts[-1]
    if leaf_key in current_level:
        raise click.BadParameter(f"Option `{key}` was provided more than once.")
    current_level[leaf_key] = value


def parse_request_options(ctx, param, values):
    """Parses repeatable `key=value` request options from the CLI."""
    options = {}

    for raw_option in values:
        if "=" not in raw_option:
            raise click.BadParameter("Options must use the `key=value` form.")

        key, raw_value = raw_option.split("=", 1)
        if not key:
            raise click.BadParameter("Option keys cannot be empty.")

        add_request_option(options, key, parse_request_option_value(raw_value))

    try:
        validate_request_options(options)
    except (TypeError, ValueError) as error:
        raise click.BadParameter(str(error)) from error
    return options


@click.group(cls=DefaultGroup, default="prompt", default_if_no_args=True)
@click.version_option(package_name="lmterminal")
def lmt():
    """
    Talk to ChatGPT.

    Use lmt prompt --help for prompt options, including --reasoning-effort.
    Reasoning effort is model-dependent.

    Documentation: https://github.com/sderev/lmterminal
    """


@lmt.command()
@click.argument(
    "prompt_input",
    type=str,
    required=False,
    nargs=-1,
)
@click.option(
    "--model",
    "-m",
    default=DEFAULT_MODEL,
    help="The model to use for the requests.",
    show_default=True,
    callback=validate_model_name,
)
@click.option(
    "--template",
    "-t",
    help="Named YAML template; cannot be combined with --system.",
    shell_complete=complete_template,
)
@click.option(
    "--system",
    "-s",
    help="The system to use for the requests.",
)
@click.option("--text", help="Literal content; positional arguments supply task instructions.")
@click.option("--emoji", is_flag=True, help="Add emotions and emojis.")
@click.option(
    "--temperature",
    callback=validate_temperature,
    default=None,
    type=float,
    help=(
        "Sampling temperature: defaults to 1 where supported; omitted for restricted "
        "or unverified models. Explicit values are validated locally when known, "
        "otherwise by the API."
    ),
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(REASONING_EFFORTS),
    help=(
        "Model-dependent reasoning effort. Use none to disable reasoning where supported. "
        "Omitted model/effort use gpt-6-luna with none; "
        "explicit models retain their provider default when effort is omitted."
    ),
)
@click.option(
    "-o",
    "--option",
    "request_options",
    multiple=True,
    callback=parse_request_options,
    help="Pass additional Chat Completions options as key=value (JSON values or text).",
)
@click.option(
    "--tokens",
    is_flag=True,
    help=("Estimate input tokens and Standard uncached input cost locally."),
)
@click.option(
    "--no-stream",
    is_flag=True,
    default=False,
    help="Disable the streaming of the response.",
)
@click.option(
    "--raw",
    "-r",
    is_flag=True,
    default=False,
    help="Disable colors and formatting, and print the raw response.",
)
@click.option(
    "--rich",
    "-R",
    is_flag=True,
    default=False,
    help="Force Rich formatting.",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Print prompts, model and requested temperature to stderr.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Request timings on stderr; repeat for setup/events (-vv) and usage counts (-vvv).",
)
@click.pass_context
def prompt(
    ctx,
    model,
    template,
    system,
    emoji,
    text,
    temperature,
    reasoning_effort,
    request_options,
    tokens,
    no_stream,
    raw,
    rich,
    prompt_input,
    debug,
    verbose,
):
    """
    Talk to ChatGPT.

    Example: lmt prompt "Say hello" --emoji
    """
    diagnostics = RequestDiagnostics(verbose) if verbose else None
    if diagnostics:
        diagnostics.mark("prompt handling started", level=2)
    prompt_input = " ".join(prompt_input)

    def supplied(name, value):
        return UNSET if ctx.get_parameter_source(name) is ParameterSource.DEFAULT else value

    settings = {
        "system": supplied("system", system),
        "model": supplied("model", model),
        "temperature": supplied("temperature", temperature),
        "reasoning_effort": supplied("reasoning_effort", reasoning_effort),
        "request_options": request_options,
    }
    if diagnostics:
        diagnostics.mark("request preparation started", level=2)
    try:
        # Conflict and template/control validation must precede any stdin/key access.
        if template is not None and settings["system"] is not UNSET:
            resolve_request(template=template, **settings)
        stored = load_template(template) if template is not None else None
        request = resolve_request(template=stored, prompt=prompt_input, text=text or "", **settings)
    except (TemplateError, RequestResolutionError) as error:
        raise click.UsageError(str(error)) from error

    content = text or ""
    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        if piped and text is not None:
            raise click.UsageError("Cannot combine `--text` with nonempty stdin.")
        content = piped if text is None else text
    elif not request.messages[-1]["content"]:
        instructions = (
            "Write or paste your message below. Use <Enter> for new lines."
            "\nTo send your message, press Ctrl+D."
        )
        if sys.stdout.isatty():
            click.secho(instructions, fg="yellow")
            click.echo("---")
        else:
            with open("/dev/tty", "w", encoding="UTF-8") as output_stream:
                click.secho(instructions, fg="yellow", file=output_stream)
                click.echo("---", file=output_stream)
        content = sys.stdin.read()
    try:
        request = resolve_request(template=stored, prompt=prompt_input, text=content, **settings)
    except (TemplateError, RequestResolutionError) as error:
        raise click.UsageError(str(error)) from error

    # If *not* in an interactive shell or redirecting to a file,
    # enable the `--raw` option, viz. disabling `Rich` formatting
    if not sys.stdout.isatty():
        raw = True

    # If `--rich` is enabled, force `--raw` to be disabled
    if rich:
        raw = False

    # If in an interactive shell, add a new line after the prompt for better readability
    if sys.stdout.isatty():
        click.echo()

    prepare_and_generate_response(
        request,
        emoji=emoji,
        tokens=tokens,
        no_stream=no_stream,
        raw=raw,
        debug=debug,
        diagnostics=diagnostics,
    )

    # Same as above (readibility), but after the LLM's response
    if sys.stdout.isatty() and not no_stream:
        click.echo()


@lmt.command()
def models():
    """
    List the available models.
    """
    for model, aliases in get_valid_models().items():
        click.echo(model)
        if aliases:
            if len(aliases) == 1:
                click.echo(f"  Alias: {aliases[0]}")
            else:
                click.echo(f"  Aliases: {', '.join(aliases)}")


@lmt.group()
def templates():
    """
    Manage the templates.
    """


lmt.add_command(templates, name="template")


@templates.command("list")
def print_templates_list():
    """
    List the available templates.
    """
    templates_names_list = list_templates()
    if templates_names_list:
        click.echo("\n".join(templates_names_list))


@templates.command("view")
@click.argument("template", shell_complete=complete_template)
def view_template(template):
    """
    View a template.
    """
    template = _template_path(template)
    if template.exists():
        with open(template, "r", encoding="UTF-8") as template_file:
            click.echo(template_file.read())


@templates.command()
@click.argument("template", shell_complete=complete_template)
def edit(template):
    """
    Edit a template.
    """
    template_file = _template_path(template)
    if template_file.exists():
        original_file_content = template_file.read_text()
        click.edit(filename=str(template_file))

        if original_file_content == template_file.read_text():
            click.echo("No changes were made.")
        else:
            click.echo(
                f"{click.style('Success!', fg='green')} Template"
                f" {click.style(template, fg='green')} was updated."
            )

    else:
        click.secho("Error: ", fg="red", nl=False)
        click.echo("Template ", nl=False)
        click.secho(template, fg="red", nl=False)
        click.echo(" does not exist.")
        click.echo(f"Use `{click.style(f'lmt templates add {template}', fg='blue')}` to create it.")


@templates.command("add")
@click.argument("template", required=False)
def add_template(template):
    """
    Create a new template
    """
    if not template:
        template = click.prompt("Template name")
    template_file = _template_path(template)
    template_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with template_file.open("x", encoding="UTF-8") as file:
            file.write(DEFAULT_TEMPLATE_CONTENT)
    except FileExistsError as error:
        raise click.UsageError(f"Template `{template}` already exists.") from error
    click.edit(filename=str(template_file))
    if template_file.read_text(encoding="UTF-8") == DEFAULT_TEMPLATE_CONTENT:
        click.secho("Aborting: ", fg="red", nl=False)
        click.echo("The template has not been created because no changes were made.")
        template_file.unlink()
    else:
        click.echo(
            f"{click.style('Success!', fg='green')} Template"
            f" '{click.style(template, fg='green')}' created."
        )


@templates.command("delete")
@click.argument("template", required=True, shell_complete=complete_template)
def delete_template(template):
    """
    Delete the template.
    """
    template_file = _template_path(template)
    if template_file.exists():
        click.confirm(
            f"Are you sure you want to delete the template '{template}'?",
            abort=True,
        )
        template_file.unlink()
        click.echo(
            f"{click.style('Success!', fg='green')} Template"
            f" '{click.style(template, fg='blue')}' deleted."
        )
    else:
        click.secho("Error: ", fg="red", nl=False)
        click.echo("The template '", nl=False)
        click.secho(template, fg="red", nl=False)
        click.echo("' does not exist.")


@templates.command("rename")
@click.argument("template", required=True, shell_complete=complete_template)
def rename_template(template):
    """
    Rename the template.
    """
    template_file = _template_path(template)
    if template_file.exists():
        new_template_name = click.prompt("New template name", default=template)
        new_template_file = _template_path(new_template_name)
        try:
            # A hard link refuses an occupied destination, unlike rename on Linux.
            os.link(template_file, new_template_file, follow_symlinks=False)
        except FileExistsError as error:
            raise click.UsageError(f"Template `{new_template_name}` already exists.") from error
        template_file.unlink()
        click.echo(
            f"{click.style('Success!', fg='green')} Template"
            f" '{click.style(template, fg='blue')}' renamed to"
            f" '{click.style(new_template_name, fg='green')}'."
        )
    else:
        click.secho("Error: ", fg="red", nl=False)
        click.echo(f"The template '{template}' does not exist.")


@lmt.group()
def key():
    """
    Manage the OpenAI API key.
    """


@key.command(name="edit")
def edit_api_key():
    """
    Edit the OpenAI API key.
    """
    edit_key()


@key.command(name="set")
def set_api_key():
    """
    Set the OpenAI API key.
    """
    set_key()
