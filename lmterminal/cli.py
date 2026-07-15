import filecmp
import json
import shutil
import sys

import click
from click_default_group import DefaultGroup

from .lib import DEFAULT_MODEL, edit_key, prepare_and_generate_response, set_key
from .model_registry import REASONING_EFFORTS, get_valid_models, resolve_model_name
from .templates import TEMPLATES_DIR, get_default_template_file_path

RESERVED_REQUEST_OPTION_KEYS = {
    "messages": None,
    "model": "--model",
    "n": None,
    "reasoning_effort": "--reasoning-effort",
    "stream": "--no-stream",
    "temperature": "--temperature",
}

VALID_MODELS = get_valid_models()


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
    if 0 <= value <= 2:
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

    return options


@click.group(cls=DefaultGroup, default="prompt", default_if_no_args=True)
@click.version_option(package_name="lmterminal")
def lmt():
    """
    Talk to OpenAI models.

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
    callback=validate_model_name,
)
@click.option(
    "--template",
    "-t",
    help="The template to use for the requests.",
)
@click.option(
    "--system",
    "-s",
    help="The system to use for the requests.",
)
@click.option("--emoji", is_flag=True, help="Add emotions and emojis.")
@click.option(
    "--temperature",
    callback=validate_temperature,
    default=1,
    type=float,
    help="The temperature to use for the requests.",
    show_default=True,
)
@click.option(
    "--reasoning-effort",
    type=click.Choice(REASONING_EFFORTS),
    help="Set the reasoning effort for supported models.",
)
@click.option(
    "-o",
    "--option",
    "request_options",
    multiple=True,
    callback=parse_request_options,
    help="Pass additional Chat Completions request options as `key=value`.",
)
@click.option(
    "--tokens",
    is_flag=True,
    help=("Count the number of tokens in the prompt, and display the cost of the request."),
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
    help="Print debug information.",
)
@click.pass_context
def prompt(
    ctx,
    model,
    template,
    system,
    emoji,
    temperature,
    reasoning_effort,
    request_options,
    tokens,
    no_stream,
    raw,
    rich,
    prompt_input,
    debug,
):
    """
    Talk to OpenAI models.

    Example: lmt prompt "Say hello" --emoji
    """
    prompt_input = " ".join(prompt_input).strip()

    # Allow for the appending of an additional prompt to the piped stdin content
    if not sys.stdin.isatty() and prompt_input:
        prompt_input = sys.stdin.read().strip() + "\n___\n" + prompt_input

    if not prompt_input:
        # Read piped or redirected stdin content.
        if not sys.stdin.isatty():
            prompt_input = sys.stdin.read().strip()

        # Allow for structured prompts in the terminal.
        if sys.stdin.isatty():
            input_prompt_instructions = (
                "Write or paste your message below. Use <Enter> for new lines."
                "\nTo send your message, press Ctrl+D."
            )

            if sys.stdout.isatty():
                click.secho(input_prompt_instructions, fg="yellow")
                click.echo("---")

            # Display instructions in the terminal only, not in redirected or piped output.
            # This ensures the user sees the instructions without affecting the file output.
            if not sys.stdout.isatty():
                with open("/dev/tty", "w", encoding="UTF-8") as output_stream:
                    click.secho(input_prompt_instructions, fg="yellow", file=output_stream)
                    click.echo("---", file=output_stream)

            # Read user input from stdin
            prompt_input = sys.stdin.read().strip()

    if system and template:
        raise click.BadOptionUsage(
            option_name="template",
            message=click.style(
                "You cannot use both `--template` and `--system` at the same time.",
                fg="red",
            ),
        )

    conflicting_request_keys = RESERVED_REQUEST_OPTION_KEYS.keys() & request_options.keys()
    if conflicting_request_keys:
        conflicting_request_key = sorted(conflicting_request_keys)[0]
        dedicated_option = RESERVED_REQUEST_OPTION_KEYS[conflicting_request_key]
        if dedicated_option is None:
            detail = f"`-o/--option` cannot set `{conflicting_request_key}`."
        else:
            detail = (
                f"`-o/--option` cannot override `{conflicting_request_key}`."
                f" Use `{dedicated_option}` instead."
            )
        raise click.BadOptionUsage(
            option_name="option",
            message=click.style(detail, fg="red"),
        )

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
        system,
        template,
        model,
        emoji,
        prompt_input,
        temperature,
        reasoning_effort,
        request_options,
        tokens,
        no_stream,
        raw,
        debug,
    )

    # Same as above (readibility), but after the LLM's response
    if sys.stdout.isatty() and not no_stream:
        click.echo()


@lmt.command()
def models():
    """
    List the available models.
    """
    for model, aliases in VALID_MODELS.items():
        click.echo(model)
        if aliases:
            if len(aliases) == 1:
                click.echo(f"  Alias: {aliases[0]}")
            click.echo(f"  Aliases: {', '.join(aliases)}")


@lmt.group()
def templates():
    """
    Manage the templates.
    """


@templates.command("list")
def print_templates_list():
    """
    List the available templates.
    """
    templates_names_list = sorted([template.stem for template in TEMPLATES_DIR.iterdir()])
    if templates_names_list:
        click.echo("\n".join(templates_names_list))


@templates.command("view")
@click.argument("template")
def view_template(template):
    """
    View a template.
    """
    template = TEMPLATES_DIR / f"{template}.yaml"
    if template.exists():
        with open(template, "r", encoding="UTF-8") as template_file:
            click.echo(template_file.read())


@templates.command()
@click.argument("template")
def edit(template):
    """
    Edit a template.
    """
    template_file = TEMPLATES_DIR / f"{template}.yaml"
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
        if template in [template.name for template in TEMPLATES_DIR.iterdir()]:
            click.secho("Error: ", fg="red", nl=False)
            click.echo("Template ", nl=False)
            click.secho(template, fg="red", nl=False)
            click.echo(" already exists.")
            click.echo(
                f"Use `{click.style(f'lmt templates edit {template}', fg='blue')}` to edit it."
            )
            return

    template_file = TEMPLATES_DIR / f"{template}.yaml"
    default_template_file = get_default_template_file_path()

    shutil.copyfile(default_template_file, template_file)

    click.edit(filename=str(template_file))

    if filecmp.cmp(default_template_file, template_file, shallow=False):
        click.secho("Aborting: ", fg="red", nl=False)
        click.echo("The template has not been created because no changes were made.")
        template_file.unlink()
    else:
        click.echo(
            f"{click.style('Success!', fg='green')} Template"
            f" '{click.style(template, fg='green')}' created."
        )


@templates.command("delete")
@click.argument("template", required=True)
def delete_template(template):
    """
    Delete the template.
    """
    template_file = TEMPLATES_DIR / f"{template}.yaml"
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
@click.argument("template", required=True)
def rename_template(template):
    """
    Rename the template.
    """
    template_file = TEMPLATES_DIR / template
    if template_file.exists():
        new_template_name = click.prompt("New template name", default=template)
        new_template_file = TEMPLATES_DIR / new_template_name
        template_file.rename(new_template_file)
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
