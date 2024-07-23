import filecmp
import shutil
import sys

import click
from click_default_group import DefaultGroup

from .lib import *
from .templates import TEMPLATES_DIR, get_default_template_file_path

VALID_MODELS = {
    "chatgpt": "gpt-3.5-turbo-1106",
    "3.5": "gpt-3.5-turbo",
    "3.5-0613": "gpt-3.5-turbo-0613",
    "3.5-16k-0613": "gpt-3.5-turbo-16k-0613",
    "gpt-instruct": "gpt-3.5-turbo-instruct",
    "4": "gpt-4",
    "gpt4": "gpt-4",
    "4-turbo": "gpt-4-1106-preview",
    "gpt4-turbo": "gpt-4-1106-preview",
    "4-32k": "gpt-4-32k-0613",
    "gpt4-32k": "gpt-4-32k-0613",
}


# The first two parameters are required by Click for a callback.
def validate_model_name(ctx, param, value):
    """
    Validates the model name parameter.
    """
    model_name = value.lower()

    if model_name in VALID_MODELS:
        return VALID_MODELS[model_name]

    if model_name in VALID_MODELS.values():
        return model_name

    valid_model_names = "\n".join(sorted(set(VALID_MODELS.values())))
    aliases_model_names = "\n".join(sorted(VALID_MODELS.keys()))

    error_message = (
        f"{click.style('Invalid model name:', fg='red')} {model_name}\n\n"
        f"{click.style('Valid models are:', fg='blue')}\n{valid_model_names}\n\n"
        f"{click.style('You can also use the aliases:', fg='blue')}\n{aliases_model_names}\n"
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


@click.group(cls=DefaultGroup, default="prompt", default_if_no_args=True)
@click.version_option(package_name="lmt-cli")
def lmt():
    """
    Talk to ChatGPT.

    Documentation: https://github.com/sderev/lmt
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
    default="gpt-3.5-turbo",
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
    "--tokens",
    is_flag=True,
    help=(
        "Count the number of tokens in the prompt, and display the cost of the request."
    ),
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
    tokens,
    no_stream,
    raw,
    rich,
    prompt_input,
    debug,
):
    """
    Talk to ChatGPT.

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
                click.echo(click.style(input_prompt_instructions, fg="yellow"))
                click.echo("---")

            # Display instructions in the terminal only, not in redirected or piped output.
            # This ensures the user sees the instructions without affecting the file output.
            if not sys.stdout.isatty():
                with open("/dev/tty", "w", encoding="UTF-8") as output_stream:
                    click.echo(
                        click.style(input_prompt_instructions, fg="yellow"),
                        file=output_stream,
                    )
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
        tokens,
        no_stream,
        raw,
        debug,
    )

    # Same as above (readibility), but after the LLM's response
    if sys.stdout.isatty():
        click.echo()


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
    templates_names_list = sorted(
        [template.stem for template in TEMPLATES_DIR.iterdir()]
    )
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
        click.edit(filename=template_file)

        if original_file_content == template_file.read_text():
            click.echo("No changes were made.")
        else:
            click.echo(
                f"{click.style('Success!', fg='green')} Template"
                f" {click.style(template, fg='green')} was updated."
            )

    else:
        click.echo(
            click.style("Error: ", fg="red")
            + f"Template {click.style(template, fg='red')} does not exist."
        )
        click.echo(
            f"Use `{click.style(f'lmt templates add {template}', fg='blue')}` to"
            " create it."
        )


@templates.command("add")
@click.argument("template", required=False)
def add_template(template):
    """
    Create a new template
    """
    if not template:
        template = click.prompt("Template name")
        if template in [template.name for template in TEMPLATES_DIR.iterdir()]:
            click.echo(
                click.style("Error: ", fg="red")
                + f"Template {click.style(template, fg='red')} already exists."
            )
            click.echo(
                f"Use `{click.style(f'lmt templates edit {template}', fg='blue')}` to"
                " edit it."
            )
            return

    template_file = TEMPLATES_DIR / f"{template}.yaml"
    default_template_file = get_default_template_file_path()

    shutil.copyfile(default_template_file, template_file)

    click.edit(filename=template_file)

    if filecmp.cmp(default_template_file, template_file, shallow=False):
        click.echo(
            click.style("Aborting: ", fg="red")
            + "The template has not been created because no changes were made."
        )
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
        click.echo(
            click.style("Error: ", fg="red")
            + f"The template '{click.style(template, fg='red')}' does not exist."
        )


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
        click.echo(
            click.style("Error: ", fg="red")
            + f"The template '{template}' does not exist."
        )


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
