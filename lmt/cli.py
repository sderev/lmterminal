import filecmp
import shutil
import sys

import click

from .lib import *

VALID_MODELS = {
    "chatgpt": "gpt-3.5-turbo",
    "chatgpt-16k": "gpt-3.5-turbo-16k",
    "3.5": "gpt-3.5-turbo",
    "3.5-16k": "gpt-3.5-turbo-16k",
    "4": "gpt-4",
    "gpt4": "gpt-4",
    "4-32k": "gpt-4-32k",
    "gpt4-32k": "gpt-4-32k",
}


def validate_model_name(ctx, param, value):
    """
    Validates the model name parameter.
    """
    model_name = value.lower()
    if model_name in VALID_MODELS:
        return VALID_MODELS[model_name]
    elif model_name in VALID_MODELS.values():
        return model_name
    else:
        raise click.BadParameter(f"Invalid model: {model_name}")


def validate_temperature(ctx, param, value):
    """
    Validates the temperature parameter.
    """
    if 0 <= value <= 2:
        return value
    else:
        raise click.BadParameter("Temperature must be between 0 and 2.")


@click.group()
@click.version_option()
def lmt():
    """
    Talk to ChatGPT.

    Documentation: https://github.com/sderev/lmt
    """
    pass


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
    prompt_input,
    debug,
):
    """
    Talk to ChatGPT.

    Example: lmt prompt "Say hello" --emoji
    """
    if not prompt_input:
        if template:
            pass
        elif not sys.stdin.isatty():
            prompt_input = sys.stdin.read()
        elif sys.stdin.isatty():
            click.echo(
                click.style(
                    (
                        "You can paste your prompt below. Press <Enter> to"
                        " validate.\nOnce you've done, press Ctrl+D (or Ctrl+Z on"
                        " Windows) to send it."
                    ),
                    fg="yellow",
                )
                + "\n---"
            )
            prompt_input = sys.stdin.read()
            click.echo()
    prompt_input = "".join(prompt_input).rstrip()

    if system and template:
        raise click.BadOptionUsage(
            option_name="template",
            message=click.style(
                "You cannot use both `--template` and `--system` at the same time.",
                fg="red",
            ),
        )

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


@lmt.group()
def templates():
    """
    Manage the templates.
    """
    pass


@click.command(name="template")
def template():
    templates()


@templates.command("list")
def print_templates_list():
    """
    List the available templates.
    """
    templates = [template.stem for template in TEMPLATES_DIR.iterdir()]
    if templates:
        click.echo("\n".join(templates))


@templates.command("view")
@click.argument("template")
def view_template(template):
    """
    View a template.
    """
    template = TEMPLATES_DIR / f"{template}.yaml"
    if template.exists():
        with open(template, "r") as template_file:
            click.echo(template_file.read())


@templates.command()
@click.argument("template")
def edit(template):
    """
    Edit a template.
    """
    template_file = TEMPLATES_DIR / f"{template}.yaml"
    if template_file.exists():
        click.edit(filename=template_file)
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

