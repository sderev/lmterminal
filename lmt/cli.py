import filecmp
import shutil
import subprocess
import sys
from pathlib import Path

import click
import openai
import yaml
from click.exceptions import BadOptionUsage
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from lmt import gpt_integration as openai_utils

MODEL_ALIASES = {
    "chatgpt": "gpt-3.5-turbo",
    "chatgpt-16k": "gpt-3.5-turbo-16k",
    "3.5": "gpt-3.5-turbo",
    "3.5-16k": "gpt-3.5-turbo-16k",
    "4": "gpt-4",
    "gpt4": "gpt-4",
    "4-32k": "gpt-4-32k",
    "gpt4-32k": "gpt-4-32k",
}

EMOJI = (
    ". Add plenty of emojis as a colorful way to convey emotions. However, don't"
    " mention it."
)


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
    "prompt",
    type=str,
    required=False,
    nargs=-1,
)
@click.option(
    "--model",
    "-m",
    default="gpt-3.5-turbo",
    help="The model to use for the requests.",
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
    prompt,
    debug,
):
    """
    Talk to ChatGPT.

    Example: lmt prompt "Say hello" --emoji
    """
    if not prompt:
        if template:
            pass
        elif not sys.stdin.isatty():
            prompt = sys.stdin.read()
        elif sys.stdin.isatty():
            click.echo(
                click.style(
                    (
                        "You can paste your prompt below.\nOnce you've done, press"
                        " Ctrl+D (or Ctrl+Z on Windows) to send it."
                    ),
                    fg="yellow",
                )
                + "\n---"
            )
            prompt = sys.stdin.read()
            click.echo()
    prompt = "".join(prompt)

    if not system:
        system = ""

    if template and system:
        raise click.BadOptionUsage(
            option_name="template",
            message=click.style(
                "You cannot use both `--template` and `--system` at the same time.",
                fg="red",
            ),
        )

    if template:
        template_content = get_template(template)
        system = update_from_template(template_content, "system", system)
        prompt = update_from_template(template_content, "user", prompt)
        model = template_content.get("model", model) or model

    if emoji:
        system += EMOJI

    if model:
        model = get_model_name(model.lower())

    prompt = openai_utils.format_prompt(system, prompt)

    if debug:
        display_debug_information(prompt, model, temperature)

    if tokens:
        full_prompt = prompt[0]["content"] + prompt[1]["content"]
        number_of_tokens = openai_utils.num_tokens_from_string(full_prompt, model)
        cost = openai_utils.estimate_prompt_cost(prompt)[model]

        click.echo(
            "Number of tokens in the prompt:"
            f" {click.style(str(number_of_tokens), fg='yellow')}."
        )
        click.echo(
            f"Cost of the prompt for the {click.style(model, fg='blue')} model is:"
            f" {click.style(f'${cost}', fg='yellow')}."
        )
        click.echo(
            "Please note that this cost applies only to the prompt, not the"
            " subsequent response."
        )
        sys.exit(0)

    stream = False if no_stream else True

    markdown_stream = ""
    with Live(Markdown(markdown_stream), refresh_per_second=25) as live:

        def update_markdown_stream(chunk: str) -> None:
            nonlocal markdown_stream
            markdown_stream += chunk
            if raw:
                print("".join(chunk), end="")
            else:
                rich_markdown_stream = Markdown(markdown_stream)
                live.update(rich_markdown_stream)

        try:
            content, response_time, response = openai_utils.chatgpt_request(
                prompt,
                model=model,
                stream=stream,
                temperature=temperature,
                update_markdown_stream=update_markdown_stream,
            )
            if not stream:
                print(content)

        except openai.error.RateLimitError as error:
            click.echo(click.style("Error: ", fg="red") + f"{error}")
            handle_rate_limit_error()
        except Exception as error:
            click.echo(click.style(f"Error: {error}", fg="red"))


def update_from_template(template_content, key, value):
    if template_content.get(key, value) is None:
        template_content[key] = ""
    if value:
        value = template_content.get(key, value) + value
    else:
        value = template_content.get(key, value)
    return value


@lmt.group()
def templates():
    """
    Manage templates.
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
    templates = [template.name for template in TEMPLATES_DIR.iterdir()]
    click.echo("\n".join(templates))


@templates.command("view")
def view_template(template):
    """
    View a template.
    """
    template = TEMPLATES_DIR / template
    if template.exists():
        subprocess.run(["cat", template])


@templates.command()
@click.argument("template")
def edit(template):
    """
    Edit a template.
    """
    template_file = TEMPLATES_DIR / template
    if template_file.exists():
        click.edit(filename=template_file)
    else:
        click.echo(
            click.style("Error: ", fg="red")
            + "Template {click.style(template, fg='red')} does not exist."
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
    default_template_file = get_default_template_file()

    # Copy the default template to the new file
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


def get_template(template):
    """
    Reads the template YAML file and returns a dictionary with its content.
    """
    template_file = TEMPLATES_DIR / f"{template}.yaml"

    try:
        with open(template_file, "rt") as file:
            template_content = yaml.safe_load(file)
    except FileNotFoundError:
        click.echo(
            click.style("Error: ", fg="red")
            + f"The template '{click.style(template, fg='red')}' does not exist."
        )
        sys.exit(1)
    else:
        return template_content


def get_templates_dir():
    """
    Returns the path to the templates directory.
    """
    templates_dir = Path.home() / ".config" / "lmt" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


def get_default_template_file():
    """
    Returns the path to the default template.
    """
    default_dir = Path.home() / ".config" / "lmt" / "default"
    default_dir.mkdir(parents=True, exist_ok=True)

    default_template_file = default_dir / "template.yaml"
    if not default_template_file.exists():
        click.echo("The default template does not exist. Creating it...")
        with open(default_template_file, "w") as file:
            file.write(DEFAULT_TEMPLATE_CONTENT)
    return default_template_file


def get_model_name(model_name):
    if model_name in MODEL_ALIASES:
        return MODEL_ALIASES[model_name]
    elif model_name in MODEL_ALIASES.values():
        return model_name
    else:
        click.echo(
            click.style("Error: '", fg="red")
            + click.style(f"{model_name}", fg="blue")
            + click.style("' not found.", fg="red")
        )
        click.echo(f"Please check the spelling and try again.")
        sys.exit(1)


def display_debug_information(prompt, model, temperature):
    """
    Displays debug information.
    """
    click.echo()
    click.echo("---\n" + click.style("Debug information:", fg="yellow"))
    click.echo()

    click.echo(click.style("Prompt:", fg="red"), nl=False)
    for role in prompt:
        click.echo()
        click.echo(click.style(f"{role['role']}:", fg="blue"))
        click.echo(f"{role}")
    click.echo()

    click.echo(click.style("Model:", fg="red"))
    click.echo(f"{model=}")
    click.echo()

    click.echo(click.style("Temperature:", fg="red"))
    click.echo(f"{temperature=}")
    click.echo()

    click.echo(click.style("End of debug information.", fg="yellow"))
    click.echo("---\n")


def handle_rate_limit_error():
    """
    Provides guidance on how to handle a rate limit error.
    """
    click.echo()
    click.echo(
        click.style(
            (
                "You might not have set a usage rate limit in your"
                " OpenAI account settings. "
            ),
            fg="blue",
        )
    )
    click.echo(
        "If that's the case, you can set it"
        " here:\nhttps://platform.openai.com/account/billing/limits"
    )

    click.echo()
    click.echo(
        click.style(
            "If you have set a usage rate limit, please try the following steps:",
            fg="blue",
        )
    )
    click.echo("- Wait a few seconds before trying again.")
    click.echo()
    click.echo(
        "- Reduce your request rate or batch tokens. You can read the"
        " OpenAI rate limits"
        " here:\nhttps://platform.openai.com/account/rate-limits"
    )
    click.echo()
    click.echo(
        "- If you are using the free plan, you can upgrade to the paid"
        " plan"
        " here:\nhttps://platform.openai.com/account/billing/overview"
    )
    click.echo()
    click.echo(
        "- If you are using the paid plan, you can increase your usage"
        " rate limit"
        " here:\nhttps://platform.openai.com/account/billing/limits"
    )
    click.echo()


TEMPLATES_DIR = get_templates_dir()

DEFAULT_TEMPLATE_CONTENT = """# Documentation: https://github.com/sderev/lmt

# You can leave either of the fields empty. 


# Here, you can instruct how you want ChatGPT to behave.
# For example, you might say:
# "You are an AI modeled after Emil Cioran, the Romanian philosopher and essayist.
# You have a deep understanding of existentialism and philosophical pessimism."
# The more precise and detailed, the better!
system:




# Here, you should write what you want to say to ChatGPT.
# For example: "As a language model yourself, how do you view your own existence?"
user:




# You can change the model according to the list below.
# The default model is "gpt-3.5-turbo".
# More advanced models might provide more accurate or detailed responses,
# but they also have a higher "cost per 1K tokens". Tokens refer to chunks of text that AI models read. More tokens usually mean more cost.
model: "gpt-3.5-turbo"


# Model Aliases
#
# "chatgpt" or "gpt-3.5-turbo"
# "chatgpt-16k" or "gpt-3.5-turbo-16k"
#
# "3.5" or "gpt-3.5-turbo"
# "3.5-16k" or "gpt-3.5-turbo-16k"
#
# "4" or "gpt-4"
# "gpt4" or "gpt-4"
#
# "4-32k" or "gpt-4-32k"
# "gpt4-32k" or "gpt-4-32k"
"""
