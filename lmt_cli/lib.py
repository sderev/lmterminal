import sys
from pathlib import Path

import click
import openai
import yaml
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from lmt_cli import gpt_integration as openai_utils

RED = "\x1b[91m"
RESET = "\x1b[0m"


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
):
    """
    Handles the parameters.
    """
    if not system:
        system = ""

    if template:
        system, prompt_input, model_template = handle_template(
            template, system, prompt_input, model
        )

    if not model:
        model = model_template

    if emoji:
        system = add_emoji(system)

    prompt = openai_utils.format_prompt(system, prompt_input)

    if debug:
        display_debug_information(prompt, model, temperature)

    if tokens:
        display_tokens_count_and_cost(prompt, model)

    stream = False if no_stream else True

    generate_response(
        debug,
        emoji,
        model,
        prompt,
        raw,
        stream,
        system,
        temperature,
        template,
        tokens,
    )


def handle_template(template: str, system: str, prompt_input: str, model: str) -> tuple:
    """
    Handles the template parameter.
    """
    template_content = get_template_content(template)
    system = update_from_template(template_content, "system", system)
    prompt_input = update_from_template(template_content, "user", prompt_input)
    model = template_content.get("model", model) or model

    return system, prompt_input, model


def update_from_template(template_content, key, value):
    """
    Updates the value of a key from a template.
    """
    existing_value = template_content.setdefault(key, "")
    if existing_value is None:
        template_content[key] = existing_value = ""

    return template_content.get(key, value) + (value or "")


def add_emoji(system: str) -> str:
    """
    Adds an emoji to the system message.
    """
    emoji_message = (
        "Add plenty of emojis as a colorful way to convey emotions. However, don't"
        " mention it."
    )
    system = system.rstrip()

    if system == "":
        return emoji_message

    if not system.endswith("."):
        system += "."
    return system + " " + emoji_message


def generate_response(
    debug: bool = False,
    emoji: bool = False,
    model: str = "gpt-3.5-turbo",
    prompt: str = None,
    raw: bool = False,
    stream: bool = True,
    system: str = None,
    temperature: float = 1,
    template: str = None,
    tokens: bool = False,
):
    """
    Generates a response from a ChatGPT.
    """
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
            print(f"{RED}Error: {error}{RESET}")
            handle_rate_limit_error()

        except Exception as error:
            print(f"{RED}Error: {error}{RESET}")

        else:
            return content, response_time, response



def get_template_content(template):
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


def get_templates_dir() -> Path:
    """
    Returns the path to the templates directory.
    """
    templates_dir = Path.home() / ".config" / "lmt" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


def get_default_template_file_path() -> Path:
    """
    Returns the path to the default template file.
    """
    default_dir = Path.home() / ".config" / "lmt" / "default"
    default_dir.mkdir(parents=True, exist_ok=True)

    default_template_file = default_dir / "template.yaml"
    if not default_template_file.exists():
        click.echo("The default template does not exist. Creating it...")
        with open(default_template_file, "w") as file:
            file.write(DEFAULT_TEMPLATE_CONTENT)

    return default_template_file


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


def display_tokens_count_and_cost(prompt, model):
    """
    Displays the number of tokens in the prompt and the cost of the prompt.
    """
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


# Valid Models
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
