import sys
from pathlib import Path

import click
import yaml


def handle_template(template: str, system: str, prompt_input: str, model: str) -> tuple:
    """
    Handles the template used for the prompt.
    """
    template_content = get_template_content(template)
    system = update_from_template(template_content, "system", system)
    prompt_input = update_from_template(template_content, "user", prompt_input)
    model_template = template_content.get("model", model) or model

    return system, prompt_input, model_template


def update_from_template(template_content, key, value):
    """
    Updates the value of a key from a template.
    """
    existing_value = template_content.setdefault(key, "")
    if existing_value is None:
        template_content[key] = existing_value = ""

    return existing_value.rstrip() + (value or "")


def get_template_content(template):
    """
    Reads the template YAML file and returns a dictionary with its content.
    """
    template_file = TEMPLATES_DIR / f"{template}.yaml"

    try:
        with open(template_file, "rt", encoding="UTF-8") as file:
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
        with open(default_template_file, "w", encoding="UTF-8") as file:
            file.write(DEFAULT_TEMPLATE_CONTENT)

    return default_template_file


TEMPLATES_DIR = get_templates_dir()

DEFAULT_TEMPLATE_CONTENT = """# Documentation: https://github.com/sderev/lmt

# You may leave either of the fields empty. 


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
