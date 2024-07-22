import sys
from importlib.resources import read_text
from pathlib import Path

import click
import yaml
from lmterminal import DEFAULT_MODEL


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
    template_file = get_templates_dir() / f"{template}.yaml"

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


def get_starter_template_file_path() -> Path:
    """
    Returns the path to the default template file.

    Returns:
        Path: The path object representing the starter template file.
    """
    starter_template_dir = Path.home() / ".config" / "lmt" / "templates" / "starter"
    starter_template_dir.mkdir(parents=True, exist_ok=True)

    starter_template_path = starter_template_dir / "starter_template.yaml"
    if not starter_template_path.exists():
        starter_template_content = read_text("lmterminal.starter_template", "starter_template.yaml")
        with open(starter_template_path, "w", encoding="UTF-8") as file:
            file.write(starter_template_content)

    return starter_template_path


def get_template_names(ctx=None, param=None, incomplete=None) -> List[str]:
    """
    Returns a list of template names. The list is sorted alphabetically.

    This function can be used for shell completion.
    The arguments are only used for shell completion; they are given by `Click`.
 
    Args:
        ctx: Click context.
        param: Click parameter.
        incomplete: The incomplete template name.

    Returns:
        List[str]: The list of template names.
    """
    templates = sorted([template.stem for template in TEMPLATES_DIR.iterdir()])
    if ctx and param and incomplete:  # For shell completion.
        return [name for name in templates if name.startswith(incomplete)]
    else:
        return templates  # For plain listing.

TEMPLATES_DIR = get_templates_dir()
