import sys
from importlib.resources import read_text
from pathlib import Path
from typing import Dict, Optional, Tuple

import click
import yaml


def handle_template(
    template: str, system: str, prompt_input: str, model: str
) -> Tuple[str, str, str]:
    """
    Handles the template used for the prompt.
    """
    template_content = get_template_content(template)
    system = update_from_template(template_content, "system", system)
    prompt_input = update_from_template(template_content, "user", prompt_input)
    model_template = template_content.get("model", model) or model

    return system, prompt_input, model_template


def update_from_template(
    template_content: dict, key: str, prompt: Optional[str]
) -> str:
    """
    Updates the value of a key from a template.
    """
    existing_value = template_content.setdefault(key, "")
    if existing_value is None:
        template_content[key] = existing_value = ""

    return existing_value.rstrip() + (prompt or "")


def get_template_content(template: str) -> Dict[str, Optional[str]]:
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

    Creates the directory and its parents if they do not exist.

    Returns:
        Path: The path object representing the templates directory.
    """
    templates_dir = Path.home() / ".config" / "lmt" / "templates"
    templates_dir.mkdir(parents=True, exist_ok=True)
    return templates_dir


def get_default_template_file_path() -> Path:
    """
    Returns the path to the default template file.
    """
    default_template_dir = Path.home() / ".config" / "lmt" / "default"
    default_template_dir.mkdir(parents=True, exist_ok=True)

    default_template_file = default_template_dir / "default.yaml"
    if not default_template_file.exists():
        default_template = read_text("lmt_cli.default_template", "default.yaml")
        with open(default_template_file, "w", encoding="UTF-8") as file:
            file.write(default_template)

    return default_template_file
