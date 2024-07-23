import sys
from importlib.resources import read_text
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import yaml
from lmterminal import DEFAULT_MODEL


def prepare_prompt_from_template(
    *, template: str, system: str, user_prompt: str, model: str
) -> Tuple[str, str, str]:
    """
    Prepares a prompt by applying a template and updating system and user content.

    This allows to append additional system instructions and user content to the template.

    Args:
        template (str): The name of the template to use.
        system (str): The system instructions to incorporate into the template.
        user_prompt (str): The user's prompt to incorporate into the template.
        model (str): The model to use.

    Returns:
        Tuple[str, str, str]: Updated system content, updated user prompt, and model.
    """
    template_content = get_template_content(template)

    def update_content(key: str, content: Optional[str]) -> str:
        existing_value = template_content.get(key, "")
        if existing_value is None:
            existing_value = ""
        return existing_value.rstrip() + (content or "")

    updated_system = update_content("system", system)
    updated_user_prompt = update_content("user", user_prompt)

    # If a specific model name is given in the options (different from the global default model),
    # it will override the default model specified in the template.
    if model != DEFAULT_MODEL:
        updated_model = model
    else:
        updated_model = template_content.get("model", model) or DEFAULT_MODEL

    return updated_system, updated_user_prompt, updated_model


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
