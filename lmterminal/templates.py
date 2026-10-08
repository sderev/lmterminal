from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

import yaml

from .request_options import snapshot_containers, validate_request_options


class TemplateError(ValueError):
    """A template name, file or field cannot be used."""


@dataclass(frozen=True)
class Template:
    """Task instructions, content and optional request settings; null settings are omitted."""

    system: str = ""
    prompt: str = ""
    text: str = ""
    model: str | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    request_options: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ("system", "prompt", "text"):
            value = getattr(self, name)
            if value is None:
                object.__setattr__(self, name, "")
            elif not isinstance(value, str):
                raise TemplateError(f"Field `{name}` must be text or null.")
        for name in ("model", "reasoning_effort"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise TemplateError(f"Field `{name}` must be nonempty text or null.")
        value = self.temperature
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or not 0 <= value <= 2
        ):
            raise TemplateError("Field `temperature` must be a number between 0 and 2 or null.")
        if not isinstance(self.request_options, Mapping):
            raise TemplateError("Field `request_options` must be a mapping.")
        try:
            options = snapshot_containers(self.request_options)
            validate_request_options(options)
        except (TypeError, ValueError) as error:
            raise TemplateError(f"Field `request_options`: {error}") from error
        object.__setattr__(self, "request_options", options)


def template_path(name: str, directory: Path) -> Path:
    """Resolve an extensionless basename without creating directories."""
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).suffix
    ):
        raise TemplateError("Template names must be extensionless basenames.")
    return Path(directory) / f"{name}.yaml"


def list_templates(directory: Path) -> list[str]:
    """List only YAML files; a missing directory is an empty collection."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml") if path.is_file())


def load_template(name: str, directory: Path) -> Template:
    """Load and validate YAML without printing errors or disclosing file contents."""
    path = template_path(name, directory)
    try:
        content = yaml.safe_load(path.read_text(encoding="UTF-8"))
    except (OSError, UnicodeError) as error:
        raise TemplateError(f"Cannot read template `{name}`.") from error
    except (yaml.YAMLError, ValueError):
        raise TemplateError(f"Template `{name}` contains invalid YAML.") from None
    if content is None:
        content = {}
    if not isinstance(content, Mapping):
        raise TemplateError(f"Template `{name}` must contain a mapping.")
    if "user" in content:
        raise TemplateError(
            f"Template `{name}`: replace `user` with `prompt` for instructions or `text` for content."
        )
    if any(key not in Template.__dataclass_fields__ for key in content):
        raise TemplateError(f"Template `{name}` contains an unsupported field.")
    try:
        return Template(**content)
    except TemplateError as error:
        raise TemplateError(f"Template `{name}`: {error}") from error


DEFAULT_TEMPLATE_CONTENT = """# Task instructions and content are separate; no variable substitution.
system:
prompt:
text:
# Null settings inherit the package defaults.
model:
temperature:
reasoning_effort:
request_options: {}
"""
