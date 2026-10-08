"""Serialize saved template fields without applying execution defaults or validation."""

from collections.abc import Mapping
from pathlib import Path

import yaml

from .templates import TemplateError, template_path


class _ViewDumper(yaml.SafeDumper):
    pass


def _represent_text(dumper, value):
    if "\x85" in value:
        # YAML normalizes raw next-line characters, even in literal blocks.
        style = '"'
    else:
        style = "|" if "\n" in value else None
    # PyYAML selects quoting/chomping that preserves whitespace and final newlines.
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_ViewDumper.add_representer(str, _represent_text)


def serialize_saved_template(name: str, directory: Path) -> str:
    """Read a YAML mapping for inspection, retaining stored values and key order."""
    path = template_path(name, directory)
    try:
        content = yaml.safe_load(path.read_text(encoding="UTF-8"))
    except (OSError, UnicodeError) as error:
        raise TemplateError(f"Cannot read template `{name}`.") from error
    except (yaml.YAMLError, ValueError):
        raise TemplateError(f"Template `{name}` contains invalid YAML.") from None
    if not isinstance(content, Mapping):
        raise TemplateError(f"Template `{name}` must contain a mapping.")
    return yaml.dump(
        content,
        Dumper=_ViewDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=float("inf"),
    )
