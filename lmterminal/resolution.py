"""Compose text and explicit template values without application resources."""

from .model_registry import composed_messages, resolve_model_name
from .request_options import (
    UNSET,
    PreparedRequest,
    prepare_request,
    snapshot_containers,
    validate_request_options,
)
from .templates import Template


class RequestResolutionError(ValueError):
    """Invocation inputs or effective request settings cannot be resolved."""


class SystemTemplateConflict(RequestResolutionError):
    """A supplied system argument cannot accompany a template."""


def _join_inputs(stored, supplied):
    return stored + "\n\n" + supplied if stored and supplied else stored or supplied


def resolve_request(
    *,
    template: Template | None = None,
    system=UNSET,
    prompt: str = "",
    text: str = "",
    model=UNSET,
    temperature=UNSET,
    reasoning_effort=UNSET,
    request_options=None,
    emoji=False,
) -> PreparedRequest:
    """Resolve inputs offline into a PreparedRequest shared by CLI and library.

    Explicit scalars override template settings, then package defaults. Explicit
    None omits a control. Options merge by top-level key. System argument presence
    conflicts with any template, even when the argument is empty or None.
    """
    if template is not None and system is not UNSET:
        raise SystemTemplateConflict("You cannot use both `--template` and `--system`.")
    if template is not None and not isinstance(template, Template):
        raise RequestResolutionError("Template must be a Template or None.")
    for name, value in (("prompt", prompt), ("text", text)):
        if not isinstance(value, str):
            raise RequestResolutionError(f"Argument `{name}` must be text.")
    if system is not UNSET and system is not None and not isinstance(system, str):
        raise RequestResolutionError("Argument `system` must be text or None.")
    stored = template if template is not None else Template()
    system = stored.system if system is UNSET else system or ""
    task = _join_inputs(stored.prompt, prompt)
    content = _join_inputs(stored.text, text)
    user = content + "\n___\n" + task if content and task else content or task
    if model is UNSET and stored.model is not None:
        model = stored.model
    if model is not UNSET:
        canonical = resolve_model_name(model) if isinstance(model, str) else None
        if canonical is None:
            raise RequestResolutionError("Argument/field `model` must be a registered model name.")
        model = canonical
    if temperature is UNSET and stored.temperature is not None:
        temperature = stored.temperature
    if (
        temperature is not UNSET
        and temperature is not None
        and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not 0 <= temperature <= 2
        )
    ):
        raise RequestResolutionError("Argument `temperature` must be between 0 and 2 or None.")
    if reasoning_effort is UNSET and stored.reasoning_effort is not None:
        reasoning_effort = stored.reasoning_effort
    try:
        supplied_options = snapshot_containers(request_options)
        validate_request_options(supplied_options)
        options = {**stored.request_options, **(supplied_options or {})}
        messages = composed_messages(model, add_emoji(system) if emoji else system, user)
        return prepare_request(model, messages, temperature, reasoning_effort, options)
    except (TypeError, ValueError) as error:
        raise RequestResolutionError(str(error)) from error


def add_emoji(system: str) -> str:
    """
    Adds an emoji to the system message.
    """
    emoji_message = (
        "Add plenty of emojis as a colorful way to convey emotions. However, don't mention it."
    )
    system = system.rstrip()

    if system == "":
        return emoji_message

    if not system.endswith("."):
        system += "."
    return system + " " + emoji_message
