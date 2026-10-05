import re
from collections.abc import Mapping
from dataclasses import dataclass

from .model_registry import get_request_model_spec, resolve_model_name

DEFAULT_MODEL = "gpt-6-luna"
# Distinguish omission from an explicit model or library reasoning_effort=None.
UNSET = object()

RESERVED_REQUEST_OPTION_KEYS = {
    "messages": None,
    "model": "--model",
    "n": None,
    "stop": None,
    "reasoning_effort": "--reasoning-effort",
    "stream": "--no-stream",
    "temperature": "--temperature",
}


def validate_request_options(options):
    """Protect fields owned by the helpers, including SDK body overrides."""
    if options is None:
        return
    if not isinstance(options, Mapping):
        raise TypeError("Request options must be a mapping.")
    for key in options:
        if key in RESERVED_REQUEST_OPTION_KEYS:
            flag = RESERVED_REQUEST_OPTION_KEYS[key]
            hint = f" Use {flag} instead." if flag else ""
            raise ValueError(f"Option `{key}` is reserved.{hint}")
    if "extra_body" in options:
        validate_request_options(options["extra_body"])


def sampling_supported(model, reasoning_effort):
    """Apply the documented sampling restrictions for registered families."""
    spec = get_request_model_spec(model)
    if spec is not None and spec.reasoning_efforts is not None:
        effective_effort = (
            spec.default_reasoning_effort if reasoning_effort is None else reasoning_effort
        )
        return effective_effort == "none" and "none" in spec.reasoning_efforts
    family = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    if family in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        return False
    if family in {"gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.4-mini", "gpt-5.4-nano"}:
        # These models default to none. Do not send a reasoning override.
        return reasoning_effort in (None, "none")
    return not (family.startswith(("o1", "o3", "o4")) or "search-preview" in family)


def prepare_request_controls(model, temperature, reasoning_effort, request_options):
    """Return optional controls without changing the provider's reasoning default."""
    validate_request_options(request_options)
    spec = get_request_model_spec(model)
    if spec is not None and not spec.chat_completions:
        raise ValueError(f"Model `{model}` is not supported for Chat Completions generation.")
    if (
        reasoning_effort is not None
        and spec is not None
        and spec.reasoning_efforts is not None
        and reasoning_effort not in spec.reasoning_efforts
    ):
        supported = ", ".join(spec.reasoning_efforts)
        raise ValueError(
            f"Reasoning effort `{reasoning_effort}` is not supported for `{model}`. "
            f"Use --reasoning-effort with one of: {supported}, or omit it for the model default."
        )
    controls = dict(request_options or {})
    if reasoning_effort is not None:
        controls["reasoning_effort"] = reasoning_effort
    if sampling_supported(model, reasoning_effort):
        if temperature is not None:
            controls["temperature"] = temperature
    else:
        # Keep the public default of 1 while omitting an unsupported field.
        if temperature not in (None, 1):
            raise ValueError(
                f"Temperature is not supported for `{model}` with this reasoning effort."
            )
        body = controls.get("extra_body") or {}
        for key in ("top_p", "logprobs", "top_logprobs"):
            if key in controls or key in body:
                raise ValueError(
                    f"Option `{key}` is not supported for `{model}` with this reasoning effort."
                )
    return controls


@dataclass(frozen=True)
class PreparedRequest:
    """Effective messages, canonical model and validated optional controls."""

    model: str
    messages: object
    controls: dict


def prepare_request(model, messages, temperature=1, reasoning_effort=UNSET, request_options=None):
    """Finalize once before estimation or transport; unknown library models pass through."""
    implicit_model = model is UNSET
    if implicit_model:
        model = DEFAULT_MODEL
    if reasoning_effort is UNSET:
        reasoning_effort = "none" if implicit_model else None
    if not isinstance(model, str) or not model:
        raise ValueError("Model must be a nonempty string.")
    model = resolve_model_name(model) or model
    controls = prepare_request_controls(model, temperature, reasoning_effort, request_options)
    return PreparedRequest(model, messages, controls)
