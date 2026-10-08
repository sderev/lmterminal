import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from .model_registry import get_request_model_spec, resolve_model_name

DEFAULT_MODEL = "gpt-6-luna"
# Distinguish omitted model/controls from explicit choices, including library None.
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
    """Return True/False for known sampling rules, None for unverified acceptance."""
    spec = get_request_model_spec(model)
    if spec is not None:
        if spec.sampling_policy == "unverified":
            return None
        if spec.sampling_policy == "none_only":
            effective_effort = (
                spec.default_reasoning_effort if reasoning_effort is None else reasoning_effort
            )
            return effective_effort == "none"
        return spec.sampling_policy == "always"
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
    sampling = sampling_supported(model, reasoning_effort)
    if sampling is not False:
        if temperature is UNSET:
            if sampling is True:
                controls["temperature"] = 1
        elif temperature is not None:
            controls["temperature"] = temperature
    else:
        if temperature is not UNSET and temperature is not None:
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


def snapshot_containers(value, active=None):
    """Copy acyclic mappings, lists and tuples; borrow opaque provider values.

    This intentionally does not deepcopy SDK objects, iterators or other opaque
    values. Mapping keys (normally strings) are borrowed too.
    """
    if not isinstance(value, (Mapping, list, tuple)):
        return value
    active = set() if active is None else active
    identity = id(value)
    if identity in active:
        raise ValueError("Request containers must be acyclic.")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            return {key: snapshot_containers(item, active) for key, item in value.items()}
        items = [snapshot_containers(item, active) for item in value]
        return tuple(items) if isinstance(value, tuple) else items
    finally:
        active.remove(identity)


@dataclass(frozen=True, init=False)
class PreparedRequest:
    """Owned container snapshot; public access returns independent container copies.

    Construct with prepare_request/resolve_request to validate controls. Opaque
    objects remain borrowed: callers must keep them stable through estimation/send.
    """

    model: str
    _messages: object = field(repr=False)
    _controls: dict = field(repr=False)

    def __init__(self, model, messages, controls):
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "_messages", snapshot_containers(messages))
        object.__setattr__(self, "_controls", snapshot_containers(controls))

    @property
    def messages(self):
        return snapshot_containers(self._messages)

    @property
    def controls(self):
        return snapshot_containers(self._controls)


def prepare_request(
    model=UNSET, messages=None, temperature=UNSET, reasoning_effort=UNSET, request_options=None
):
    """Finalize once before estimation or transport; unknown library models pass through."""
    implicit_model = model is UNSET
    if implicit_model:
        model = DEFAULT_MODEL
    if reasoning_effort is UNSET:
        reasoning_effort = "none" if implicit_model else None
    if not isinstance(model, str) or not model:
        raise ValueError("Model must be a nonempty string.")
    model = resolve_model_name(model) or model
    options = snapshot_containers(request_options)
    controls = prepare_request_controls(model, temperature, reasoning_effort, options)
    return PreparedRequest(model, messages, controls)
