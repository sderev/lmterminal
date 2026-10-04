"""Local text-message estimates, independent of provider transport.

Supported messages contain string role/content and optional string name. Framing
is a heuristic: three tokens per message, one extra per name, three reply-priming
tokens. This is not a model-wide guarantee about provider usage.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from importlib.metadata import version

import tiktoken

from .model_registry import get_input_price_per_million, get_price_band, get_tokenizer_model
from .request_options import PreparedRequest

# These controls do not contribute input text. Anything unclassified is omitted
# explicitly, rather than treating arbitrary JSON serialization as token usage.
_NON_INPUT_CONTROLS = {
    "temperature",
    "reasoning_effort",
    "verbosity",
    "max_tokens",
    "max_completion_tokens",
    "frequency_penalty",
    "presence_penalty",
    "top_p",
    "logprobs",
    "top_logprobs",
    "logit_bias",
    "seed",
    "stream_options",
    "parallel_tool_calls",
    "store",
    "metadata",
    "user",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
    "service_tier",
}
_TRANSPORT_CONTROLS = {"extra_headers", "extra_query", "timeout"}


@dataclass(frozen=True)
class InputEstimate:
    model: str
    encoding: str | None = None
    tokenizer_version: str = version("tiktoken")
    method: str = "chat_text_heuristic"
    message_tokens: int | None = None
    request_complete: bool = False
    input_tokens: int | None = None
    input_rate_usd_per_million: Decimal | None = None
    input_cost_usd: Decimal | None = None
    pricing_context: str | None = None
    warnings: tuple[str, ...] = ()


def _option_omissions(controls, prefix=""):
    reasons = []
    for key, value in controls.items():
        path = f"{prefix}{key}"
        if key == "extra_body" and not prefix:
            if value is not None:
                reasons.extend(_option_omissions(value, "extra_body."))
        elif key not in _NON_INPUT_CONTROLS and not (not prefix and key in _TRANSPORT_CONTROLS):
            reasons.append(f"Option `{path}` is not counted locally.")
    return reasons


def _message_problem(messages):
    if not isinstance(messages, (list, tuple)):
        return "Messages must be a list or tuple of text-message mappings for estimation."
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            return f"Message {index} is not a text-message mapping."
        if set(message) - {"role", "content", "name"}:
            return f"Message {index} has unsupported fields; only role/content/name are counted."
        if not all(isinstance(message.get(key), str) for key in ("role", "content")):
            return (
                f"Message {index} requires string role/content; content parts/null are unsupported."
            )
        if message["role"] not in {"system", "developer", "user", "assistant"}:
            return f"Message {index} has an unsupported role for text estimation."
        if "name" in message and not isinstance(message["name"], str):
            return f"Message {index} requires a string name for estimation."
    return None


def _count_messages(messages, encoding):
    return 3 + sum(
        3
        + sum(len(encoding.encode_ordinary(value)) for value in message.values())
        + (1 if "name" in message else 0)
        for message in messages
    )


def input_price(model: str, tokens: int) -> tuple[Decimal, Decimal, str | None]:
    """Return Standard uncached USD rate, cost and tier for an estimated count."""
    rate = Decimal(str(get_input_price_per_million(model, tokens)))
    _, tier = get_price_band(model, tokens)
    return rate, Decimal(tokens) * rate / Decimal(1_000_000), tier


def estimate_request(request: PreparedRequest) -> InputEstimate:
    """Estimate a finalized request; never read credentials or invoke a provider.

    Tiktoken may provision public tokenizer data on first use. Loading failures
    are reported without substituting an unrelated encoder. A message subtotal
    is retained only when every message has the supported text shape.
    """
    omissions = _option_omissions(request.controls)
    result = InputEstimate(model=request.model, warnings=tuple(omissions))
    problem = _message_problem(request.messages)
    if problem:
        return replace(result, warnings=(*result.warnings, problem))
    try:
        encoding_name = tiktoken.encoding_name_for_model(get_tokenizer_model(request.model))
    except KeyError:
        return replace(result, warnings=(*result.warnings, "No known tokenizer for this model."))
    result = replace(result, encoding=encoding_name)
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:  # noqa: BLE001 - tokenizer loaders use varied filesystem/network errors.
        return replace(
            result,
            warnings=(
                *result.warnings,
                (
                    f"Tokenizer data for {encoding_name} is unavailable. Allow a first-use public-data "
                    "download or provide a populated TIKTOKEN_CACHE_DIR, then retry; no prompt is sent."
                ),
            ),
        )
    tokens = _count_messages(request.messages, encoding)
    result = replace(result, message_tokens=tokens, request_complete=not omissions)
    if omissions:
        return result
    result = replace(result, input_tokens=tokens)
    body = request.controls.get("extra_body") or {}
    tier = body.get("service_tier", request.controls.get("service_tier"))
    if tier not in (None, "default"):
        return replace(
            result,
            warnings=(
                "Requested service_tier is not Standard/default; its input cost is unpriced locally.",
            ),
        )
    try:
        rate, cost, context = input_price(request.model, tokens)
    except KeyError:
        return replace(result, warnings=("No known Standard input price for this model.",))
    return replace(
        result, input_rate_usd_per_million=rate, input_cost_usd=cost, pricing_context=context
    )
