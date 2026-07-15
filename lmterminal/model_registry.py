from __future__ import annotations

from dataclasses import dataclass

SHORT_CONTEXT_TOKEN_THRESHOLD = 272_000
REASONING_EFFORTS = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


@dataclass(frozen=True)
class PriceBand:
    input: float | None = None
    cached_input: float | None = None
    output: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    aliases: tuple[str, ...] = ()
    short_context: PriceBand = PriceBand()
    long_context: PriceBand | None = None
    tokenizer_model: str | None = None
    show_in_cli: bool = True


def _spec(
    *,
    aliases: tuple[str, ...] = (),
    short_input: float | None = None,
    short_cached_input: float | None = None,
    short_output: float | None = None,
    long_input: float | None = None,
    long_cached_input: float | None = None,
    long_output: float | None = None,
    tokenizer_model: str | None = None,
    show_in_cli: bool = True,
) -> ModelSpec:
    long_context = None
    if any(value is not None for value in (long_input, long_cached_input, long_output)):
        long_context = PriceBand(long_input, long_cached_input, long_output)

    return ModelSpec(
        aliases=aliases,
        short_context=PriceBand(short_input, short_cached_input, short_output),
        long_context=long_context,
        tokenizer_model=tokenizer_model,
        show_in_cli=show_in_cli,
    )


MODEL_REGISTRY = {
    "gpt-3.5-turbo": _spec(
        aliases=("chatgpt", "3.5"),
        short_input=0.50,
    ),
    "gpt-3.5-turbo-0125": _spec(
        short_input=0.50,
        tokenizer_model="gpt-3.5-turbo",
        show_in_cli=False,
    ),
    "gpt-3.5-turbo-1106": _spec(
        short_input=0.50,
        tokenizer_model="gpt-3.5-turbo",
        show_in_cli=False,
    ),
    "gpt-3.5-turbo-instruct": _spec(short_input=1.50),
    "gpt-4": _spec(
        aliases=("4", "gpt4"),
        short_input=30,
    ),
    "gpt-4-turbo": _spec(
        aliases=("4t", "4-turbo", "gpt4-turbo"),
        short_input=10,
    ),
    "gpt-4-turbo-preview": _spec(
        short_input=10,
        tokenizer_model="gpt-4-turbo",
        show_in_cli=False,
    ),
    "gpt-4-turbo-2024-04-09": _spec(
        short_input=0.01,
        tokenizer_model="gpt-4-turbo",
        show_in_cli=False,
    ),
    "gpt-4-0613": _spec(
        short_input=0.03,
        tokenizer_model="gpt-4",
        show_in_cli=False,
    ),
    "gpt-4-32k": _spec(
        aliases=("4-32k", "gpt4-32k"),
        short_input=60,
    ),
    "gpt-4-1106-preview": _spec(
        short_input=10,
        tokenizer_model="gpt-4-turbo",
        show_in_cli=False,
    ),
    "gpt-4-0125-preview": _spec(
        short_input=10,
        tokenizer_model="gpt-4-turbo",
        show_in_cli=False,
    ),
    "gpt-4-32k-0613": _spec(
        short_input=60,
        tokenizer_model="gpt-4-32k",
        show_in_cli=False,
    ),
    "gpt-4o": _spec(
        aliases=("4o",),
        short_input=2.50,
        short_cached_input=1.25,
        short_output=10.00,
    ),
    "gpt-4o-2024-05-13": _spec(
        short_input=5.00,
        short_output=15.00,
        tokenizer_model="gpt-4o",
    ),
    "gpt-4o-2024-08-06": _spec(
        short_input=2.50,
        short_cached_input=1.25,
        short_output=10.00,
        tokenizer_model="gpt-4o",
    ),
    "gpt-4o-2024-11-20": _spec(
        short_input=2.50,
        short_cached_input=1.25,
        short_output=10.00,
        tokenizer_model="gpt-4o",
    ),
    "gpt-4o-mini": _spec(
        aliases=("4o-mini", "4omini", "4om"),
        short_input=0.15,
        short_cached_input=0.075,
        short_output=0.60,
    ),
    "gpt-4o-mini-2024-07-18": _spec(
        short_input=0.15,
        short_cached_input=0.075,
        short_output=0.60,
        tokenizer_model="gpt-4o-mini",
    ),
    "chatgpt-4o-latest": _spec(
        short_input=5.00,
        short_output=15.00,
        tokenizer_model="gpt-4o",
    ),
    "o1": _spec(
        short_input=15.00,
        short_cached_input=7.50,
        short_output=60.00,
    ),
    "o1-2024-12-17": _spec(
        short_input=15.00,
        short_cached_input=7.50,
        short_output=60.00,
        tokenizer_model="o1",
    ),
    "o1-preview": _spec(
        short_input=15.00,
        tokenizer_model="o1",
    ),
    "o1-preview-2024-09-12": _spec(
        short_input=15.00,
        tokenizer_model="o1",
    ),
    "o1-mini": _spec(
        short_input=1.10,
        short_cached_input=0.55,
        short_output=4.40,
    ),
    "o1-mini-2024-09-12": _spec(
        short_input=1.10,
        short_cached_input=0.55,
        short_output=4.40,
        tokenizer_model="o1-mini",
    ),
    "o1-pro": _spec(
        short_input=150.00,
        short_output=600.00,
    ),
    "o1-pro-2025-03-19": _spec(
        short_input=150.00,
        short_output=600.00,
        tokenizer_model="o1-pro",
    ),
    "gpt-4.1": _spec(
        aliases=("4.1",),
        short_input=2.00,
        short_cached_input=0.50,
        short_output=8.00,
    ),
    "gpt-4.1-2025-04-14": _spec(
        short_input=2.00,
        short_cached_input=0.50,
        short_output=8.00,
        tokenizer_model="gpt-4.1",
    ),
    "gpt-4.1-mini": _spec(
        aliases=("4.1-mini",),
        short_input=0.40,
        short_cached_input=0.10,
        short_output=1.60,
    ),
    "gpt-4.1-mini-2025-04-14": _spec(
        short_input=0.40,
        short_cached_input=0.10,
        short_output=1.60,
        tokenizer_model="gpt-4.1-mini",
    ),
    "gpt-4.1-nano": _spec(
        aliases=("4.1-nano",),
        short_input=0.10,
        short_cached_input=0.025,
        short_output=0.40,
    ),
    "gpt-4.1-nano-2025-04-14": _spec(
        short_input=0.10,
        short_cached_input=0.025,
        short_output=0.40,
        tokenizer_model="gpt-4.1-nano",
    ),
    "gpt-4.5-preview": _spec(short_input=75),
    "o3": _spec(
        short_input=2.00,
        short_cached_input=0.50,
        short_output=8.00,
    ),
    "o3-2025-04-16": _spec(
        short_input=2.00,
        short_cached_input=0.50,
        short_output=8.00,
        tokenizer_model="o3",
    ),
    "o3-mini": _spec(
        short_input=1.10,
        short_cached_input=0.55,
        short_output=4.40,
    ),
    "o3-mini-2025-01-31": _spec(
        short_input=1.10,
        short_cached_input=0.55,
        short_output=4.40,
        tokenizer_model="o3-mini",
    ),
    "o3-pro": _spec(
        short_input=20.00,
        short_output=80.00,
    ),
    "o4-mini": _spec(
        short_input=1.10,
        short_cached_input=0.275,
        short_output=4.40,
    ),
    "o4-mini-2025-04-16": _spec(
        short_input=1.10,
        short_cached_input=0.275,
        short_output=4.40,
        tokenizer_model="o4-mini",
    ),
    "codex-mini-latest": _spec(
        short_input=1.50,
        short_cached_input=0.375,
        short_output=6.00,
    ),
    "gpt-4o-search-preview": _spec(short_input=2.50),
    "gpt-4o-search-preview-2025-03-11": _spec(
        short_input=2.50,
        tokenizer_model="gpt-4o",
    ),
    "gpt-4o-mini-search-preview": _spec(short_input=0.15),
    "gpt-4o-mini-search-preview-2025-03-11": _spec(
        short_input=0.15,
        tokenizer_model="gpt-4o-mini",
    ),
    "gpt-5": _spec(
        aliases=("5", "gpt5"),
        short_input=1.25,
        short_cached_input=0.125,
        short_output=10.00,
    ),
    "gpt-5-mini": _spec(
        aliases=("5-mini",),
        short_input=0.25,
        short_cached_input=0.025,
        short_output=2.00,
    ),
    "gpt-5-nano": _spec(
        aliases=("5-nano",),
        short_input=0.05,
        short_cached_input=0.005,
        short_output=0.40,
    ),
    "gpt-5-chat-latest": _spec(
        short_input=1.25,
        short_cached_input=0.125,
        short_output=10.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5-codex": _spec(
        short_input=1.25,
        short_cached_input=0.125,
        short_output=10.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5-pro": _spec(
        aliases=("5-pro",),
        short_input=15.00,
        short_output=120.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.1": _spec(
        aliases=("5.1",),
        short_input=1.25,
        short_cached_input=0.125,
        short_output=10.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.1-chat-latest": _spec(
        short_input=1.25,
        short_cached_input=0.125,
        short_output=10.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.1-codex": _spec(
        short_input=1.25,
        short_cached_input=0.125,
        short_output=10.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.1-codex-max": _spec(
        short_input=1.25,
        short_cached_input=0.125,
        short_output=10.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.1-codex-mini": _spec(
        short_input=0.25,
        short_cached_input=0.025,
        short_output=2.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.2": _spec(
        aliases=("5.2",),
        short_input=1.75,
        short_cached_input=0.175,
        short_output=14.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.2-chat-latest": _spec(
        short_input=1.75,
        short_cached_input=0.175,
        short_output=14.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.2-codex": _spec(
        short_input=1.75,
        short_cached_input=0.175,
        short_output=14.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.2-pro": _spec(
        aliases=("5.2-pro",),
        short_input=21.00,
        short_output=168.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.3-chat-latest": _spec(
        short_input=1.75,
        short_cached_input=0.175,
        short_output=14.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.3-codex": _spec(
        short_input=1.75,
        short_cached_input=0.175,
        short_output=14.00,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.4": _spec(
        aliases=("5.4",),
        short_input=2.50,
        short_cached_input=0.25,
        short_output=15.00,
        long_input=5.00,
        long_cached_input=0.50,
        long_output=22.50,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.4-mini": _spec(
        aliases=("5.4-mini",),
        short_input=0.75,
        short_cached_input=0.075,
        short_output=4.50,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.4-nano": _spec(
        aliases=("5.4-nano",),
        short_input=0.20,
        short_cached_input=0.02,
        short_output=1.25,
        tokenizer_model="gpt-5",
    ),
    "gpt-5.4-pro": _spec(
        aliases=("5.4-pro",),
        short_input=30.00,
        short_output=180.00,
        long_input=60.00,
        long_output=270.00,
        tokenizer_model="gpt-5",
    ),
}


def get_valid_models() -> dict[str, tuple[str, ...] | None]:
    return {
        model_name: spec.aliases or None
        for model_name, spec in MODEL_REGISTRY.items()
        if spec.show_in_cli
    }


def resolve_model_name(model_name: str) -> str | None:
    normalized_name = model_name.lower()

    for canonical_model_name, spec in MODEL_REGISTRY.items():
        if not spec.show_in_cli:
            continue
        if normalized_name == canonical_model_name:
            return canonical_model_name
        if normalized_name in spec.aliases:
            return canonical_model_name

    return None


def get_model_spec(model_name: str) -> ModelSpec:
    return MODEL_REGISTRY[model_name]


def get_tokenizer_model(model_name: str) -> str:
    spec = MODEL_REGISTRY.get(model_name)
    if spec is None:
        return model_name
    return spec.tokenizer_model or model_name


def get_price_band(model_name: str, prompt_tokens: int) -> tuple[PriceBand, str | None]:
    spec = get_model_spec(model_name)
    if spec.long_context and prompt_tokens >= SHORT_CONTEXT_TOKEN_THRESHOLD:
        return spec.long_context, "long"
    return spec.short_context, "short" if spec.long_context else None


def get_input_price_per_million(model_name: str, prompt_tokens: int) -> float:
    price_band, _ = get_price_band(model_name, prompt_tokens)
    if price_band.input is None:
        raise KeyError(f"No input price configured for model: {model_name}")
    return price_band.input
