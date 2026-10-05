# LMterminal (`lmt`)

Send a prompt to an OpenAI model from your terminal. Responses stream by default.

## Installation

```bash
pipx install LMterminal
lmt key set
```

From a source checkout:

```bash
uv sync --group dev
uv run lmt --help
```

The console script is `lmt`. Use `uv run lmt ...` from a checkout.

## Usage

```bash
lmt "Say hello"
lmt --system "Reply in French." "Say hello"
lmt models
lmt --model gpt-5.4 "Explain this function"
lmt --tokens --model gpt-5.4 "Estimate this prompt"
```

With no model or reasoning choice, LMT uses `gpt-6-luna` with reasoning effort
`none`. An explicit CLI or template model keeps that model's provider reasoning
default unless you pass `--reasoning-effort`, including when you select Luna.

`lmt models` lists the registered Chat Completions models and aliases. Other
endpoints, including Responses-only Pro and Codex models, and retired models are
not supported. Library requests also reject known Responses-only Pro snapshots.
Model access and availability depend on your OpenAI account.

GPT-6 Luna, GPT-6.1 Sol and GPT-6 Astra are available for text
generation through Chat Completions. Their version aliases omit the `gpt-` prefix.
The family aliases `sol`, `luna` and `astra` select the newest supported plain
release in the installed LMT catalog, currently `gpt-6.1-sol`, `gpt-6-luna` and
`gpt-6-astra`. Package updates can advance these aliases; use an explicit model
ID or version alias such as `6.1-sol` to choose a version. Resolution and listing
run offline, without checking account access or tokenizer availability. An
unavailable tokenizer or API error does not trigger a fallback to an older model.
GPT-6 Astra and GPT-6.1 Sol require Responses for tool calling; GPT-6 Luna
supports Chat Completions function calling only with reasoning effort `none`.

`--tokens` estimates text input tokens and **Standard uncached input** cost in USD,
without reading your API key or sending a provider request. Message framing is a
heuristic; provider usage may differ. Actual input may cost less with cached
tokens. Output/reasoning, tool fees and service-tier adjustments are excluded.
For GPT-5.4 and the models above, the long-context rate applies above 272,000
estimated input tokens; uncertainty near that boundary can change the applicable
rate. Estimates are not bills.
LMterminal requires `tiktoken` 0.14.0 or later and asks its resolver for the exact
canonical model ID. GPT-3.5/4/Turbo use `cl100k_base`; supported GPT-4o/4.1,
o-series and GPT-5 models use `o200k_base`. GPT-6 estimates remain unavailable
while upstream has no mapping; no other model's tokenizer is substituted.

Tools, functions, schemas and unclassified request options produce a message-only
subtotal with a named omission, without a whole-request token or cost estimate.
Unsupported message content (including content parts and tool-call messages), an
unknown tokenizer or unavailable tokenizer data produces an unavailable result.
Unknown prices and nonstandard `service_tier` values retain the token estimate but
leave cost unavailable. Unsupported messages/tokenizers exit with status 1;
partial estimates exit with status 0. Invalid request controls fail before either
estimation or generation.

First use may download public tokenizer data through `tiktoken`; no prompt is sent
in that download. A warmed cache works locally without a provider key. For a
separate cache location, set `TIKTOKEN_CACHE_DIR`. A cold installation is not
guaranteed to work offline.

Read a prompt from a file or append instructions to piped text:

```bash
cat example.py | lmt "Explain this code"
lmt < prompt.txt
lmt "Say hello" > response.txt
lmt --no-stream "Say hello"
```

In a terminal, run `lmt` without a prompt to enter multiple lines; press Ctrl+D to
send. `--raw` disables formatting. Streaming to pipes or terminals that cannot
render interactive Markdown produces plain text, including with `--rich`.
`--no-stream` waits for the complete response and prints plain text.
Content-free tool-call responses print an empty line; library callers retain the
raw response payload.

In Vim, filter selected lines with `:'<,'>!lmt "Rewrite this paragraph"`.

## Diagnosing response delays

Repeat `-v/--verbose` for timing diagnostics on stderr. For example, rerun the
delayed command from this checkout with its existing model/template/options and
add `-vvv`:

```bash
uv run lmt -vvv --model gpt-5.4 "Say hello" 2> timings.log
uv run lmt -vvv --raw "Say hello" > response.txt 2> timings.log
```

`-v` shows the finalized model, stream mode, API route, request dispatch, first
nonempty text, output and completion timings. `-vv` adds prompt/request setup,
client/output readiness, response/stream readiness, the first SDK stream event
(which can contain metadata or empty text), and output submission. `-vvv` adds
event/text-chunk counts and numeric usage totals when the provider already returns
them; it does not request usage or change generation settings. The response model
is shown when present on the response or first stream event.

The `+...s` timestamps use a monotonic clock starting at prompt handling; they
exclude process imports and CLI argument parsing, and include any wait for stdin.
The final `request_ttft_s` measures dispatch to first nonempty text. Compare it
with `first_flush_s` for plain output or `first_refresh_s` for Markdown. A large
wait before text reaches the SDK could involve network setup, SDK retries or the
provider; these timings do not distinguish those causes. A large gap after text
reception points to local output work. Markdown refresh completion means the
renderer returned, and does not guarantee text was visible on screen. Missing
text is reported as `unavailable`, including content-free tool responses.
`--no-stream` reports full-response reception rather than stream events.
`request_s` spans dispatch through response consumption, including output work
during streaming; `output complete` follows final rendering or output flushing.

Timing diagnostics never print prompts, response text, headers, keys, full URLs or
arbitrary request options, even at `-vvv`. The separate `--debug` option prints
prompts; use `-vvv` alone when sharing timings. Checkout changes do not update a
separately installed `lmt` or consumer such as LMtoolbox's `translate`.

## Request controls

```bash
lmt -m gpt-5.4 --reasoning-effort high -o verbosity=low "Explain this function"
lmt -m gpt-5.4 --reasoning-effort none --temperature 0.7 "Write a greeting"
lmt -m gpt-6-luna --reasoning-effort none "Translate this sentence into French"
lmt -m gpt-6.1-sol --reasoning-effort max "Explain this function"
lmt -o max_completion_tokens=500 -o stream_options.include_usage=true "Say hello"
```

`--reasoning-effort` overrides the effort for the selected model. Supported
values depend on the model. The default temperature of 1 is omitted for models
that do not support sampling, including GPT-5/nano/mini and o-series models;
other temperature values are rejected for these models. GPT-5.1, GPT-5.2 and
GPT-5.4 variants allow sampling with reasoning effort `none` (their default).

GPT-6 Luna supports `none`, `low`, `medium`, `high`, `xhigh`
and `max`; an explicitly selected Luna model with omitted effort preserves its
provider `medium` default. GPT-6.1 Sol and
GPT-6 Astra support `low`, `medium`, `high`, `xhigh` and `max`; neither accepts
`none` or `minimal`. GPT-6.1 Sol defaults to `medium`. These model/effort
combinations are checked locally for both CLI and library requests. For these
three models, temperature, `top_p`, `logprobs` and `top_logprobs` require an
effective reasoning effort of `none` on GPT-6 Luna. Implicit requests use `none`;
if you explicitly select Luna and omit effort, set `--reasoning-effort none` to
use these controls. The default temperature of 1 is otherwise omitted; explicit
incompatible controls are rejected.

Repeat `-o/--option key=value` for additional Chat Completions parameters. Values
use JSON when valid, otherwise text; dotted keys build nested objects. Owned
fields such as `model`, `messages`, `stream`, `n`, `stop`, `temperature` and
`reasoning_effort` cannot be overridden with `-o`. Other model-specific options
are checked by the API.

## Templates

```bash
lmt templates add explain
lmt templates edit explain
lmt templates list
lmt --template explain "Explain this code"
```

Templates are YAML files in `~/.config/lmt/templates/` with `system`, `user` and
`model` fields. They prepend instructions to your prompt. `--template` and
`--system` cannot be used together.

## Library input estimates

`lib.generate_response` and `gpt_integration.chatgpt_request` use the same
Luna/`none` default when both model and effort are omitted. Passing a model
preserves its provider effort default; passing `reasoning_effort=None` explicitly
omits that request field. Other explicit controls remain subject to model validation.

Use `lmterminal.request_options.prepare_request(model, messages, ...)` followed by
`lmterminal.estimation.estimate_request(request)`. The `InputEstimate` result
contains `message_tokens`, `input_tokens`, `request_complete`, numeric `Decimal`
`input_cost_usd`/`input_rate_usd_per_million`, and `warnings`; unavailable fields
are `None`. A complete estimate still uses heuristic framing: 3 tokens per
message, 1 extra per name, and 3 reply-priming tokens, plus ordinary-text encoding
of each string value. Supported message fields are `role`, `content`, and optional
`name`; supported roles are system, developer, user, and assistant. Tokenizer name,
version and method are included in the result.

This replaces the token/count/cost helpers previously in `gpt_integration`.
Transport helpers retain unknown-model pass-through; estimation never substitutes
an unrelated tokenizer or guesses an unknown price.

## Colors

`code_block_theme` accepts the bundled `alabaster` palette or an installed Pygments
style. Alabaster code blocks use a `#f0f0f0` background for contrast with Alabaster
terminals, preserving the palette's foreground colors.
Unknown styles fail before a provider request when Rich formatting is used;
choose an available style or use `--raw`. Plain output and `--tokens` do not validate
response themes. `inline_code_theme` accepts a Rich foreground/background style.
Only the top-level keys below configure these colors; nested `shellgenius` settings
belong to ShellGenius.

Theme reads use defaults without creating or changing configuration. To customize
colors, create or edit `~/.config/lmt/config.json`, preserving any existing fields:

```json
{
    "code_block_theme": "monokai",
    "inline_code_theme": "blue on black"
}
```

## Tests

Provision the two public vocabularies once, then run the encoder checks and gate:

```bash
export LMT_TEST_TIKTOKEN_CACHE="$PWD/.cache/tokenizer-tests"
TIKTOKEN_CACHE_DIR="$LMT_TEST_TIKTOKEN_CACHE" uv run --locked --group dev python -c 'import tiktoken; tiktoken.get_encoding("cl100k_base"); tiktoken.get_encoding("o200k_base")'
uv run --locked --group dev pytest tests/tokenizer_test.py tests/estimation_test.py
gate
```

Tests copy these assets into an isolated cache and block downloads, key access
and provider clients. Both vocabularies have fixed token-count oracles; every
registered Chat Completions model and alias has an independent resolver
expectation. CI provisions the assets for each Python version and fails if they
are absent or invalid. Local runs without `LMT_TEST_TIKTOKEN_CACHE` skip the
vocabulary checks. Live API tests remain opt-in with `--run-live`.

## License

Apache-2.0. See [LICENSE](LICENSE).
