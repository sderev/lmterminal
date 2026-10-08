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

API keys are stored in `~/.config/lmt/keys.json` as a provider-to-key mapping:

```json
{"openai": "your-api-key"}
```

`lmt key set` adds your OpenAI key; `lmt key edit` changes it. Both prompts hide
your input. Other provider entries are preserved when changing OpenAI, but only
OpenAI requests are supported. The file must contain a JSON object with string
values; a missing or empty OpenAI value means no OpenAI key is set.

Reading a missing key creates no file. Missing-key errors go to stderr, leaving
redirected response files empty. Setting or changing a key creates the file with
owner-only permissions (`0600`) and applies `0600` before replacing stored values.
Reading an existing key or leaving it unchanged does not change its permissions.

This storage change requires reconfiguration: run `lmt key set` again after
upgrading. Legacy `key.env` and `API_keys.json` files are neither read nor migrated.

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
The former `chatgpt` alias for GPT-3.5 has been removed. Use `gpt-3.5-turbo`
or `3.5` to select that model.
Model access and availability depend on your OpenAI account. The catalog includes
GPT-5.5, GPT-5.6 Sol/Terra/Luna, GPT-6 Sol and documented regular-text GPT-5
snapshots. Use the exact dated ID to select a snapshot. Specialized Search API
models are outside this catalog.

GPT-5.5, GPT-5.6 Sol/Terra/Luna, GPT-6 Sol/Luna, GPT-6.1 Sol and GPT-6 Astra
are available for text generation through Chat Completions. Version aliases omit
the `gpt-` prefix:
`5.5`, `5.6-sol`, `5.6-terra`, `5.6-luna` and `6-sol` select those exact releases.
OpenAI's `gpt-5.6` alias and LMT's `5.6` shorthand resolve to `gpt-5.6-sol`.
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
heuristic; provider usage may differ. Cache reads and writes are excluded;
actual input charges may be lower or higher. Output/reasoning, tool fees and
service-tier adjustments are excluded. For GPT-5.4 (base), GPT-5.5, GPT-5.6 and
GPT-6 models, the long-context rate applies above 272,000
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
Successful responses redirected to files or pipes end with a newline in either
stream mode, including with `--raw`. A missing final newline is added without
trimming whitespace or changing existing trailing newlines.
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

Run `lmt prompt --help` for prompt options (also accepted without `prompt`).
`--reasoning-effort none` requests no reasoning where supported. Omitting model
and effort uses Luna with `none`; an explicitly selected model keeps its provider
reasoning default when effort is omitted. Supported values depend on the model.
Published effort lists are validated locally. GPT-5.5 accepts `none`, `low`,
`medium`, `high`, `xhigh`; GPT-5.6 variants,
GPT-6 Sol and GPT-6 Luna also accept `max`. These models default to `medium`.
GPT-6.1 Sol and GPT-6 Astra accept `low`, `medium`, `high`, `xhigh`, `max`,
excluding `none` and `minimal`; GPT-6.1 Sol defaults to `medium`.

Temperature defaults to 1 where sampling is documented as allowed and is omitted
for restricted or unverified combinations. GPT-5/nano/mini and o-series models
reject explicit sampling controls. GPT-5.1/5.2/5.4 variants allow sampling with
reasoning `none` (their default). GPT-6 Sol and Luna require effective `none`;
implicit Luna requests use `none`, while explicitly selected Sol/Luna models
require `--reasoning-effort none` to enable sampling.
GPT-6.1 Sol and Astra reject sampling. These rules cover temperature, `top_p`,
`logprobs` and `top_logprobs`, including explicit temperature `1`, which previously
was silently omitted for incompatible combinations.

GPT-5.5/5.6 sampling acceptance is unverified: implicit temperature is omitted,
and explicit controls are forwarded unchanged for API validation, including with
`none`. Provider errors remain failures. `--tokens` estimates locally and does not
establish provider acceptance. Omitted library temperature follows this policy;
explicit `temperature=None` omits the field. Numeric values, including `1`, are
explicit controls. Explicit library `reasoning_effort=None` omits effort even
when the model is omitted.

Repeat `-o/--option key=value` for additional Chat Completions parameters. Values
use JSON when valid, otherwise text; dotted keys build nested objects. Owned
fields such as `model`, `messages`, `stream`, `n`, `stop`, `temperature` and
`reasoning_effort` cannot be overridden with `-o`. Other model-specific options
are checked by the API.

## Templates

Templates are YAML files in `~/.config/lmt/templates/`. For example, save
`translate.yaml` with:

```yaml
prompt: Translate into English.
```

```bash
printf 'Bonjour.' | lmt -t translate "Keep product names unchanged."
lmt -t translate --text 'Bonjour.' "Keep product names unchanged."
lmt templates add summarize
lmt templates edit translate
lmt templates list
lmt templates --help
```

`template` is an alias for `templates`, for example `lmt template view translate`.
Both spellings select template management. To send a prompt starting with the word
`template`, use the explicit prompt command: `lmt prompt template ...`.

`lmt templates add` opens the editor selected by `VISUAL` or `EDITOR`; an
unchanged draft is removed. With shell completion enabled, template names complete
for `-t` and `view`, `edit`, `delete`, and `rename` under either group spelling.

`lmt templates view translate` shows only saved fields, in file order, with
multiline text and terminal highlighting using `code_block_theme`. Redirected
output is plain YAML without wrapping or padding. Parsing removes comments and
source formatting, preserves values (including nulls), and adds no defaults.
Any parseable YAML mapping can be inspected, even if its fields would fail
execution validation. Missing or unreadable files, malformed YAML and non-mapping
roots (including empty files) fail without printing template contents.

Names are extensionless basenames. Fields: `system`, `prompt` (task instructions),
`text` (content), `model`, `temperature`, `reasoning_effort`, `request_options`.
Positional arguments append to template instructions; stdin or literal `--text`
appends to template content. Each append inserts a blank line and preserves
whitespace. Content precedes instructions with `\n___\n` between them. There is
no variable substitution. `--text` with nonempty stdin is an error; a template
with a task can run in a terminal without waiting for Ctrl+D.

Explicit settings override template settings, then package defaults. Template
null scalars are unspecified. Omitted temperature follows the sampling policy
above; a numeric template temperature, including `1`, is explicit and subject to
model validation. Library `temperature=None` overrides a stored value and omits
the field. An explicit model keeps its provider reasoning
default unless an effort is supplied by the template or invocation. Additional
options merge by top-level key; an explicit nested object replaces the stored
object. Any supplied `--system`, including an empty value, conflicts with
`--template`, even when that template has no system field.

Migrate old `user` fields to `prompt` for instructions or `text` for content;
`user` is rejected. Execution accepts empty YAML as an empty template; inspection
requires a saved mapping. Invalid YAML constructors produce content-free errors.

## Library requests

Compose a request without configuration discovery, keys, terminal output or
provider access. Load a named template from an explicit directory, or construct
an in-memory `Template`:

```python
from pathlib import Path

from lmterminal.resolution import resolve_request
from lmterminal.templates import load_template

request = resolve_request(
    template=load_template("translate", Path("my-templates")),
    text="Bonjour.",
    prompt="Keep product names unchanged.",
)
```

`load_template(name, directory)` raises `TemplateError` for name/file/schema
errors. `resolve_request` accepts a `Template` or `None`, requires a registered
model, and raises `RequestResolutionError` for invocation/control errors.
`SystemTemplateConflict` covers any explicit `system` with a template, including
`""` and `None`. Omitted controls inherit; explicit `temperature=None` or
`reasoning_effort=None` omits that control. `emoji=True` is applied during
composition, before estimation or sending. The legacy composed o1 policy omits
the system message when an o1 model is selected; raw-message preparation retains
caller-supplied roles. This preserves existing behavior, not a capability claim.

For caller-supplied Chat Completions messages, use
`prepare_request(messages=messages, model=...)` from `lmterminal.request_options`.
It shares control validation and alias resolution but passes unknown model names
to the provider; known unsupported endpoints still fail. Omitted model and effort
use Luna/none. An explicit model keeps its provider default; explicit effort
`None` omits that field.

`PreparedRequest` owns snapshots of acyclic mappings, lists and tuples. Public
`messages` and `controls` access returns container copies, so mutating the source
or a returned container cannot change the prepared request. Mappings become dicts;
list/tuple shape is retained. Cycles are rejected. Opaque values (including SDK
objects, iterators and mapping keys) remain borrowed and must be kept stable by
the caller. They are not deep-copied. Use the preparer/resolver to validate a
request; private fields are implementation details.

Send with a client owned by your application (the following performs a provider
request and requires your application's key):

```python
from openai import OpenAI
from lmterminal.gpt_integration import send_prepared_request

with OpenAI(api_key=your_api_key) as client:
    text, elapsed_seconds, raw_response = send_prepared_request(
        client, request, stream=True, on_text=lambda chunk: print(chunk, end="", flush=True)
    )
```

Transport returns unmodified text and the raw SDK response (or collected stream
events), raises errors, and neither prints nor exits. The optional `on_text`
callback receives each nonempty text chunk before the next streaming event;
without it transport stays silent. Nonstream responses return text without
calling `on_text`. Callers close/reuse clients themselves. `n` and `stop` remain
explicit transport keywords; the returned text follows the first choice.
The CLI owns formatting, error messages, successful final newlines and client
closure. Optional `diagnostics` is an explicit observer; CLI verbosity supplies
one that writes metadata to stderr.

Breaking API changes: `lmterminal.lib`, `generate_response`,
`prepare_and_generate_response`, `chatgpt_request` and `format_prompt` are removed.
Import composition from `lmterminal.resolution` and send prepared values through
`send_prepared_request(client, request, ...)`; replace `update_markdown_stream`
with `on_text`. Template path/list/load and saved-template serialization functions
now require an explicit directory. Storage helpers in `lmterminal.storage` take
explicit paths (`load_config`, `read_api_key`, `write_key`) and raise
`StorageError` or filesystem errors without printing. No compatibility wrappers
or automatic personal configuration are provided to library callers.

## Library input estimates

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
Raw-message preparation retains unknown-model pass-through; estimation never
substitutes an unrelated tokenizer or guesses an unknown price.

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

Offline tests isolate HOME/config before application imports and block network
access and real provider clients. Encoder tests copy these public assets into an
isolated cache and also forbid credential access. Both vocabularies have fixed
token-count oracles; every registered Chat Completions model and alias has an independent resolver
expectation. CI provisions the assets for each Python version and fails if they
are absent or invalid. Local runs without `LMT_TEST_TIKTOKEN_CACHE` skip the
vocabulary checks. Live API tests remain opt-in with `--run-live`.

For an offline CLI walkthrough, `tests/manual_cli.py` supplies a fake client,
a temporary HOME, `translate` and `sparse` templates, and Alabaster colors. Its
default response echoes the composed user message so precedence is inspectable:

```bash
printf 'Bonjour.' | uv run python tests/manual_cli.py -t translate 'Keep names.'
uv run python tests/manual_cli.py --raw 'Hello' > response.txt
uv run python tests/manual_cli.py --no-stream 'Hello'
uv run python tests/manual_cli.py --fixture partial-failure --raw 'Hello' > partial.txt
uv run python tests/manual_cli.py --fixture markdown 'Hello'
uv run python tests/manual_cli.py templates view sparse
uv run python tests/manual_cli.py --fixture missing-key --raw 'Hello'
```

Expect content, `___`, then template/task instructions in the first command;
`Hello` plus a final LF in `response.txt` and no-stream output; nonzero exit with
exactly `partial` (no LF) in `partial.txt` on failure; colored fenced and inline
code in an interactive terminal; only the two stored fields for `sparse`; and a
stderr-only missing-key error. These commands never call a provider. The automated
counterparts live in `templates_test.py`, `cli_test.py`, `cli_output_test.py`,
`template_view_test.py` and `keys_test.py`; visual inspection remains a manual step.

## License

Apache-2.0. See [LICENSE](LICENSE).
