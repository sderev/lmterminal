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

`lmt models` lists the registered Chat Completions models and aliases. Other
endpoints, including Responses-only Pro and Codex models, and retired models are
not supported. Library requests also reject known Responses-only Pro snapshots.
Model access and availability depend on your OpenAI account.

`--tokens` estimates input tokens and cost without reading your API key or sending
a request. Estimates include message overhead but exclude output, caching,
service-tier adjustments and tool charges. For GPT-5.4, the long-context input
rate applies above 272,000 estimated tokens. Estimates are not bills.

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

## Request controls

```bash
lmt -m gpt-5.4 --reasoning-effort high -o verbosity=low "Explain this function"
lmt -m gpt-5.4 --reasoning-effort none --temperature 0.7 "Write a greeting"
lmt -o max_completion_tokens=500 -o stream_options.include_usage=true "Say hello"
```

`--reasoning-effort` leaves the model default unchanged when omitted. Supported
values depend on the model. The default temperature of 1 is omitted for models
that do not support sampling, including GPT-5/nano/mini and o-series models;
other temperature values are rejected for these models. GPT-5.1, GPT-5.2 and
GPT-5.4 variants allow sampling with reasoning effort `none` (their default).

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

## Colors

Theme reads use defaults without creating or changing configuration. To customize
colors, create or edit `~/.config/lmt/config.json`, preserving any existing fields:

```json
{
    "code_block_theme": "monokai",
    "inline_code_theme": "blue on black"
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
