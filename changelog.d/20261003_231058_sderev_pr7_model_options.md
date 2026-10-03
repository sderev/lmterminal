### Added

* Add GPT-5.4 models to `lmt models`, with shared aliases, tokenizer metadata and input-price estimates.

### Fixed

* Reject retired generation models and models that require another endpoint, including known Responses-only Pro snapshots, and accept registered Chat Completions snapshots consistently.
* Display single model aliases once in `lmt models`.
* Include message overhead in `--tokens` and apply GPT-5.4 long-context input pricing only above 272,000 estimated tokens.
* Correct the per-million input-price units for the GPT-4 and GPT-4 Turbo snapshots.
