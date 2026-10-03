### Added

* Add optional `--reasoning-effort` and repeatable `-o/--option key=value` Chat Completions controls, including dotted keys for nested options.

### Fixed

* Retain usage-only stream events without attempting to render a missing choice.
* Keep content-free nonstream responses usable by returning empty text and preserving the raw payload.
* Reject generic overrides of request fields owned by the library and CLI.
* Omit the default temperature for models that do not support sampling, while leaving their reasoning defaults unchanged.
