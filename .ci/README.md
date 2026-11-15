# Developer checks

Run these commands from the repository root after `uv sync --locked --group dev`.

## Distribution contents

```sh
uv run --offline --no-sync python .ci/check-distributions.py
```

The check copies tracked source files, adds synthetic hidden local data and a
visible package asset, then builds a wheel, an sdist and a wheel from that sdist.
It verifies that hidden package paths are absent and package files, assets and
license text remain byte-for-byte intact, including the Alabaster attribution
and MIT terms in `code_themes.py`. It never copies ignored personal settings.
The declared build backend must already be cached; all three builds use PEP 517
and disable network access through uv's `--offline` flag.
