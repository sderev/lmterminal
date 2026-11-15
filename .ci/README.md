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

## Offline and live tests

`gate` runs offline on Python 3.10–3.14 and always excludes the `integration`
marker. `GATE_PYTEST_ARGS` can narrow the suite but cannot select live tests;
live selectors in it or `PYTEST_ADDOPTS` are rejected. Ordinary tests always use
a temporary HOME, scrubbed credential environment, blocked sockets and blocked
provider clients, including when another test has live access.

Check the live boundary with fake clients and inspect the flags without making
provider requests:

```sh
uv run --offline --no-sync pytest tests/live_opt_in_test.py tests/live_test.py
uv run --offline --no-sync pytest --help
```

The single case in `tests/live_test.py` skips by default. A live run requires
both `--run-live` and `--live-model MODEL`; a missing flag or an unknown model is
a local usage error. Models and aliases resolve against the installed offline
catalog (`lmt models`), without checking account access. There is no default live
model or registry-wide parameterization. A marker alone grants no live access.

**Paid provider opt-in:** the following command uses your existing configuration
and stored key and can incur charges. Replace `MODEL` with one registered model
or alias you intend to test, and invoke this one case directly:

```sh
uv run --locked --group dev pytest tests/live_test.py::test_live_call --run-live --live-model MODEL
```

Only that case temporarily restores the caller's HOME/config environment and
provider access. Guards are restored before the next test. Do not use `gate` for
live testing; it repeats the offline suite across five Python versions.
