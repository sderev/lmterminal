### Added

* Save a CLI model preference with top-level `default_model` in `config.json`.
  CLI and template models take precedence; selected invalid preferences fail
  locally. A saved model retains its provider reasoning default when effort is
  omitted, including Luna. The package-only Luna fallback still uses `none`.
