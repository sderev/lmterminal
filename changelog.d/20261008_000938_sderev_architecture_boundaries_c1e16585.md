Changed
-------
* Library composition now uses `lmterminal.resolution.resolve_request` with explicit
  template values and storage paths. `send_prepared_request` takes a caller-owned
  client and optional `on_text` callback; transport stays silent and raises errors.
  The overlapping `lmterminal.lib` and `chatgpt_request` APIs are removed.
* Prepared requests snapshot nested mappings, lists and tuples; modifying inputs
  or returned containers no longer changes a prepared request. Opaque provider
  values remain caller-owned.

Fixed
-----
* Missing-key generation reports errors on stderr without creating an empty key file.
* Execution templates report YAML constructor failures as template errors.
