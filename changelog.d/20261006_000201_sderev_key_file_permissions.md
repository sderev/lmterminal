Changed
-------
* Store provider keys in `~/.config/lmt/keys.json` as a JSON object, preserving
  other provider entries when changing the OpenAI key. Run `lmt key set` again
  after upgrading; legacy `key.env` and `API_keys.json` files are not read or
  migrated. Requests still use OpenAI only.

Security
--------
* Create `keys.json` with owner-only permissions (`0600`) and apply `0600` before
  overwriting a stored API key. Reading or leaving an existing key unchanged does
  not change its permissions.
