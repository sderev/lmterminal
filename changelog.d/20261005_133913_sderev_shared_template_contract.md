Changed
-------
* Templates now use `prompt` for task instructions and `text` for content. Migrate
  the removed `user` field to the appropriate field. CLI and library share
  `load_template`/`Template` and `resolve_request`, which returns a `PreparedRequest`;
  the previous template helpers are removed, and `prepare_and_generate_response`
  takes a prepared request and keyword output flags.
* Explicit settings override template values; template settings override package
  defaults. Appends insert a blank line and preserve input whitespace. Additional
  request options merge by top-level key. Any explicit system argument with a
  template is rejected, including empty strings and library `None`.
  Omitted/null template temperature follows the model's sampling policy; stored
  numeric values are explicit controls, and invocation `temperature=None`
  overrides them and omits the field.
* Use `--text` for literal content without a pipe. Combining it with nonempty
  stdin is rejected; template-only tasks no longer require terminal input.

Fixed
-----
* Template add and rename refuse occupied destinations. Reads and listing no
  longer create storage directories; listing and completion include YAML files.
