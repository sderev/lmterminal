## Changed

* `lmt templates view` shows saved fields as themed YAML in terminals and plain
  YAML when redirected, without comments, source formatting or added defaults.
  Parsed values and key order are preserved; any YAML mapping can be inspected
  without applying execution schema validation.
* Missing or unreadable templates, malformed YAML and non-mapping roots (including
  empty files) now fail with an error instead of printing raw contents or silently
  succeeding.
