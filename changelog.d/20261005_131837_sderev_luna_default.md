Changed
-------
* Use `gpt-6-luna` with reasoning effort `none` when model and effort are omitted
  in CLI and generation helpers. Explicit models retain their provider effort
  default; explicit library `reasoning_effort=None` still omits the field.
* Preserve explicit model choices over template models, including an explicit
  selection of the default model.
