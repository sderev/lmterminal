Fixed
-----
* Warn on stderr when `config.json` contains malformed UTF-8/JSON or a non-object
  root, then continue with defaults without changing the file. Share the loaded
  configuration between model selection and response themes. Library
  `load_config` raises `StorageError` for these invalid contents.
