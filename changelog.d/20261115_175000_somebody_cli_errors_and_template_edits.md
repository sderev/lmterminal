### Fixed
* Failed template management and existing-key `set` commands now report errors
  on stderr with status 1. Request errors restore terminal output before showing
  guidance, omit SDK body dumps, and exit quietly when an output pipe closes.
