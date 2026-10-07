### Fixed

* End successful streamed responses redirected to files or pipes with a newline,
  including `--raw` and content-free responses. Preserve whitespace and existing
  trailing newlines, matching `--no-stream` output.
