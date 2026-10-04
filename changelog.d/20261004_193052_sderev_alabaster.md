Fixed
-----
* Honor `code_block_theme: alabaster` with the bundled ShellGenius palette, preserving inline-code colors. Formatted responses now reject unavailable code-block themes before provider work instead of silently using Pygments' default style; plain output and `--tokens` remain independent of response themes.
