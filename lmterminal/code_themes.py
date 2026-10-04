"""Code-block theme resolution for Rich Markdown.

AlabasterStyle is adapted from shellgenius/theme.py (Apache-2.0):
https://github.com/sderev/shellgenius/blob/89fb50fb9ac891e121fb2832dccbfb67b23da79b/shellgenius/theme.py
Changes: copy the standalone style, resolve it locally without a plugin, and
use a #f0f0f0 background for contrast with Alabaster terminals.
ShellGenius's Apache-2.0 terms are included in the repository LICENSE.

The palette derives from sderev/alabaster.vim (MIT):
https://github.com/sderev/alabaster.vim/tree/3f22e44b3a0cc971cbb63c1d38643e8d9a5bcdf9

MIT License

Copyright (c) 2025 Sébastien De Revière

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import ClassVar

from pygments.style import Style
from pygments.styles import get_style_by_name
from pygments.token import (
    Comment,
    Error,
    Generic,
    Keyword,
    Name,
    Number,
    Operator,
    Punctuation,
    String,
    Token,
)
from pygments.util import ClassNotFound
from rich.syntax import RICH_SYNTAX_THEMES, PygmentsSyntaxTheme, Syntax, SyntaxTheme


class AlabasterStyle(Style):
    """Light Pygments style derived from sderev/alabaster.vim.

    Color palette: https://github.com/sderev/alabaster.vim
    Background adapted to #f0f0f0 for contrast with Alabaster terminals.
    """

    name = "alabaster"
    background_color = "#f0f0f0"

    styles: ClassVar[dict] = {
        Token: "#000000",
        Comment: "#aa3731",
        Comment.Preproc: "#aa3731",
        String: "#448C27",
        Number: "#7a3e9d",
        Keyword: "#7a3e9d",
        Keyword.Type: "#000000",
        Name.Function: "#325cc0",
        Name.Class: "#325cc0",
        Name.Decorator: "#325cc0",
        Name.Tag: "#007acc",
        Name.Attribute: "#325cc0",
        Name.Builtin: "#000000",
        Operator: "#000000",
        Punctuation: "#777777",
        Generic.Heading: "#325cc0",
        Generic.Subheading: "#325cc0",
        Generic.Deleted: "#aa3731",
        Generic.Inserted: "#448C27",
        Generic.Error: "#aa3731",
        Generic.Emph: "italic",
        Generic.Strong: "bold",
        Error: "#aa3731",
    }


def resolve_code_theme(name: str) -> SyntaxTheme:
    """Resolve a configured name without Rich's silent unknown-style fallback."""
    if name == "alabaster":
        return PygmentsSyntaxTheme(AlabasterStyle)
    if name in RICH_SYNTAX_THEMES:
        return Syntax.get_theme(name)
    try:
        style = get_style_by_name(name)
    except ClassNotFound as error:
        raise ValueError(
            f"Code-block theme {name!r} is unavailable. Set `code_block_theme` in "
            "~/.config/lmt/config.json to `alabaster` or an installed Pygments style, "
            "or use --raw for plain output."
        ) from error
    return PygmentsSyntaxTheme(style)
