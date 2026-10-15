import os
import re
import select
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path

import pytest

pty = pytest.importorskip("pty")


@pytest.fixture(params=["bash", "zsh"])
def interactive_completion(request, tmp_path, no_provider_or_download):
    """Read the actual edit buffer after Tab in an isolated shell."""
    shell = shutil.which(request.param)
    if shell is None:
        pytest.skip(f"{request.param} is not installed")
    source = Path(__file__).resolve().parents[1]
    wrapper = tmp_path / "lmt"
    wrapper.write_text(
        f"#!/bin/sh\nexec {shlex.quote(sys.executable)} -B "
        f'{shlex.quote(str(source / "tests/manual_cli.py"))} "$@"\n'
    )
    wrapper.chmod(0o700)
    environment = {
        "HOME": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "ZDOTDIR": str(tmp_path),
        "PATH": f"{tmp_path}:/usr/bin:/bin",
        "TERM": "xterm",
        "LC_ALL": "C.UTF-8",
        "PS1": "LMT_TEST> ",
        "PROMPT": "LMT_TEST> ",
        "HISTFILE": "/dev/null",
        "PYTHONPATH": str(source),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    arguments = (
        [shell, "--noprofile", "--norc", "-i"] if request.param == "bash" else [shell, "-f", "-i"]
    )
    pid, descriptor = pty.fork()
    if pid == 0:
        os.execve(shell, arguments, environment)

    def read_until(marker):
        output = bytearray()
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if select.select([descriptor], [], [], 0.1)[0]:
                output.extend(os.read(descriptor, 65536))
                if marker.encode() in output:
                    return output.decode(errors="replace")
        pytest.fail(f"Shell timed out waiting for {marker!r}: {output!r}")

    def command(text):
        os.write(descriptor, (text + "\r").encode())
        return read_until("LMT_TEST> ")

    def complete(line):
        # The dump widget runs only after the real completion widget finishes.
        os.write(descriptor, (line + "\t\x18").encode())
        output = read_until("__LMT_END__")
        match = re.search(r"__LMT_BUFFER__(.*?)__LMT_END__", output, re.DOTALL)
        assert match, output
        os.write(descriptor, b"\x03")
        read_until("LMT_TEST> ")
        return match[1]

    try:
        read_until("LMT_TEST> ")
        if request.param == "bash":
            command('eval "$(_LMT_COMPLETE=bash_source lmt)"')
            command(
                "lmt_dump_buffer() { printf '\\n__LMT_BUFFER__%s__LMT_END__\\n' "
                '"$READLINE_LINE"; }; bind -x \'"\\C-x":lmt_dump_buffer\''
            )
        else:
            command('autoload -Uz compinit; compinit -D; eval "$(_LMT_COMPLETE=zsh_source lmt)"')
            command(
                'lmt_dump_buffer() { print -r -- "" "__LMT_BUFFER__${BUFFER}__LMT_END__"; '
                'zle redisplay; }; zle -N lmt_dump_buffer; bindkey "^X" lmt_dump_buffer'
            )
        yield complete
    finally:
        os.killpg(pid, signal.SIGKILL)
        os.close(descriptor)
        os.waitpid(pid, 0)


@pytest.mark.parametrize("command", ["lmt", "lmt prompt"])
@pytest.mark.parametrize(
    "partial, completed",
    [
        ("--template=trans", "--template=translate"),
        ("--reasoning-effort=no", "--reasoning-effort=none"),
    ],
)
def test_joined_option_tab_preserves_option(interactive_completion, command, partial, completed):
    assert interactive_completion(f"{command} {partial}") == f"{command} {completed} "
