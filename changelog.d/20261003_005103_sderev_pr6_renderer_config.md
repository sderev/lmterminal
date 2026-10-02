Fixed
-----
* Keep raw streaming fragments adjacent and flush plain output for pipes, dumb terminals, and terminals with `TTY_INTERACTIVE=0`; refresh formatted terminal output on each nonempty chunk.
