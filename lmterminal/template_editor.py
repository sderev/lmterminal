"""Stage CLI editor saves before publishing a validated template."""

import os
import shutil
import stat
import tempfile
from errno import EEXIST, ENOTEMPTY
from pathlib import Path

import click

from .templates import DEFAULT_TEMPLATE_CONTENT, TemplateError, load_template


def _discard_draft(candidate):
    candidate.unlink()
    try:
        candidate.parent.rmdir()
    except OSError as error:
        # Editors may leave backups; remove only our own candidate.
        if error.errno not in (EEXIST, ENOTEMPTY):
            raise


def edit_template_file(name: str, destination: Path, *, create: bool) -> bool:
    """Return whether a save was published; keep changed drafts on failure."""
    candidate = None
    seed = b""
    failure = "Cannot prepare the template editor."
    try:
        if create:
            if os.path.lexists(destination):
                raise click.UsageError(f"Template `{name}` already exists.")
            seed = DEFAULT_TEMPLATE_CONTENT.encode("UTF-8")
        else:
            # Follow the established editor target, leaving any symlink in place.
            target = destination.resolve(strict=True)
            seed = target.read_bytes()
            permissions = stat.S_IMODE(target.stat().st_mode)

        destination.parent.mkdir(parents=True, exist_ok=True)
        drafts = destination.parent / ".drafts"
        drafts.mkdir(mode=0o700, exist_ok=True)
        drafts.chmod(0o700)
        directory = Path(tempfile.mkdtemp(dir=drafts))
        candidate = directory / destination.name
        candidate.touch(mode=0o600, exist_ok=False)
        candidate.write_bytes(seed)

        failure = "The editor did not finish successfully."
        click.edit(filename=str(candidate))
        failure = "Cannot read the saved draft."
        candidate.chmod(0o600)
        changed = candidate.read_bytes() != seed
        if not changed:
            _discard_draft(candidate)
            return False

        failure = "Cannot validate the saved draft."
        load_template(name, directory)
        failure = "Cannot publish the saved draft."
        if create:
            # Exclusive publication: an occupied name is never overwritten.
            os.link(candidate, destination)
        else:
            if destination.resolve(strict=True) != target or target.read_bytes() != seed:
                raise click.ClickException("The template changed while the editor was open.")
            # Keep the private recovery copy until replacement has succeeded.
            publication = directory / ".publish"
            shutil.copyfile(candidate, publication)
            publication.chmod(permissions)
            os.replace(publication, target)
    except (OSError, click.ClickException, TemplateError, KeyboardInterrupt) as error:
        if candidate is None:
            if isinstance(error, click.UsageError):
                raise
            raise click.ClickException(failure) from error
        try:
            candidate.chmod(0o600)
            unchanged = candidate.read_bytes() == seed
        except OSError:
            unchanged = False
        if unchanged:
            _discard_draft(candidate)
            recovery = ""
        else:
            recovery = f" Saved draft retained at: {candidate.absolute()}"
        if isinstance(error, (TemplateError, click.ClickException)):
            failure = str(error)
        raise click.ClickException(f"{failure}{recovery}") from error

    try:
        _discard_draft(candidate)
    except OSError as error:
        raise click.ClickException(
            f"Template saved, but cannot remove the draft at: {candidate.absolute()}"
        ) from error
    return True
