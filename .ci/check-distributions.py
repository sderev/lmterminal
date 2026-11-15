"""Build offline from tracked sources with synthetic local data; inspect all archives.

Run from the repository root: uv run --no-sync python .ci/check-distributions.py
Requires uv and a cached backend satisfying the project's build-system requirement.
"""

import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path


def build(source, output, kind):
    subprocess.run(
        [
            "uv",
            "--no-config",
            "build",
            "--offline",
            "--no-python-downloads",
            "--force-pep517",
            "--python",
            sys.executable,
            f"--{kind}",
            "--out-dir",
            str(output),
            str(source),
        ],
        check=True,
    )
    archives = list(output.glob("*.whl" if kind == "wheel" else "*.tar.gz"))
    assert len(archives) == 1, archives
    return archives[0]


def contents(archive):
    if archive.suffix == ".whl":
        with zipfile.ZipFile(archive) as wheel:
            return {name: wheel.read(name) for name in wheel.namelist() if not name.endswith("/")}
    with tarfile.open(archive) as sdist:
        return {
            member.name.partition("/")[2]: sdist.extractfile(member).read()
            for member in sdist.getmembers()
            if member.isfile()
        }


def main():
    root = Path(__file__).resolve().parent.parent
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    with tempfile.TemporaryDirectory(prefix="lmt-distributions-") as temporary:
        scratch = Path(temporary)
        source = scratch / "source"
        for name in filter(None, tracked):
            destination = source / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)

        hidden = [
            "lmterminal/.venv/local-marker.txt",
            "lmterminal/.claude/local-marker.json",
            "lmterminal/.ruff_cache/local-marker.txt",
            "lmterminal/assets/.nested/local-marker.txt",
            "lmterminal/.local-data",
        ]
        for name in hidden:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic local data\n")
        asset = "lmterminal/assets/kept.txt"
        (source / asset).write_text("synthetic package asset\n")
        expected = {
            name: (source / name).read_bytes()
            for name in [*filter(lambda name: name.startswith("lmterminal/"), tracked), asset]
        }
        license_text = (source / "LICENSE").read_bytes()

        wheel = build(source, scratch / "direct", "wheel")
        sdist = build(source, scratch / "sdist", "sdist")
        rebuilt = build(sdist, scratch / "rebuilt", "wheel")
        for label, archive in [
            ("direct wheel", wheel),
            ("sdist", sdist),
            ("rebuilt wheel", rebuilt),
        ]:
            files = contents(archive)
            assert not set(hidden) & files.keys(), files.keys()
            assert all(
                not part.startswith(".")
                for name in files
                if name.startswith("lmterminal/")
                for part in Path(name).parts
            ), files.keys()
            assert all(files.get(name) == data for name, data in expected.items()), files.keys()
            licenses = [data for name, data in files.items() if Path(name).name == "LICENSE"]
            assert licenses == [license_text], files.keys()
            print(
                f"{label}: {len(expected)} package files retained byte-for-byte; local data absent"
            )
            print("\n".join(sorted(files)))


if __name__ == "__main__":
    main()
