"""Read application data at explicit paths; only writes create files."""

import json
import os
from pathlib import Path


class StorageError(ValueError):
    """Stored application data cannot be used."""


def load_config(config_path: Path) -> dict:
    """
    Reads the config file without creating or changing it.
    """
    try:
        with open(config_path, "r", encoding="UTF-8") as file:
            config = json.load(file)
    except (json.decoder.JSONDecodeError, UnicodeError, OSError):
        return {}

    if not isinstance(config, dict):
        return {}

    return config


def read_api_key(key_file_path: Path) -> str:
    """
    Return the OpenAI API key.
    """
    return _read_keys(key_file_path).get("openai", "").strip()


def _read_keys(key_file_path: Path) -> dict[str, str]:
    """Read the provider-key mapping without exposing invalid contents in errors."""
    try:
        with open(key_file_path, "r", encoding="UTF-8") as key_file:
            keys = json.load(key_file)
    except FileNotFoundError:
        return {}
    except (json.JSONDecodeError, UnicodeError):
        raise StorageError("keys.json must contain valid UTF-8 JSON.") from None
    except OSError:
        raise StorageError("Cannot read keys.json.") from None
    if not isinstance(keys, dict) or any(not isinstance(value, str) for value in keys.values()):
        raise StorageError("keys.json must contain an object with string key values.")
    return keys


def write_key(key_file_path: Path, key: str) -> None:
    """
    Write the OpenAI API key, preserving other provider entries.
    """
    if not isinstance(key, str):
        raise StorageError("API key must be a string.")
    keys = _read_keys(key_file_path)
    keys["openai"] = key
    key_file_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(key_file_path, os.O_WRONLY | os.O_CREAT, 0o600)
    with os.fdopen(descriptor, "w", encoding="UTF-8") as key_file:
        # Restrict the opened file before replacing any stored key.
        os.fchmod(key_file.fileno(), 0o600)
        key_file.truncate(0)
        json.dump(keys, key_file, indent=4)
        key_file.write("\n")
