"""Load the project .env file. Existing process variables are left unchanged."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_local_env(path: Path | None = None) -> bool:
    """Read KEY=value lines from the project .env file into the environment."""
    env_file = project_root() / ".env" if path is None else Path(path)
    if not env_file.is_file():
        return False
    return bool(load_dotenv(env_file, override=False))
