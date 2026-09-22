"""Small helpers shared across the app. No dependencies beyond the stdlib."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(name: str) -> str | None:
    """Read one key from the project's .env file.

    Deliberately tolerant of spaces around the '=' and of quoted values, because
    hand-edited .env files have both.
    """
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'") or None
    return None
