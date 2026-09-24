"""Small helpers shared across the app. No third-party dependencies."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_env(name: str) -> str | None:
    """Find one configuration value, wherever this happens to be running.

    Looked for in order:
      1. the process environment
      2. Streamlit secrets, which is how a deployed app receives them
      3. the project's .env file, which is how it works locally

    Deliberately tolerant of spaces around the '=' and of quoted values in .env,
    because hand-edited files have both.
    """
    from_env = os.environ.get(name)
    if from_env:
        return from_env

    # Only when running inside Streamlit. Importing it elsewhere -- in tests, or
    # in the command line tools -- would be a pointless dependency, and reading
    # secrets outside a Streamlit run warns or raises depending on the version.
    try:
        import streamlit as st

        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass

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
