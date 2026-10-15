"""Shared bash-syntax check used by the CLI, server, eval harness, and the
adapter sanity-check generator. A single implementation so all four agree on
what counts as valid bash (previously defined independently in each place,
and three of the four didn't check for an empty command -- `bash -n -c ""`
exits 0, so they reported an empty generation as valid bash)."""

import subprocess


def is_valid_bash(cmd: str) -> bool:
    if not cmd.strip():
        return False
    try:
        p = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True, timeout=5)
        return p.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False
