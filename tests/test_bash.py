import subprocess

from common.bash import is_valid_bash


def test_valid_command() -> None:
    assert is_valid_bash("echo hello") is True


def test_syntax_error() -> None:
    assert is_valid_bash("if [ 1 -eq 1 ]; then echo hi") is False  # missing fi


def test_empty_string() -> None:
    assert is_valid_bash("") is False


def test_whitespace_only_string() -> None:
    assert is_valid_bash("   \n\t  ") is False


def test_timeout_path(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="bash", timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert is_valid_bash("echo hello") is False
