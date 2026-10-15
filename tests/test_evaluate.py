from eval.evaluate import fuzzy_ratio, normalize


def test_normalize_collapses_whitespace() -> None:
    assert normalize("ls   -la\n") == normalize("ls -la")


def test_normalize_whitespace_differences_count_as_exact_match() -> None:
    a = "find   .   -name   '*.py'"
    b = "find . -name '*.py'"
    assert normalize(a) == normalize(b)


def test_fuzzy_ratio_identical_commands() -> None:
    assert fuzzy_ratio("ls -la", "ls -la") == 1.0


def test_fuzzy_ratio_whitespace_only_difference() -> None:
    assert fuzzy_ratio("ls   -la", "ls -la") == 1.0


def test_fuzzy_ratio_unrelated_commands_score_low() -> None:
    assert fuzzy_ratio("ls -la /tmp", "systemctl restart nginx") < 0.4
