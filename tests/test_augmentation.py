from train.augmentation import oversample_rare_tldr


def test_rare_tools_duplicated_frequent_tools_not() -> None:
    records = [
        {"source": "tldr", "command": "raretool --flag"},
        {"source": "tldr", "command": "raretool --other"},
        {"source": "tldr", "command": "freqtool a"},
        {"source": "tldr", "command": "freqtool b"},
        {"source": "tldr", "command": "freqtool c"},
        # not tldr -- shouldn't count toward freq
        {"source": "synthetic", "command": "raretool ignored"},
    ]

    result = oversample_rare_tldr(records)  # default factor=3, threshold=2

    raretool_count = sum(1 for r in result if r["command"].startswith("raretool"))
    freqtool_count = sum(1 for r in result if r["command"].startswith("freqtool"))

    # raretool appears 2x in tldr (<= threshold) -> each of those 2 rows gets
    # (factor - 1) = 2 extra copies; the 3rd raretool row isn't tldr-sourced.
    assert raretool_count == 3 + 2 * 2
    # freqtool appears 3x in tldr (> threshold) -> no duplication.
    assert freqtool_count == 3
    assert len(result) == len(records) + 4
