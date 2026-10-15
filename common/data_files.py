"""Shared list of large data files published to the Hugging Face dataset repo
and restored by scripts/download_data.py.

data/final/eval.jsonl is deliberately excluded -- it stays in git so the eval
set is visible without a download step. data/synthetic/* is also excluded --
those files are small and were never a git-bloat problem.
"""

DATA_FILES = [
    "data/seed/nl2bash_all.cm",
    "data/seed/nl2bash_all.nl",
    "data/seed/seed_en.jsonl",
    "data/seed/seed_combined.jsonl",
    "data/seed/tldr_en.jsonl",
    "data/seed/tldr_ru.jsonl",
    "data/final/dataset_v1.jsonl",
    "data/final/train.jsonl",
]
