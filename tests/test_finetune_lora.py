from train.finetune_lora import PROMPT_TEMPLATE, build_dataset


class FakeTokenizer:
    """One 'token id' per whitespace-separated word (its length). eos_token has
    a leading space so it always splits off as its own token when concatenated
    directly onto a command, the way build_dataset does."""

    eos_token = " <EOS>"

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        return {"input_ids": [len(w) for w in text.split()]}


def test_build_dataset_masks_prompt_and_appends_eos() -> None:
    tok = FakeTokenizer()
    records = [{"instruction": "list files", "command": "ls -la"}]
    ds = build_dataset(records, tok, max_length=96)
    row = ds[0]

    prompt_len = len(PROMPT_TEMPLATE.format(instruction="list files").split())
    completion_len = len(("ls -la" + tok.eos_token).split())

    assert len(row["input_ids"]) == prompt_len + completion_len
    assert row["labels"][:prompt_len] == [-100] * prompt_len
    assert row["labels"][prompt_len:] == row["input_ids"][prompt_len:]
    assert row["attention_mask"] == [1] * len(row["input_ids"])
    assert row["input_ids"][-1] == len("<EOS>")  # the appended EOS token


def test_build_dataset_truncation_keeps_inputs_and_labels_aligned() -> None:
    tok = FakeTokenizer()
    records = [
        {
            "instruction": "a very long winded instruction that keeps going and going",
            "command": "echo one two three four five six seven eight",
        }
    ]
    ds = build_dataset(records, tok, max_length=5)
    row = ds[0]

    assert len(row["input_ids"]) == 5
    assert len(row["labels"]) == 5
    assert len(row["attention_mask"]) == 5
