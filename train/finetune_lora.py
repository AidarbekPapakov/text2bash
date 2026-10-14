"""LoRA fine-tune a small causal LM on the text2bash (instruction -> bash command) dataset.

Runs on GPU automatically when CUDA is available (pass --cpu to force CPU, matching
the project's original CPU-only premise). Use --max_steps for a quick
pipeline-verification run, or omit it (use --epochs) for a full run.
"""

import argparse
import json
import os
from typing import Any

from datasets import Dataset

from train.augmentation import oversample_rare_tldr

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Command:\n"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_dataset(records: list[dict], tokenizer: Any, max_length: int) -> Dataset:
    def tokenize(example: dict) -> dict:
        prompt = PROMPT_TEMPLATE.format(instruction=example["instruction"])
        completion = example["command"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]

        input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids

        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

    ds = Dataset.from_list(records)
    return ds.map(tokenize, remove_columns=ds.column_names)


def main() -> None:
    # Deferred so importing this module (e.g. to test build_dataset) doesn't
    # require the full torch/peft/transformers stack -- same reasoning as
    # cli/text2bash.py's generate_direct().
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--train_file", default="data/final/train.jsonl")
    ap.add_argument("--eval_file", default="data/final/eval.jsonl")
    ap.add_argument("--output_dir", default="train/out/lora-adapter")
    ap.add_argument("--max_length", type=int, default=96)
    ap.add_argument(
        "--train_subset",
        type=int,
        default=None,
        help="If set, use only this many training examples (for quick verification runs)",
    )
    ap.add_argument("--eval_subset", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument(
        "--max_steps",
        type=int,
        default=-1,
        help="If >0, stop after this many steps regardless of --epochs",
    )
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument(
        "--cpu", action="store_true", help="Force CPU training even if CUDA is available"
    )
    ap.add_argument(
        "--no_oversample_rare_tldr",
        action="store_true",
        help="Disable oversampling of low-frequency tldr tools (on by default)",
    )
    ap.add_argument(
        "--no_load_best_model",
        action="store_true",
        help="Disable restoring the best eval_loss checkpoint at the end (on by default)",
    )
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    dtype = torch.bfloat16 if use_cuda else torch.float32
    device_desc = "cuda (" + torch.cuda.get_device_name(0) + ")" if use_cuda else "cpu"
    print(f"Device: {device_desc}, dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = load_jsonl(args.train_file)
    if not args.no_oversample_rare_tldr:
        train_records = oversample_rare_tldr(train_records)
    train_ds = build_dataset(train_records, tokenizer, args.max_length)
    eval_ds = build_dataset(load_jsonl(args.eval_file), tokenizer, args.max_length)

    if args.train_subset:
        train_ds = train_ds.select(range(min(args.train_subset, len(train_ds))))
    if args.eval_subset:
        eval_ds = eval_ds.select(range(min(args.eval_subset, len(eval_ds))))

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=max(args.logging_steps * 5, 50),
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        use_cpu=not use_cuda,
        bf16=use_cuda,
        remove_unused_columns=False,
        load_best_model_at_end=not args.no_load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    log_history_path = os.path.join(args.output_dir, "log_history.json")
    with open(log_history_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print(f"Saved LoRA adapter to {args.output_dir}")
    print(f"Saved training log history to {log_history_path}")


if __name__ == "__main__":
    main()
