"""Evaluate one or more models on the text2bash eval set.

For each model (base = zero-shot pretrained, or an adapter dir = fine-tuned),
generates a command for every eval example and reports, overall and split by
language:
  - exact match rate (normalized whitespace)
  - fuzzy match (difflib SequenceMatcher ratio, averaged)
  - bash -n syntax validity rate

Usage:
  python eval/evaluate.py --adapter_dir train/out/lora-adapter --label finetuned
  python eval/evaluate.py --label base                      # zero-shot baseline
"""
import argparse
import difflib
import json
import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Command:\n"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def is_valid_bash(cmd):
    if not cmd.strip():
        return False
    try:
        p = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True, timeout=5)
        return p.returncode == 0
    except Exception:
        return False


def normalize(cmd):
    return " ".join(cmd.split())


def fuzzy_ratio(a, b):
    return difflib.SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def generate_batch(model, tokenizer, instructions, device, max_new_tokens=48, batch_size=16):
    outputs = []
    for i in range(0, len(instructions), batch_size):
        batch = instructions[i:i + batch_size]
        prompts = [PROMPT_TEMPLATE.format(instruction=instr) for instr in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        for j in range(len(batch)):
            gen_ids = out[j][inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            command = text.strip().split("\n")[0]
            outputs.append(command)
    return outputs


def evaluate(model, tokenizer, device, records, label, max_new_tokens=48, batch_size=16):
    instructions = [r["instruction"] for r in records]
    predictions = generate_batch(model, tokenizer, instructions, device, max_new_tokens, batch_size)

    per_lang = {}
    rows = []
    for r, pred in zip(records, predictions):
        ref = r["command"]
        em = normalize(pred) == normalize(ref)
        fz = fuzzy_ratio(pred, ref)
        valid = is_valid_bash(pred)
        rows.append({"instruction": r["instruction"], "lang": r["lang"], "reference": ref,
                      "prediction": pred, "exact_match": em, "fuzzy": fz, "valid_bash": valid})
        bucket = per_lang.setdefault(r["lang"], {"n": 0, "em": 0, "fz": 0.0, "valid": 0})
        bucket["n"] += 1
        bucket["em"] += int(em)
        bucket["fz"] += fz
        bucket["valid"] += int(valid)

    summary = {"label": label, "n": len(records)}
    for lang, b in per_lang.items():
        summary[f"{lang}_exact_match"] = round(b["em"] / b["n"], 4)
        summary[f"{lang}_fuzzy"] = round(b["fz"] / b["n"], 4)
        summary[f"{lang}_valid_bash"] = round(b["valid"] / b["n"], 4)
        summary[f"{lang}_n"] = b["n"]
    total_em = sum(b["em"] for b in per_lang.values())
    total_fz = sum(b["fz"] for b in per_lang.values())
    total_valid = sum(b["valid"] for b in per_lang.values())
    summary["overall_exact_match"] = round(total_em / len(records), 4)
    summary["overall_fuzzy"] = round(total_fz / len(records), 4)
    summary["overall_valid_bash"] = round(total_valid / len(records), 4)

    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--adapter_dir", default=None, help="If set, evaluate base+adapter (fine-tuned). If omitted, evaluate zero-shot base.")
    ap.add_argument("--eval_file", default="data/final/eval.jsonl")
    ap.add_argument("--n_samples", type=int, default=None, help="Subsample eval set for speed")
    ap.add_argument("--label", default="model")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--output_json", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    if args.adapter_dir:
        model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(device)
    model.eval()

    records = load_jsonl(args.eval_file)
    if args.n_samples:
        records = records[:args.n_samples]

    summary, rows = evaluate(model, tokenizer, device, records, args.label, args.max_new_tokens, args.batch_size)

    print(json.dumps(summary, indent=2))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
