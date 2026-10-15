"""Post-training quantization sweep: evaluate the merged fine-tuned model at
bf16 (reference), fp8, int8, int4, and fp4, on a sample of the eval set.

fp8 and int4 use torchao weight-only quantization (works on any CUDA device --
weights are stored in low precision and upcast to bf16 for the actual matmul,
so no native fp8/int4 tensor cores are required, just memory savings).
int8 and fp4 use bitsandbytes (LLM.int8() and the fp4 4-bit float format).

Requires train/merge_adapter.py to have been run first (quantization needs the
LoRA delta already folded into the base weights).
"""

import argparse
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from eval.evaluate import evaluate, load_jsonl


def load_bf16(model_dir: str) -> Any:
    return AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")


def load_fp8(model_dir: str) -> Any:
    from torchao.quantization import Float8WeightOnlyConfig, quantize_

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")
    quantize_(model, Float8WeightOnlyConfig())
    return model


def load_int4(model_dir: str) -> Any:
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda")
    # torchao 0.18.0's Int4WeightOnlyConfig(version=2, the default) requires
    # an unreleased internal package ("mslk"); version=1 is also broken in
    # this release (hits an `assert config.version == 2` in torchao's own
    # code) and downgrading torchao to reach a working int4 path (0.14.1)
    # instead needs fbgemm-gpu-genai, which disables torchao's C++
    # extensions for this torch build -- destabilizing the already-verified
    # fp8 path for an unconfirmed int4 one. Documented as a known
    # environment limitation (see NOTES.md) rather than forced
    # through a fragile multi-package downgrade chain.
    quantize_(model, Int4WeightOnlyConfig(group_size=64))
    return model


def load_int8_bnb(model_dir: str) -> Any:
    config = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=config, device_map="cuda:0"
    )


def load_fp4_bnb(model_dir: str) -> Any:
    config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="fp4", bnb_4bit_compute_dtype=torch.bfloat16
    )
    return AutoModelForCausalLM.from_pretrained(
        model_dir, quantization_config=config, device_map="cuda:0"
    )


LOADERS = {
    "bf16": load_bf16,
    "fp8": load_fp8,
    "int4": load_int4,
    "int8": load_int8_bnb,
    "fp4": load_fp4_bnb,
}


def model_footprint_mb(model: Any) -> float:
    total_bytes = 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
    for b in model.buffers():
        total_bytes += b.numel() * b.element_size()
    return total_bytes / (1024**2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="train/out/merged-bf16")
    ap.add_argument("--eval_file", default="data/final/eval.jsonl")
    ap.add_argument("--n_samples", type=int, default=300)
    ap.add_argument("--dtypes", nargs="+", default=["bf16", "fp8", "int4", "int8", "fp4"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output_json", default="eval/out/quantization_sweep.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    records = load_jsonl(args.eval_file)
    if args.n_samples:
        records = records[: args.n_samples]

    results = {}
    for dtype in args.dtypes:
        print(f"\n=== {dtype} ===")
        try:
            model = LOADERS[dtype](args.model_dir)
            model.eval()
            footprint = model_footprint_mb(model)
            summary, _ = evaluate(
                model, tokenizer, "cuda", records, dtype, batch_size=args.batch_size
            )
            summary["footprint_mb"] = round(footprint, 1)
            results[dtype] = summary
            print(json.dumps(summary, indent=2))
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            # Broad on purpose: one dtype's loader failing (e.g. int4's known
            # mslk gap, see load_int4 above) shouldn't abort measuring the rest.
            print(f"FAILED: {dtype}: {e}")
            results[dtype] = {"error": str(e)}
            torch.cuda.empty_cache()

    import os

    os.makedirs("eval/out", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
