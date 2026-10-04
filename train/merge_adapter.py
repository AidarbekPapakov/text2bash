"""Merge a trained LoRA adapter into its base model and save the full merged checkpoint.

Needed before post-training quantization: bitsandbytes/torchao quantize a model's own
weights at load time, so the LoRA delta must already be folded into the base weights.
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--adapter_dir", default="train/out/lora-adapter")
    ap.add_argument("--output_dir", default="train/out/merged-bf16")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model = model.merge_and_unload()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved merged model to {args.output_dir}")


if __name__ == "__main__":
    main()
