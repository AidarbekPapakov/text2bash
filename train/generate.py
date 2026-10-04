"""Load the base model + a LoRA adapter and generate bash commands for a few instructions.

Used to sanity-check a trained adapter: for each prompt, print the generated command
and whether it passes `bash -n` syntax validation.
"""
import argparse
import subprocess

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Command:\n"


def is_valid_bash(cmd):
    try:
        p = subprocess.run(["bash", "-n", "-c", cmd], capture_output=True, text=True, timeout=5)
        return p.returncode == 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--adapter_dir", default="train/out/lora-adapter")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument("prompts", nargs="*", default=[
        "list all files larger than 100MB in the current directory",
        "show the current git branch",
        "покажи использование диска в человекочитаемом виде",
        "найди все файлы с расширением .log старше 7 дней",
        "kill the process listening on port 8080",
    ])
    args = ap.parse_args()

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, args.adapter_dir)
    model.to(device)
    model.eval()

    for instruction in args.prompts:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        command = text.strip().split("\n")[0]
        valid = is_valid_bash(command)
        print(f"[{'OK' if valid else 'BAD'}] {instruction!r} -> {command!r}")


if __name__ == "__main__":
    main()
