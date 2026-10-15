#!/usr/bin/env python3
"""text2bash: translate a natural-language request (English or Russian) into a bash command.

Print-only by design: this tool never executes anything, it only prints a
suggested command. Always review a command before running it yourself.

If a persistent server (cli/server.py) is running, this talks to it over a
Unix socket for a fast response (no model reload). Otherwise it falls back
to loading the model directly in-process, same as before -- just slower.

Usage:
    python cli/text2bash.py find all files larger than 100MB
    python cli/text2bash.py "покажи использование диска в человекочитаемом виде"
"""

import argparse
import json
import os
import socket as socket_lib
import sys
from pathlib import Path

from common.bash import is_valid_bash

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Command:\n"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
# Resolved against this script's own location, not the caller's cwd, so the
# tool works when invoked from anywhere (e.g. `text2bash "..."` from $HOME),
# not only from inside the project directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_DIR = str(PROJECT_ROOT / "train" / "out" / "lora-adapter")
DEFAULT_SOCKET = Path.home() / ".text2bash" / "server.sock"


def try_server(instruction: str, socket_path: Path, max_new_tokens: int) -> tuple[str, bool] | None:
    """Returns (command, valid_bash) if a server answered, else None. Deliberately
    avoids importing torch/transformers -- this path should stay fast even when
    the server isn't running, since checking for it costs nothing but socket I/O."""
    if not socket_path.exists():
        return None
    try:
        s = socket_lib.socket(socket_lib.AF_UNIX, socket_lib.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(str(socket_path))
        req = json.dumps({"instruction": instruction, "max_new_tokens": max_new_tokens}) + "\n"
        s.sendall(req.encode())
        resp = json.loads(s.makefile().readline())
        s.close()
        if "error" in resp:
            return None
        return resp["command"], resp["valid_bash"]
    except (OSError, json.JSONDecodeError, TimeoutError):
        return None


def generate_direct(
    instruction: str, base_model: str, adapter_dir: str, use_gpu: bool, max_new_tokens: int
) -> str:
    """Fallback path: load the model in-process. Only imports the heavy ML stack
    here, so the server fast-path above never pays this cost when it doesn't
    need to."""
    import logging
    import warnings

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore")
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()

    use_cuda = use_gpu and torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.to(device)
    model.eval()

    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return text.strip().split("\n")[0]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Translate a natural-language request into a bash command. "
            "Prints only -- never executes."
        ),
    )
    ap.add_argument("instruction", nargs="+", help="What you want to do, in English or Russian")
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter_dir", default=DEFAULT_ADAPTER_DIR)
    ap.add_argument(
        "--gpu",
        action="store_true",
        help=(
            "Use CUDA if available (default: CPU -- measured no faster for this "
            "model size, see NOTES.md)"
        ),
    )
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument(
        "--no-server", action="store_true", help="Skip the persistent server even if one is running"
    )
    ap.add_argument("--server-socket", default=str(DEFAULT_SOCKET))
    args = ap.parse_args()

    instruction = " ".join(args.instruction)

    # Only use the server if it's running AND the caller hasn't asked for
    # something the server (CPU, default model/adapter) can't provide --
    # otherwise silently ignoring --base_model/--adapter_dir/--gpu would be
    # a correctness bug, not just a missed optimization.
    using_defaults = (
        args.base_model == DEFAULT_BASE_MODEL
        and args.adapter_dir == DEFAULT_ADAPTER_DIR
        and not args.gpu
    )
    result = None
    if using_defaults and not args.no_server:
        result = try_server(instruction, Path(args.server_socket), args.max_new_tokens)

    if result is not None:
        command, valid = result
    else:
        command = generate_direct(
            instruction, args.base_model, args.adapter_dir, args.gpu, args.max_new_tokens
        )
        valid = is_valid_bash(command)
        if using_defaults and not args.no_server:
            server_script = PROJECT_ROOT / "cli" / "server.py"
            venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
            print(
                f"# tip: start the persistent server for faster responses: "
                f"{venv_python} {server_script} &",
                file=sys.stderr,
            )

    if not valid:
        print(
            "# warning: suggested command failed a bash syntax check -- review carefully",
            file=sys.stderr,
        )

    print(command)


if __name__ == "__main__":
    main()
