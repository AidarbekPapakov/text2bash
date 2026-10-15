#!/usr/bin/env python3
"""Persistent text2bash server: loads the model once, then serves requests over
a local Unix domain socket -- avoids the ~8-13s torch/model-load cost on every
CLI call (see NOTES.md for the cold-start measurements).

CPU only, deliberately: GPU measured no faster for this model size (the
bottleneck was always import/load time, not generation compute), and a
persistent CPU process is simpler than managing a persistent CUDA context.

Single-threaded by design: this is a personal, single-user local tool, so
requests are handled one at a time rather than adding thread-safety around
concurrent model.generate() calls.

Start:   python cli/server.py
Stop:    Ctrl+C (or `kill $(cat ~/.text2bash/server.pid)`)
"""

import argparse
import json
import logging
import os
import signal
import socketserver
import sys
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings("ignore")
logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from transformers import logging as hf_logging  # noqa: E402

from common.bash import is_valid_bash  # noqa: E402

hf_logging.set_verbosity_error()

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n### Command:\n"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_DIR = str(PROJECT_ROOT / "train" / "out" / "lora-adapter")
DEFAULT_SOCKET = Path.home() / ".text2bash" / "server.sock"
DEFAULT_PID_FILE = Path.home() / ".text2bash" / "server.pid"


class Text2BashServer(socketserver.UnixStreamServer):
    allow_reuse_address = True

    def __init__(
        self, socket_path: Path, base_model: str, adapter_dir: str, default_max_new_tokens: int
    ) -> None:
        self.default_max_new_tokens = default_max_new_tokens

        print("Loading model (CPU)...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
        self.model = PeftModel.from_pretrained(model, adapter_dir)
        self.model.eval()
        print(f"text2bash server ready on {socket_path}", file=sys.stderr)

        super().__init__(str(socket_path), _Handler)

    def generate(self, instruction: str, max_new_tokens: int | None = None) -> str:
        prompt = PROMPT_TEMPLATE.format(instruction=instruction)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.default_max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return text.strip().split("\n")[0]


class _Handler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        try:
            line = self.rfile.readline()
            req = json.loads(line)
            command = self.server.generate(req["instruction"], req.get("max_new_tokens"))
            resp = {"command": command, "valid_bash": is_valid_bash(command)}
        except Exception as e:
            # Broad on purpose: a single malformed/failing request must not
            # take down the server loop for every other client.
            logging.exception("request failed")
            resp = {"error": str(e)}
        self.wfile.write((json.dumps(resp) + "\n").encode())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter_dir", default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--socket", default=str(DEFAULT_SOCKET))
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()

    server = Text2BashServer(socket_path, args.base_model, args.adapter_dir, args.max_new_tokens)

    DEFAULT_PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_PID_FILE.write_text(str(os.getpid()))

    def cleanup(*_: object) -> None:
        socket_path.unlink(missing_ok=True)
        DEFAULT_PID_FILE.unlink(missing_ok=True)
        sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        server.serve_forever()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
