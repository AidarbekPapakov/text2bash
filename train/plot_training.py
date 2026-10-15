"""Plot train/eval loss curves from a log_history.json saved by finetune_lora.py."""

import argparse
import json

import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log_history", help="Path to log_history.json")
    ap.add_argument("--output", required=True, help="Path to save the plot PNG")
    ap.add_argument("--title", default="Training loss")
    args = ap.parse_args()

    with open(args.log_history, encoding="utf-8") as f:
        history = json.load(f)

    train_steps = [e["step"] for e in history if "loss" in e]
    train_loss = [e["loss"] for e in history if "loss" in e]
    eval_steps = [e["step"] for e in history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in history if "eval_loss" in e]

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_steps, train_loss, label="train loss", color="#4C72B0", linewidth=1.5)
    ax.plot(
        eval_steps,
        eval_loss,
        label="eval loss",
        color="#DD8452",
        linewidth=2,
        marker="o",
        markersize=4,
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(args.title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
