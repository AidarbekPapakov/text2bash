# text2bash

A small local language model that translates natural-language requests —
in English or Russian — into bash commands, so you don't have to google
command syntax. Fine-tuned Qwen3-0.6B-Base with LoRA; runs locally,
print-only (it never executes a command on your behalf, only suggests
one).

Built with every methodology choice (model, hyperparameters,
quantization) backed by an actual comparison run on this machine rather
than assumed.

## Results

Fine-tuned model vs. the zero-shot base model, on a held-out 2,593-example
eval set (full numbers: `eval/out/baseline.json`, `eval/out/finetuned.json`):

| Metric | Zero-shot baseline | Fine-tuned | Improvement |
|---|---|---|---|
| Exact match (overall) | 1.6% | 13.6% | 8.6x |
| Fuzzy match (overall) | 18.3% | 59.9% | 3.3x |
| Valid bash syntax (overall) | 22.6% | 98.5% | 4.4x |
| Exact match (English) | 1.4% | 12.0% | 8.7x |
| Exact match (Russian) | 3.6% | 28.9% | 8.1x |

The standout number: the un-fine-tuned base model only produces
*syntactically valid* bash 22.6% of the time on this task's prompt
format — fine-tuning took that to 98.5%. Exact-match is a strict metric
(many eval references are obscure, single-correct-answer commands like
`setsebool` or `mkvpropedit`), so fuzzy match is the fairer read of "how
close" the model gets even when not exact.

### Quantization

Post-training quantization sweep on the merged model (300-example
sample; see `notebooks/training_decisions.ipynb` for which technique
maps to which dtype and why):

| dtype | exact match | fuzzy | valid bash | size |
|---|---|---|---|---|
| bf16 (reference) | 14.3% | 58.0% | 99.0% | 1137MB |
| fp8 | 13.7% | 58.2% | 99.7% | 1434MB* |
| int8 | 14.0% | 56.1% | 98.0% | 717MB |
| fp4 | 7.3% | 49.5% | 96.7% | 507MB |
| int4 | — | — | — | not run, see below |

fp8 and int8 are essentially free (within ~1 point of bf16); fp4 trades
real accuracy for size. int4 could not be measured — torchao's int4 path
on this machine requires an unreleased internal package, and chasing a
working fallback meant downgrading torchao in a way that would have
destabilized the already-verified fp8 result. Documented as a real,
encountered limitation rather than silently skipped — see `NOTES.md`
Phase 4.1 for the full account.

\* fp8 here is *larger* than bf16, not smaller — this is a weight-only
quantization (stores a quantized tensor alongside dequantization
metadata, doesn't drop the original compute path) measuring accuracy
impact, not a memory-optimized deployment format. This machine's GPU
(RTX 3070 / Ampere) also has no native fp8 tensor cores, so this doesn't
represent a speed win either, only an accuracy measurement.

## Why this model, and why no bigger hyperparameter search

Before committing to Qwen3-0.6B-Base, 4 other small CPU/GPU-friendly
candidates were run zero-shot on this exact task (`notebooks/model_selection.ipynb`):
Qwen2.5-0.5B, and three encoder-decoder models (mT5-small, FLAN-T5-small,
ByT5-small — the architecture family the original NL2Bash research used).
All three encoder-decoder models failed outright zero-shot (sentinel
tokens, empty output, or hallucination); the Qwen decoder-only models
already produced near-correct bash with no fine-tuning at all. That's
what decided the architecture, not a guess taken on faith.

Similarly, LoRA rank/alpha were **not** swept — multiple sources
(notably Sebastian Raschka's multi-hundred-run ablation work) converge on
"negligible difference" once a sane default is picked. Learning rate, the
lever research says actually matters, *was* swept (a cheap 5-point check
on a data subset), and found 8e-4 clearly beats the naive default of
2e-4 (eval_loss 1.768 vs. 1.846 on the subset check; the full retrain
confirmed it: 1.281 vs. 1.430 final eval_loss). QLoRA and
quantization-aware training were both evaluated and skipped, with
specific reasons written down in `notebooks/training_decisions.ipynb`
rather than assumed either way.

This discipline — bake-off before picking a model, cheap research before
an expensive tuning decision, a real baseline before claiming a result,
decisions documented with sources — is what kept this from turning into
guesswork dressed up as engineering.

## Usage

Print-only: this tool suggests a command, it never runs one. Always
review before running it yourself.

```bash
# one-shot (slower, ~8-13s -- reloads the model every call)
.venv/bin/python cli/text2bash.py "find all files larger than 100MB"
.venv/bin/python cli/text2bash.py "покажи использование диска в человекочитаемом виде"

# with the persistent server (fast, ~0.5-1.3s per call after the first load)
.venv/bin/python cli/server.py &      # start once
.venv/bin/python cli/text2bash.py "..."   # now fast; falls back to one-shot automatically if the server isn't running
kill $(cat ~/.text2bash/server.pid)   # stop when done
```

Recommended as shell aliases for use from any directory (adjust the path
to wherever this repo lives):

```bash
alias t2b='/path/to/text2bash/.venv/bin/python /path/to/text2bash/cli/text2bash.py'
alias t2bstart='/path/to/text2bash/.venv/bin/python /path/to/text2bash/cli/server.py &'
alias t2bstop='kill $(cat ~/.text2bash/server.pid) 2>/dev/null'
```

Runs on CPU by default (`--gpu` is available but measured *no faster*
for this model size — the bottleneck is library import/load time, not
inference compute; see `NOTES.md` Phase 5 for the measurement). No
language flag needed; the model handles English and Russian input
directly.

## Reproducing from scratch

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers peft accelerate datasets huggingface_hub \
            bitsandbytes torchao jupyter nbformat nbconvert ipykernel \
            tiktoken sentencepiece protobuf

# Phase 1-2: build the dataset (seed + synthetic) -> data/final/dataset_v1.jsonl
#   see NOTES.md for the exact download/generation steps

# Phase 3: fine-tune
python train/finetune_lora.py --epochs 3 --lr 8e-4 --batch_size 8 --grad_accum 2

# Phase 4: evaluate against baseline
python eval/evaluate.py --label baseline --output_json eval/out/baseline.json
python eval/evaluate.py --adapter_dir train/out/lora-adapter --label finetuned --output_json eval/out/finetuned.json

# Phase 4.1: quantization sweep (optional)
python train/merge_adapter.py
python eval/quantize_eval.py --n_samples 300

# Phase 5: use it
python cli/text2bash.py "your request here"
```

Full phase-by-phase detail, every decision made and why, and every
number measured along the way is in `NOTES.md` — written to be read cold
by someone who wasn't in the original planning conversation.

## Project structure

```
data/
  seed/        NL2Bash + tldr-pages (EN + native RU), 47,074 records
  synthetic/   synthetically generated augmentation, 4,798 records
  final/       merged dataset + train/eval split (51,872 / 49,279 / 2,593)
notebooks/
  model_selection.ipynb      executed zero-shot bake-off across 5 candidate models
  training_decisions.ipynb   HPO/QLoRA/QAT research + quantization dtype/hardware table
train/
  finetune_lora.py   LoRA fine-tuning (GPU/CPU, configurable)
  generate.py         adapter sanity-check generation
  merge_adapter.py    folds a LoRA adapter into the base model
  out/lora-adapter     the final trained adapter (95MB, gitignored -- regenerate with finetune_lora.py)
eval/
  evaluate.py          exact-match/fuzzy/valid-bash harness, baseline vs. fine-tuned
  quantize_eval.py     same harness across quantization dtypes
cli/
  text2bash.py   the actual CLI (print-only)
  server.py      optional persistent server for fast repeat calls
NOTES.md         full project history: every phase, decision, and measurement
```

## Data sources & licenses

- [NL2Bash](https://github.com/TellinaTool/nl2bash) (TellinaTool) — English
  instruction → bash command pairs.
- [tldr-pages](https://github.com/tldr-pages/tldr) — English and native
  (community-translated, not machine-translated) Russian command examples.
- Base model: [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
  (Apache-2.0).
- Synthetic augmentation data was generated specifically for this
  project, not sourced from a third party.

## Known limitations

- **EN/RU imbalance**: training data skews ~9:1 toward English (improved
  from the seed data's ~17:1, but still imbalanced) — Russian performance
  is strong on this eval set but trained on meaningfully less data.
- **int4 quantization unmeasured** — see Quantization section above.
- **Scale**: trained on ~52k examples — a deliberate pilot-scale choice,
  documented in `NOTES.md`, with room to scale up the synthetic data
  generation further if warranted.
- **No execution, by design** — this tool will never run a command for
  you, including ones it's highly confident about. That's a safety
  decision, not a missing feature.
