# Project notes

Running log of how text2bash actually got built — decisions, dead ends,
and the numbers behind them. Mostly here so I don't forget *why* I made a
call six weeks from now, and so the README's claims have something to
point back to.

Started 2026-10-02.

## The idea

A small local model that turns a plain-English or Russian request into a
bash command — something I'd actually reach for instead of googling
syntax for the fifth time. Two things have to both be true for this to
count as a win: it has to be a decent portfolio piece (data generation at
scale, fine-tuning, a real eval harness, the whole thing benchmarked
properly), and it has to be something I'd genuinely use day to day. I'm
trying not to let either goal crowd out the other — no skipping the eval
harness to ship faster, no skipping the CLI polish for a cleaner results
table.

Early calls, mostly settled before writing any code:

- English **and** Russian input, one multilingual model — no separate
  language classifier, just train on mixed EN+RU pairs and let the model
  sort it out.
- Data: start from public datasets as seed, then generate synthetic
  augmentation on top to fill gaps and fix the EN/RU imbalance.
- Base model: pick empirically (see below), not by assumption.
- Print-only. The tool suggests a command, full stop — it never runs
  anything. That's non-negotiable; I review and run things myself.

## Seed data

Pulled [NL2Bash](https://github.com/TellinaTool/nl2bash) (12,607 English
instruction→command pairs) and [tldr-pages](https://github.com/tldr-pages/tldr)
(31,800 English examples, plus 2,667 **native** Russian ones — tldr
maintains its own Russian translations, not machine-translated, which
matters for data quality).

Had to parse tldr's `{{placeholder}}` template syntax down to plain
strings (took the first option on any `{{a|b}}` choice), and dropped
~800 "see docs for the real command" alias stubs that didn't carry any
actual instruction→command signal. Deduped everything on exact
(language, instruction, command).

Ended up with **47,074 records: 44,407 English / 2,667 Russian** —
`data/seed/seed_combined.jsonl`, with the per-source intermediates kept
alongside it. That ~17:1 EN/RU skew is rough; the plan is to lean hard
into Russian generation in the next step to start closing it.

## Synthetic augmentation (pilot round)

Rather than hand-write thousands more examples, ran this as a batch
generation job split across 8 topics (file ops, text processing,
process/system monitoring, networking, git, package management,
archiving/disk, shell scripting & env) — 300 EN + 300 RU per topic,
written to `data/synthetic/<topic>.jsonl`. Russian was generated at
parity with English per topic, not proportional to the seed's skew,
specifically to start closing that 17:1 gap.

After collecting everything: re-checked the JSON schema (clean), ran
`bash -n` on every single generated command (zero failures), and deduped
against both the seed set and itself (2 exact dupes against seed, 0
internal) — **4,798 synthetic records kept**, merged into
`data/final/dataset_v1.jsonl`.

Final tally: **51,872 records, 46,805 EN / 5,067 RU** — EN/RU ratio down
from ~17:1 to ~9:1. Still skewed, but moving the right direction. Spot
checked a random sample across languages and sources and it looked
right.

This was a pilot batch, not the full push — the original back-of-envelope
target was 140k+ examples, and this round only added under 5k. Decided to
treat 140k as a target, not a requirement, and revisit scaling it up
later rather than block everything on hitting a number I picked before
seeing how far a smaller set would get me.

## Picking a base model

Didn't want to just assume Qwen3-0.6B-Base and find out three hours into
training that it was the wrong call, so ran a quick zero-shot bake-off
first (`notebooks/model_selection.ipynb`) across 5 small, CPU-friendly
candidates: two decoder-only (Qwen3-0.6B-Base, Qwen2.5-0.5B) and three
encoder-decoder (mT5-small, FLAN-T5-small, ByT5-small — the architecture
family the original NL2Bash paper used). No fine-tuning, just load each
one and throw it at the project's own prompt format with one English and
one Russian test instruction.

Result wasn't close. All three encoder-decoder models failed outright —
mT5-small just emits its pretraining sentinel token (`<extra_id_0>`),
FLAN-T5-small generates nothing at all (immediate EOS), ByT5-small
hallucinates unrelated text. Both Qwen models, with zero fine-tuning,
produced near-correct or exactly-correct bash — Qwen2.5-0.5B nailed both
test prompts outright.

Went with **Qwen3-0.6B-Base** (newest in the family, the extra size over
Qwen2.5-0.5B wasn't a problem at this scale), with Qwen2.5-0.5B as the
documented fallback if training time ever becomes the bottleneck —
swapping it is a one-flag change in `train/finetune_lora.py`.

## Fine-tuning

Built `train/finetune_lora.py` (LoRA via peft + the HF Trainer, prompt
template `### Instruction:\n{instruction}\n### Command:\n`, loss masked
to the command completion) and `train/generate.py` for sanity-checking a
trained adapter against sample prompts with a `bash -n` check.

First did a small pipeline-verification run (2,000 train examples, 60
steps) just to make sure the training loop didn't have bugs — not meant
to produce a usable model. It worked: loss went 2.305 → 1.959, and on 5
held-out EN/RU prompts the adapter got 5/5 syntactically valid and 4/5
exactly right after only 60 steps. The one miss ("kill the process
listening on port 8080") guessed an unrelated nginx process — wrong
content, but not a pipeline bug, and the model had barely been trained
at that point.

Planned this as CPU-only originally, but partway through I noticed the
GPU (RTX 3070, 8GB) was actually mostly idle — something on the host
side had been holding onto it, and freeing that up let me reinstall
torch with CUDA support. Added a `--cpu` flag to fall back to if needed,
but GPU became the default. Worth it: a GPU re-run of the same
verification steps matched the CPU run's loss curve almost exactly but
took 28.6s instead of 478.9s — about 17x faster. Tried batch_size=32 and
hit CUDA OOM warnings; backed off to batch_size=8 / grad_accum=2
(effective 16), which ran clean.

Full run: 3 epochs over all 49,279 training examples, ~103 minutes.
eval_loss went 1.904 → 1.430, train_loss 1.423 final — a real drop, not
noise. The port-8080 prompt improved from the earlier nginx guess to
`kill $(lsof -t 8080)` — not fully idiomatic yet (missing the `-i:` port
filter) but clearly on the right track.

### Was a hyperparameter sweep worth it?

Checked the literature before spending compute on this rather than just
assuming a grid search was the right move. LoRA rank/alpha: multiple
sources — most notably Sebastian Raschka's large-scale LoRA ablation
work — agree the difference is negligible once you've picked a sane
default, so left rank=8/alpha=16 alone. Learning rate is the knob that
actually seems to matter, so ran a cheap sweep instead: 5 points
(1e-4 to 1.6e-3) on an 8k-example subset, 1 epoch each, about 3.5 minutes
per run.

| LR | eval_loss (1 epoch, 8k subset) |
|---|---|
| 1e-4 | 1.895 |
| 2e-4 (original default) | 1.846 |
| 4e-4 | 1.804 |
| **8e-4** | **1.768 (best)** |
| 1.6e-3 | 1.772 (plateaued, visibly less stable) |

8e-4 was a clean, monotonic improvement over the default — worth acting
on. Re-ran the full 3-epoch training at that LR and got eval_loss 1.281
(vs. 1.430 at the old default) — confirms the subset sweep held at full
scale. On the same 5 held-out prompts, the port-8080 case finally landed
on something genuinely correct and idiomatic:
`lsof -t -i:8080 | grep LISTEN | awk '{print $2}' | xargs kill -9`. This
run (`train/out/lora-adapter`, not committed here due to size — see
README) is the one everything downstream uses; the LR=2e-4 run is kept
locally as a comparison point.

Also checked whether QLoRA or QAT made sense here and decided no to
both, for reasons written up properly (and bluntly) in
`notebooks/training_decisions.ipynb` — short version: QLoRA solves a
memory problem this 596M model doesn't have (it fits comfortably in 8GB
VRAM with room to spare), and QAT only pays for itself if post-training
quantization turns out to lose real accuracy, which hadn't been measured
yet at that point.

## Evaluation

`eval/evaluate.py`: exact match (whitespace-normalized), fuzzy match
(difflib ratio), and `bash -n` syntax validity, each split by language,
over the full 2,593-example eval set. Baseline is the same base model
zero-shot, no adapter, same prompt template — didn't use a published
external number since nothing fits this exact task closely enough to be
a fair comparison.

| Metric | Baseline | Fine-tuned | Improvement |
|---|---|---|---|
| Overall exact match | 1.58% | 13.61% | 8.6x |
| Overall fuzzy match | 18.33% | 59.94% | 3.3x |
| Overall valid bash | 22.56% | 98.53% | 4.4x |
| EN exact match | 1.37% | 11.97% | 8.7x |
| RU exact match | 3.56% | 28.85% | 8.1x |

The number I keep coming back to: the untrained base model only produces
*syntactically valid* bash 22.6% of the time on this prompt format zero-
shot — fine-tuning took that to 98.5%. Exact match is a harsh metric
(plenty of eval references are obscure single-right-answer commands like
`mkvpropedit` or `setsebool`), so fuzzy match is the fairer read of how
close the model actually gets even when it's not a perfect match.

## Quantization sweep

Merged the LoRA adapter into the base model
(`train/merge_adapter.py` → a 1.2GB bf16 model, not kept around — it's a
pure function of base model + adapter, trivial to regenerate) and ran
`eval/quantize_eval.py` on a 300-example sample across bf16, fp8, int8,
int4, and fp4. Which library/technique maps to which dtype, and why, is
written up in `notebooks/training_decisions.ipynb`.

| dtype | exact match | fuzzy | valid bash | size (MB) |
|---|---|---|---|---|
| bf16 (reference) | 14.33% | 57.96% | 99.0% | 1136.9 |
| fp8 | 13.67% | 58.15% | 99.67% | 1433.6 |
| int8 | 14.0% | 56.11% | 98.0% | 716.9 |
| fp4 | 7.33% | 49.54% | 96.67% | 506.9 |
| int4 | — | — | — | couldn't get it to run |

fp8 and int8 are basically free — within a point of bf16 on every
metric, and int8 cuts the footprint by 37%. fp4 is the real tradeoff:
exact match nearly halves for a 55% size reduction. (fp8's footprint is
*larger* than bf16's here, not smaller — torchao's weight-only fp8
stores a quantized tensor plus dequant metadata without dropping the
original compute path, so this measures accuracy impact only, not a
deployment win — and this GPU has no native fp8 tensor cores anyway.)

int4 genuinely didn't work, and I didn't want to fake it by quietly
skipping it. torchao 0.18.0's default int4 path needs a package called
`mslk` that doesn't actually have a public release (the PyPI name exists
but is an empty placeholder). The documented fallback config is broken
in that same torchao version. Downgrading to the last torchao release
where int4 worked without `mslk` would have needed `fbgemm-gpu-genai`
and, worse, that older torchao disables its own C++ extensions against
this machine's torch build — which would've put the already-verified fp8
result in question too. Decided it wasn't worth a three-package downgrade
chain to "prove" one more number, so int4 is left unmeasured rather than
measured on a broken setup.

## CLI

`cli/text2bash.py`: prints the suggested command to stdout and nothing
else — a non-fatal warning goes to stderr only if the command fails a
`bash -n` check, so stdout stays clean enough to pipe or capture. No
execution path exists anywhere in the script.

Defaults to CPU even though training used GPU — those are independent
decisions. The whole point of the CLI is to be a lightweight thing that
works on any machine; training just wanted speed. Measured `--gpu` vs.
CPU default directly: GPU averaged ~11.0s per call, CPU ~9.7s — GPU was
not faster, if anything slightly slower. Almost the entire ~7-8s per
call is torch/transformers import and model load, not generation (which
is under 1.2s even cold, 0.28s warm) — there's just no meaningful
compute for a GPU to speed up at this model size, so it only adds CUDA
init overhead on top of the same load cost. Kept `--gpu` available since
it costs nothing to leave in and might matter for a bigger model later,
but it's not a "faster" flag for this one.

Hit a real bug during testing: the original default paths were relative
(`"train/out/lora-adapter"`), which broke the moment I ran it from `$HOME`
instead of the project directory — exactly the normal daily-use case —
with a confusing PEFT error because it tried to treat the unresolved path
as a Hugging Face Hub repo id. Fixed by resolving the default against the
script's own file location instead of the caller's cwd.

### Persistent server

Added `cli/server.py` — loads the model once and serves requests over a
Unix socket, so repeat calls skip the 7-8s load cost. CPU only, matching
the finding above. Single-threaded on purpose — this is a personal,
single-user tool, not worth building thread-safety around concurrent
`generate()` calls.

Measured a clear win: 3 calls against a running server came back in
1.30s, 0.47s, 0.54s vs. 7-13s without it — roughly 10-20x faster for
repeat use. The CLI tries the server first (a cheap socket check) and
only pays the torch/transformers import cost if it falls back to the
direct path, so running with no server active behaves exactly as before.

Aliases I actually use day to day (in `~/.bashrc`):

```bash
alias t2b='/home/aidar/projects/text2bash/.venv/bin/python /home/aidar/projects/text2bash/cli/text2bash.py'
alias t2bstart='/home/aidar/projects/text2bash/.venv/bin/python /home/aidar/projects/text2bash/cli/server.py &'
alias t2bstop='kill $(cat ~/.text2bash/server.pid) 2>/dev/null'
```

One gotcha worth remembering: the trailing `&` is baked into the
`t2bstart` alias itself, so it always backgrounds when run interactively
— but that means it doesn't compose with redirection at the call site
(`t2bstart > file.log 2>&1` won't actually redirect the server's output,
since the embedded `&` ends the command before the redirection applies).
Fine for normal use, worth knowing if I ever script around it.

## Open items / things I haven't done

- **Scale.** 140k+ examples was the original back-of-envelope target;
  this landed at ~52k. Worth another round or two of synthetic
  generation if I want to push accuracy further, but the current scale
  is enough for a real, usable first version.
- **EN/RU imbalance.** Improved from ~17:1 to ~9:1 but still skewed —
  future generation rounds should keep over-weighting Russian rather
  than generating at parity, if closing the gap further matters.
- **int4 quantization** — unmeasured, see above. Not blocking anything,
  just an honest gap.
- **Publishing** — haven't pushed the trained adapter anywhere (e.g.
  Hugging Face Hub) or decided whether it's worth doing. Low stakes,
  just hasn't happened yet.
