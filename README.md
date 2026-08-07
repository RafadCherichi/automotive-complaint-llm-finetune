# Automotive Complaint Safety Triage — QLoRA + DoRA Fine-Tuned Qwen3-8B

Fine-tunes Qwen3-8B (QLoRA + DoRA) on real NHTSA vehicle complaint data to extract
structured safety-triage JSON from raw free-text complaints — component, defect type,
safety risk, and severity — and measures the before/after accuracy against the
unmodified base model, honestly, including where it's still weak.

Project 3 of a 5-project PM+ML portfolio. Full locked scope and standing rules:
[`docs/blueprint.md`](docs/blueprint.md).

## How to read this project

In order, each building on the last:

1. **This README** — quick summary and headline results (below).
2. [`docs/pm-perspective.md`](docs/pm-perspective.md) — the business case: who this is
   for, what a miss costs vs. a false alarm, and why precision/recall (not accuracy) is
   the number that matters here.
3. [`docs/blueprint.md`](docs/blueprint.md) — the full locked design: task, dataset,
   base model, fine-tuning method, hardware constraints, and every open decision with
   its alternatives and reasoning.
4. **[`docs/learning/`](docs/learning/)** — the ML concepts behind every real decision
   (`01` through `06`, in order), taught from scratch (no assumed background), each as
   a 3-part card: what the concept is, exactly how this project uses it with a real
   code reference, and when the alternative would have actually been the better choice.
5. [`docs/label-strategy.md`](docs/label-strategy.md) — how the training data and
   labels were actually built, bugs found and fixed along the way.
6. [`docs/training-hyperparameters.md`](docs/training-hyperparameters.md) — how
   training actually ran across three real rounds, with real results, not just the
   locked hyperparameter table.
7. [`docs/code-walkthrough.md`](docs/code-walkthrough.md) — a guided tour of the actual
   code, file by file and function by function.
8. [`docs/eval-report.md`](docs/eval-report.md) — the final evaluation evidence, the
   honest error analysis, and why the shipped model is the one it is.

## Status

**Phases 1-3 done and shipped.** The trained model, the full base-vs-fine-tuned
evaluation, and the demo application code are all complete. **Phase 4 (local GGUF
quantization + demo) is paused** at a purely infrastructural step — not a modeling
problem — see [What's left](#whats-left-phase-4) below.

## Results

Measured on a fixed 140-example held-out set, identical across every round (verified
byte-for-byte unchanged throughout). Full detail, methodology, and honest limitations:
[`docs/eval-report.md`](docs/eval-report.md).

| Metric | Base (zero-shot) | Fine-tuned (shipped) | Change |
|---|---|---|---|
| JSON validity rate | 99.3% | **100.0%** | +0.7pp |
| component accuracy | 11.4% | **64.3%** | +52.9pp |
| defect_type accuracy | 4.3% | **67.1%** | +62.8pp |
| safety_risk accuracy | 36.4% | **85.0%** | +48.6pp |
| safety_risk=yes precision | 32.8% | **71.4%** | +38.6pp |
| safety_risk=yes recall | 93.3% | 88.9% | −4.4pp |

That last row isn't a regression — base "wins" on recall by flagging almost everything
as a risk (86 false positives out of 140, 32.8% precision). The fine-tuned model trades
a little recall for more than double the precision, which is what actually makes a
triage tool usable rather than a tool an analyst learns to ignore. Full reasoning, the
three-round retraining story behind the shipped model, and a measured (not assumed)
production-mitigation recommendation are in the eval report.

## How it was built

| Phase | What happened | Docs |
|---|---|---|
| 1 — Data | Streamed NHTSA's full 2.16M-row complaint database; derived all 4 target fields from NHTSA's own structured columns (zero manual labeling, zero synthetic data); built a stratified 900-row train / 140-row eval split | [`label-strategy.md`](docs/label-strategy.md) |
| 2 — Training | QLoRA + DoRA on Qwen3-8B via Unsloth, locked hyperparameters, 3 training rounds on Kaggle as the Phase 3 diagnosis drove real data fixes | [`training-hyperparameters.md`](docs/training-hyperparameters.md) |
| 3 — Evaluation | Base vs. fine-tuned on the held-out set across all 3 rounds; diagnosed a severity-tier blind spot, fixed it, found a new tradeoff, and made the shipping call with evidence, not a guess | [`eval-report.md`](docs/eval-report.md) |
| 4 — Local demo | Paused — see below | — |
| 5 — Docs | This README, `pm-perspective.md`, `docs/learning/` (6 concept cards), `code-walkthrough.md` | see [How to read this project](#how-to-read-this-project) |

## Repo structure

```
scripts/            data pipeline: NHTSA flat-file parsing, label derivation, dataset builds
notebooks/          train_qlora_dora.ipynb, eval_baseline_vs_finetuned.ipynb (Colab/Kaggle),
                     merge_and_quantize_gguf.ipynb (Phase 4, paused)
data/processed/      train.jsonl / eval.jsonl (gitignored — real NHTSA text, not synthetic)
models/              trained adapters (gitignored — see docs for which one shipped)
eval/                eval_results_v1/v2/v3.json — full graded predictions, all 3 rounds
demo/                streamlit_app.py — ready to run once a .gguf file exists
docs/                blueprint.md, pm-perspective.md, label-strategy.md,
                     training-hyperparameters.md, code-walkthrough.md, eval-report.md
docs/learning/       01-06: concept cards, one per real project decision
```

## What's left (Phase 4)

The shipped adapter, the evaluation, and `demo/streamlit_app.py` are all done and
waiting. What's left is one local-packaging step: merge the adapter into the base model
and quantize it to GGUF (Q4_K_M) so it runs on a 4GB VRAM card, per the hardware
contract in the blueprint.

This hit two rounds of real cloud-infrastructure friction, both diagnosed and fixed in
`notebooks/merge_and_quantize_gguf.ipynb`:
1. Kaggle's fixed 20GB disk couldn't fit the merge+conversion pipeline's peak disk
   usage (~32GB) — fixed by moving the notebook to Google Colab, which has more disk.
2. Colab's build then got OOM-killed compiling llama.cpp with unrestricted parallel
   jobs — fixed by capping the build to `-j 2`, plus hardening the notebook's
   success/failure checks so a truncated file can never again be silently treated as a
   successful conversion.

Neither fix has been confirmed against a real end-to-end run yet. To resume: run
`notebooks/merge_and_quantize_gguf.ipynb` on Colab, bring the resulting `.gguf` file
back to `models/`, then:
```
ollama create qwen3-8b-automotive-complaint -f models/Modelfile
venv/Scripts/python.exe scripts/verify_local_gguf.py
venv/Scripts/streamlit.exe run demo/streamlit_app.py
```

## Hardware contract

Local machine (8GB RAM, RTX 3050 4GB VRAM) never trains or merges the model — that
always runs on Colab/Kaggle's free-tier GPU. Locally it only does data prep, the eval
harness, and running the final quantized model. Full constraints: `docs/blueprint.md`
Section 1.
