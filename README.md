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
   (`01` through `07`, in order), taught from scratch (no assumed background), each as
   a 3-part card: what the concept is, exactly how this project uses it with a real
   code reference, and when the alternative would have actually been the better choice.
5. [`docs/label-strategy.md`](docs/label-strategy.md) — how the training data and
   labels were actually built, bugs found and fixed along the way.
6. [`docs/training-hyperparameters.md`](docs/training-hyperparameters.md) — how
   training actually ran across four real rounds (v1-v3, then v4), with real results,
   not just the locked hyperparameter table.
7. [`docs/code-walkthrough.md`](docs/code-walkthrough.md) — a guided tour of the actual
   code, file by file and function by function.
8. [`docs/eval-report.md`](docs/eval-report.md) — the final evaluation evidence, the
   honest error analysis, and why v4 (the current shipped model) is the one it is.

## Status

**Phases 1-3 done and shipped (v2). Round 4 done and shipped (v4, current model)** — a
label-noise investigation found and corrected real contradictions between some official
labels and the narrative text, added a 7-field target, retrained, and v4 now beats v2 on
nearly every headline metric. **Phase 4 (local GGUF quantization + local demo) is
formally descoped** — a structural infrastructure constraint, not a modeling problem —
see [Phase 4: descoped](#phase-4-descoped) below. A lightweight cloud-notebook demo
(`notebooks/demo_v4.ipynb`) replaces it.

## Results

Measured on a fixed 140-example held-out set, identical across every round (verified
byte-for-byte unchanged throughout). Numbers below are **v4**, the current shipped
model. Full detail, methodology, the v1→v2→v3→v4 comparison, and honest limitations:
[`docs/eval-report.md`](docs/eval-report.md).

| Metric | Base (zero-shot) | v4 (shipped) | Change |
|---|---|---|---|
| JSON validity rate | 100.0% | **100.0%** | +0.0pp |
| component accuracy | 12.1% | **67.1%** | +55.0pp |
| defect_type accuracy | 5.0% | **72.1%** | +67.1pp |
| safety_risk accuracy | 42.9% | **91.4%** | +48.5pp |
| safety_risk=yes precision | 35.5% | **88.4%** | +52.9pp |
| safety_risk=yes recall | 95.6% | 84.4% | −11.2pp |

That last row isn't a regression — base "wins" on recall by flagging almost everything
as a risk (very low precision). v4 trades some recall for far higher precision, which is
what actually makes a triage tool usable rather than a tool an analyst learns to ignore.
A forensic row-by-row check found this trade is better than the aggregate number alone
suggests: of every case where v4 and the previous shipped model (v2) disagree, v4 fixes
14 real mistakes and introduces 0 new clean ones — the 2 rows behind the recall dip are
contested edge cases the project's own label-noise audit now agrees with v4 on, not new
model weaknesses. Graded against that audit-corrected ground truth instead of the
official labels, `safety_risk` accuracy reaches **96.4%** (recall **95.2%**, precision
93.0%) — full forensic detail in the eval report.

Full reasoning, the four-round retraining story, the label-noise investigation behind
v4, and a measured (not assumed) production-mitigation recommendation are in the eval
report.

## How it was built

| Phase | What happened | Docs |
|---|---|---|
| 1 — Data | Streamed NHTSA's full 2.16M-row complaint database; derived all 4 target fields from NHTSA's own structured columns (zero manual labeling, zero synthetic data); built a stratified 900-row train / 140-row eval split | [`label-strategy.md`](docs/label-strategy.md) |
| 2 — Training | QLoRA + DoRA on Qwen3-8B via Unsloth, locked hyperparameters, 3 training rounds on Kaggle as the Phase 3 diagnosis drove real data fixes | [`training-hyperparameters.md`](docs/training-hyperparameters.md) |
| 3 — Evaluation | Base vs. fine-tuned on the held-out set across all 3 rounds; diagnosed a severity-tier blind spot, fixed it, found a new tradeoff, and made the shipping call with evidence, not a guess | [`eval-report.md`](docs/eval-report.md) |
| 4 — Local demo | **Descoped** — see below | — |
| 5 — Docs | This README, `pm-perspective.md`, `docs/learning/` (7 concept cards), `code-walkthrough.md` | see [How to read this project](#how-to-read-this-project) |
| Round 4 — Label correction + retrain | Found and corrected real label/text contradictions, added a 7-field atomic-decomposition target, retrained, shipped as v4 | [`eval-report.md`](docs/eval-report.md) §7, [`label-strategy.md`](docs/label-strategy.md) |

## Repo structure

```
scripts/            data pipeline: NHTSA flat-file parsing, label derivation, dataset builds
notebooks/          train_qlora_dora(_v4).ipynb, eval_baseline_vs_finetuned(_v4).ipynb (Colab/Kaggle),
                     demo_v4.ipynb (cloud-notebook demo — no local model file needed),
                     merge_and_quantize_gguf.ipynb (Phase 4, descoped — see below)
data/processed/      train.jsonl / eval.jsonl / train_v4.jsonl (gitignored — real NHTSA text, not synthetic)
models/              trained adapters (gitignored — see docs for which one shipped)
eval/                eval_results_v1/v2/v3.json, eval_results_v4_epoch2/epoch3.json — full graded predictions, all rounds
demo/                streamlit_app.py — built and ready, blocked on local model file availability (see below)
docs/                blueprint.md, pm-perspective.md, label-strategy.md,
                     training-hyperparameters.md, code-walkthrough.md, eval-report.md
docs/learning/       01-07: concept cards, one per real project decision
```

## Phase 4: descoped

**Not pursuing further.** Root cause, confirmed by two separate real runs, not a
one-off fluke: the merge-to-16-bit + GGUF-conversion step needs the merged model
(~16GB) and the F16 intermediate (~16.4GB) to coexist on disk at once — roughly 32GB
peak.
1. Kaggle's fixed 20GB disk limit couldn't fit that — confirmed twice.
2. Moving to Colab for its larger disk instead OOM-killed the from-source llama.cpp
   build, even after capping parallel compilation to `-j 2` — confirmed once, after
   that specific fix.

**This is a structural constraint of the quantization pipeline itself, not something
more retries, another platform, or a further round of infra fixes would resolve.**
Closing it out rather than continuing to spend cloud-GPU time chasing it.

`demo/streamlit_app.py` and `models/Modelfile` are left in the repo, not deleted — the
code is correct and would run immediately if a `.gguf` file were ever produced — but
both are clearly labeled as built-and-ready-but-blocked, and both still target the
superseded 4-field v2 model rather than v4's 7-field one.

**The project's evidence base is [`docs/eval-report.md`](docs/eval-report.md) and the
label-noise investigation, not a live local demo — a deliberate, disclosed scope
decision, not an oversight.** For an actually-runnable demo, use
[`notebooks/demo_v4.ipynb`](notebooks/demo_v4.ipynb) instead: it loads the base model +
v4 adapter in 4-bit on a Colab/Kaggle GPU, the same proven pattern used in every
training and eval run in this project — no merge, no GGUF conversion, no local disk or
RAM constraint to hit.

## Hardware contract

Local machine (8GB RAM, RTX 3050 4GB VRAM) never trains or merges the model — that
always runs on Colab/Kaggle's free-tier GPU. Locally it only does data prep, the eval
harness, and running the final quantized model. Full constraints: `docs/blueprint.md`
Section 1.
