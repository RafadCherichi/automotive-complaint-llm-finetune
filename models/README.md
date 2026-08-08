# Adapter versions

Not committed to git (gitignored, like all model weights in this repo) -- this file just
orients whoever's looking at the local `models/` folder.

| Folder | Trained on | Status |
|---|---|---|
| `qwen3-8b-automotive-complaint-lora/` | Phase 1 `train.jsonl` (800 rows) | **v1** -- superseded experiment, kept for reference |
| `qwen3-8b-automotive-complaint-lora-v2/` | injury-only-high fix (800 rows, swapped) | **v2** -- kept for reference; identical weights also live in `-FINAL/` |
| `qwen3-8b-automotive-complaint-lora-v3/` | medium-tier restore (900 rows, additive) | **v3** -- superseded experiment, kept for reference |
| `qwen3-8b-automotive-complaint-lora-FINAL/` | (copy of v2) | **Rounds 1-3 shipped model.** |
| `qwen3-8b-automotive-complaint-lora-v4-epoch3/` | Round 4 `train_v4.jsonl` (900 rows, 7-field target), full 3 epochs | **v4 candidate A** -- lost the eval comparison, kept for reference |
| `qwen3-8b-automotive-complaint-lora-v4-epoch2/` | same run as epoch3, checkpoint-114 (end of epoch 2) | **v4 candidate B -- won the eval comparison**; identical weights also live in `-v4-FINAL/` |
| `qwen3-8b-automotive-complaint-lora-v4-FINAL/` | (copy of v4-epoch2) | **Round 4 shipped model.** Use this one going forward. |

## Round 4: two candidate checkpoints, decided on task metrics (not loss alone)

v4's epoch2->epoch3 eval loss regression was small (0.7723 → 0.7772, 0.63%) — smaller
than v1's (0.83%) and v2's (0.79%) — small enough that it could have been validation
noise on a 140-example eval set rather than confirmed overfitting. So this round didn't
default to "epoch 2 wins" the way v1-v3 did — both checkpoints were run through the full
eval notebook and compared on real downstream task metrics.

**Epoch2 won, 11 metrics to 3 (3 ties)** — and specifically won on every `safety_risk`
metric (both the official-label and text-consistent framings) and every `severity`
metric, including the two historically hardest tiers (`medium` +3.7pp, `high` +5.6pp).
Epoch3 only won on `component` (a lower-priority field — `safety_risk` is this project's
load-bearing number, per `blueprint.md` Section 3) and `crash_described` by a thin
0.7pp margin. Full numbers in `docs/eval-report.md`'s Round 4 section.

This validates the same direction v1/v2/v3 always took by default (earlier checkpoint
over later, on this dataset size/epoch count) -- but this time on real evidence instead
of an assumption carried over from loss alone.

## Why v2 shipped, not v3 (Rounds 1-3)

See `docs/training-hyperparameters.md` and `docs/label-strategy.md` for the full
three-round comparison. Short version: v2 is the only round that solved the
`severity: high` detection problem (0% → 88.9% accuracy) while also having the best
`safety_risk` recall (88.9%) -- the number that matters most for a safety-triage tool.
v3 had better precision (86.1%) but regressed `severity: high` back to 0%, which isn't
an acceptable tradeoff when the whole point is not missing real safety signals.
