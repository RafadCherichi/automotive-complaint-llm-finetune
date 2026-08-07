# Adapter versions

Not committed to git (gitignored, like all model weights in this repo) -- this file just
orients whoever's looking at the local `models/` folder.

| Folder | Trained on | Status |
|---|---|---|
| `qwen3-8b-automotive-complaint-lora/` | Phase 1 `train.jsonl` (800 rows) | **v1** -- superseded experiment, kept for reference |
| `qwen3-8b-automotive-complaint-lora-v2/` | injury-only-high fix (800 rows, swapped) | **v2** -- kept for reference; identical weights also live in `-FINAL/` |
| `qwen3-8b-automotive-complaint-lora-v3/` | medium-tier restore (900 rows, additive) | **v3** -- superseded experiment, kept for reference |
| `qwen3-8b-automotive-complaint-lora-FINAL/` | (copy of v2) | **Shipped model.** Use this one for Phase 4. |

## Why v2 shipped, not v3

See `docs/training-hyperparameters.md` and `docs/label-strategy.md` for the full
three-round comparison. Short version: v2 is the only round that solved the
`severity: high` detection problem (0% → 88.9% accuracy) while also having the best
`safety_risk` recall (88.9%) -- the number that matters most for a safety-triage tool.
v3 had better precision (86.1%) but regressed `severity: high` back to 0%, which isn't
an acceptable tradeoff when the whole point is not missing real safety signals.
