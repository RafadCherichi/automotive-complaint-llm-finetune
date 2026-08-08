# Phase 2 training hyperparameters (locked)

Proposed options-first, confirmed, then checked against real data before locking.
Encoded in `notebooks/train_qlora_dora.ipynb`.

| Hyperparameter | Value | Reasoning |
|---|---|---|
| LoRA rank (r) | 16 | Middle ground for structured extraction on ~800 examples — enough capacity to learn a 4-field JSON schema without overfitting a small dataset. Higher (32/64) adds parameters to fit without more data to fit them on — the wrong direction with this little training data. |
| LoRA alpha | 32 | Standard alpha = 2×rank scaling convention. |
| DoRA | `use_dora=True` | Locked project-wide (blueprint.md) — same VRAM cost as plain LoRA, verified as a real, supported Unsloth parameter. |
| Learning rate | 2e-4 | Standard QLoRA default for 7-8B models. |
| Epochs | 3 | Small-dataset QLoRA norm. Eval loss logged per epoch (not just at the end) specifically to watch for overfitting — a real risk on only 800 examples, not assumed away. |
| Batch size | 4 per step × 4 gradient-accumulation steps = effective batch 16 | Pure batch-4 with no accumulation is small and noisy. Accumulating to an effective 16 gives more stable gradients. VRAM fit is an estimate (4-bit base ~4-5GB + LoRA/DoRA adapter overhead + activations with gradient checkpointing, should comfortably fit a 16GB T4) — confirmed empirically when the notebook actually runs, not assumed, per the hardware contract that forbids testing this locally. |
| Max sequence length | 768 | Not assumed from the 512 default. Checked against the *actual* Qwen3 tokenizer (`scripts/check_seq_lengths.py`) over all 940 examples formatted as the real instruction+narrative+JSON string fed to the model. Result: p95 = 528 tokens, 5.2% of examples (49/940) exceeded 512 — a meaningful truncation rate, not noise. 768 covers 99.9% (only 1 example over, by ~14 tokens). 1024 would cover 100% but roughly doubles attention compute for zero practical benefit beyond that one edge case. |

## How the sequence-length check was done

`scripts/check_seq_lengths.py` installs `transformers` **without torch** (tokenizer-only,
CPU text processing — not a training dependency, no GPU, no model weights loaded) to
load Qwen3's real tokenizer and measure the formatted-example token count directly,
rather than approximating from character count. Full distribution:

```
examples: 940
min: 123  max: 782
p50: 238  p90: 414  p95: 528  p99: 712

examples exceeding 256 tokens: 401 (42.7%)
examples exceeding 384 tokens: 116 (12.3%)
examples exceeding 512 tokens: 49 (5.2%)
examples exceeding 768 tokens: 1 (0.1%)
examples exceeding 1024 tokens: 0 (0.0%)
```

## Model checkpoint

`unsloth/Qwen3-8B-unsloth-bnb-4bit` — Unsloth's pre-quantized 4-bit build of
`Qwen/Qwen3-8B` (the instruct/chat model; `-Base` is the separate raw-pretrain
checkpoint). Verified to exist on the Hugging Face Hub (514K+ downloads) before writing
it into the notebook, rather than assumed from memory.

## This machine never runs the notebook itself

This machine has no way to execute `notebooks/train_qlora_dora.ipynb` directly — it
needs a live Colab or Kaggle GPU session, per the hardware contract in
`docs/blueprint.md`. It was built and verified correct against documented Unsloth/TRL
APIs before ever being run, then actually executed on Kaggle three times (see below) —
each run's real output (eval losses, adapter files) was checked afterward, not assumed
to have gone as planned.

## Actual training results (Kaggle T4) — three rounds

Same locked hyperparameters all three times — only the training data changed between
rounds:
- **v1** — original `train.jsonl` from Phase 2 (800 rows).
- **v2** — after the Phase 3 diagnosis found the model under-weighting injury-only
  (no crash/fire) language for `severity: high`; 57 `medium` examples swapped for 57
  new sub-pattern-A `high` examples (still 800 rows). See `docs/label-strategy.md`.
- **v3** — after v2's eval found `severity: medium` accuracy had collapsed; 100 new
  `medium` examples added on top with nothing removed (900 rows). See
  `docs/label-strategy.md`.

| Epoch | v1 eval loss | v2 eval loss | v3 eval loss |
|---|---|---|---|
| 1 | 1.0770 | 1.0745 | 1.0722 |
| **2** | **1.0595 — best, selected** | **1.0600 — best, selected** | **1.0561 — best, selected** |
| 3 | 1.0683 — regressed | 1.0684 — regressed | (not separately confirmed for v3; same checkpoint-selection logic applied) |

Epoch 3 regressing happened in both v1 and v2, at almost identical loss values — a
repeatable signal, not a fluke, that 3 full passes over a dataset this size sits right at
the edge of overfitting. Epoch 2 was selected each round for that reason.
`checkpoint-100` (v1/v2, 800 rows ÷ effective batch 16 = 50 steps/epoch) and
`checkpoint-114` (v3, 900 rows ÷ 16 = 57 steps/epoch, ceil-rounded) both land exactly on
the end of epoch 2 — confirmed against each run's own `global_step`, not assumed.

**Adapter locations:**
- v1: `models/qwen3-8b-automotive-complaint-lora/` — superseded, kept for reference
- v2: `models/qwen3-8b-automotive-complaint-lora-v2/` — **the shipped model**; identical weights also copied to `models/qwen3-8b-automotive-complaint-lora-FINAL/` for Phase 4
- v3: `models/qwen3-8b-automotive-complaint-lora-v3/` — superseded, kept for reference

(all gitignored, not committed — same as other model-weight artifacts in this repo; see `models/README.md` for the same summary)

**Completeness check performed on the extracted files, all three times** (not just
assumed from the download succeeding):
- `adapter_config.json` matches the locked hyperparameters exactly every time: `r=16`,
  `lora_alpha=32`, `use_dora=true`, `target_modules` = the 7 modules specified,
  `base_model_name_or_path=unsloth/Qwen3-8B-unsloth-bnb-4bit`.
- `adapter_model.safetensors` (180MB, 756 tensors, ~45M params, identical shape every
  run) parsed with `safetensors.safe_open` — not corrupted/truncated, all three times.
  Covers all 36 transformer layers and all 7 target modules, with `lora_A`, `lora_B`,
  **and `lora_magnitude_vector`** tensors present for every one — concrete confirmation
  DoRA actually trained (the magnitude vector is DoRA-specific; plain LoRA wouldn't have
  it), not just configured.
- `trainer_state.json`'s own logged eval losses match what was reported for each run,
  and `global_step` lands exactly on the end-of-epoch-2 step count every time —
  internally consistent with each genuinely being the epoch 2 checkpoint, not a
  mismatched or partial copy.
- The `FINAL` copy was verified byte-identical (sha256) to its `v2` source after
  copying — the copy operation itself didn't corrupt anything.
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) are
  present and non-trivial size, all three runs.

**One informational note, not a defect (applies to all three downloads):** the zip is a
raw copy of the Trainer's checkpoint directory, not the notebook's dedicated
`save_pretrained()` export cell — it includes `optimizer.pt`, `scheduler.pt`,
`scaler.pt`, and `rng_state.pth` (~104MB combined), which are training-resumption
state, not needed for inference in Phase 3/4. Harmless to leave in place; only the
`adapter_*` and tokenizer files are actually needed going forward.

## Phase 3 final result: three rounds, full comparison, and what shipped

Full data in `eval/eval_results_v1.json`, `eval_results_v2.json`, `eval_results_v3.json`
(each graded on the identical, untouched 140-example `eval.jsonl` — confirmed
byte-identical across every round by sha256, so all three are directly comparable).

| Metric | v1 | v2 | v3 |
|---|---|---|---|
| JSON validity | 100.0% | 100.0% | 100.0% |
| component accuracy | 68.6% | 64.3% | 66.4% |
| defect_type accuracy | 69.3% | 67.1% | 71.4% |
| safety_risk accuracy | 81.4% | 85.0% | 86.4% |
| severity accuracy (blended) | 70.0% | 70.7% | 75.7% |
| **safety_risk=yes precision** | 68.6% | 71.4% | **86.1%** |
| **safety_risk=yes recall** | 77.8% | **88.9%** | 68.9% |
| severity: low accuracy | 83.2% | 83.2% | 94.7% |
| severity: medium accuracy | 70.4% | 14.8% | 59.3% |
| **severity: high accuracy** | **0.0%** | **88.9%** | **0.0%** |

**v2 shipped.** Rationale: v2 is the only round that solved the `severity: high`
detection problem (0% → 88.9%) while also having the best `safety_risk` recall (88.9%)
— the number the project's PM safety-triage framing (blueprint.md Section 3)
prioritizes above all others, since a missed high-severity complaint is the failure mode
the whole tool exists to prevent. v3 has the best blended accuracy numbers almost across
the board, including the best precision (86.1%) — but it regressed `severity: high` back
to 0%, the exact same failure v1 had. Optimizing for blended accuracy or precision alone
would have shipped a model that never correctly flags a genuinely high-severity
complaint. That's not an acceptable tradeoff for a safety-triage tool, even though it
would look better on a naive metrics table.

**Also notable, though not the deciding factor:** growing `medium` from 100 to 200
examples (v3's fix, targeting the standard 200–500-per-class QLoRA range) didn't
actually fix `medium`'s own accuracy problem either — it went from 14.8% (v2) to 59.3%
(v3), better but still well below v1's 70.4%, while accidentally undoing v2's `high` fix
in the process. This suggests the three severity tiers aren't cleanly separable by
example count alone within this rule-derived label scheme — a real, open question for
any future iteration of this project, not something resolved here. Recorded as a known
limitation, not chased with a fourth round, per the explicit decision to stop after
three.

## Round 4 training run

Same locked hyperparameters as v1-v3 (`r=16`, `alpha=32`, `use_dora=True`, `lr=2e-4`, 3
epochs) — only the training data and target schema changed. `MAX_SEQ_LENGTH` raised
768→896, a mechanical consequence of the longer 7-field target JSON, checked against the
real tokenizer (`scripts/check_seq_lengths_v4.py`), not assumed.

Trained on `data/processed/train_v4.jsonl` — see `docs/label-strategy.md`'s Round 4
section for the label-correction methodology. Epoch2→epoch3 eval loss regression was
small (0.7723→0.7772, 0.63% — smaller than v1's 0.83% and v2's 0.79%), so the checkpoint
choice wasn't made on loss alone this round: both checkpoints were fully evaluated and
compared on real downstream task metrics (`docs/eval-report.md` Section 7). **Epoch2
won, 11 metrics to 3, and shipped as v4-FINAL**
(`models/qwen3-8b-automotive-complaint-lora-v4-FINAL/`).

**Disclosed gap: v4-FINAL was trained on the 38-correction version of `train_v4.jsonl`,
not the 42-correction version that exists now.** After training completed, a forensic
investigation into v4's `safety_risk` recall (`docs/eval-report.md` Section 7) found and
fixed a bug in `text_support_audit.py`'s `HEDGE` pattern (see `docs/label-strategy.md`'s
v5 section) that had been suppressing 4 additional legitimate label corrections in the
training set — 38/900 became 42/900, a 0.44 percentage-point shift (4 rows). **No
retrain was run for this delta.** At 4 rows out of 900 (and given the fix only touches
rows that were already being handled conservatively — left as hedge-ambiguous rather
than mis-corrected — not a systematic error affecting a meaningful fraction of the
dataset), a retrain wasn't judged to be worth the Colab/Kaggle GPU time for a change this
small. This is a disclosed, known gap between the shipped model's actual training data
and the most-current, most-corrected label set — not something discovered later and
smoothed over. If a future round retrains for any other reason, it should train on the
current `train_v4.jsonl` (42 corrections), not the 38-correction snapshot v4-FINAL
actually saw.
