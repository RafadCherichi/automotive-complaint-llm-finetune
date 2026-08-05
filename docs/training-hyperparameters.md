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

## What this notebook cannot do from here

This machine has no way to actually execute the notebook — it needs a live Colab or
Kaggle GPU session, which isn't available in this environment. The notebook is built
and believed correct against documented Unsloth/TRL APIs, but **has not been run**.
Report back any errors hit when actually running it so they can be fixed.

## Actual training result (Kaggle T4, run completed)

| Epoch | Eval loss |
|---|---|
| 1 | 1.0770 |
| **2** | **1.0595 — best, selected** |
| 3 | 1.0683 — regressed |

Epoch 3's eval loss climbing back up while presumably still fitting the training set
further is the exact overfitting signal the notebook's per-epoch eval-loss cell was
built to catch — with only 800 training examples, 3 full passes was already at the edge
of what the data supports, and epoch 3 confirms it: real evidence the small-dataset
overfitting risk flagged when these hyperparameters were locked wasn't hypothetical.
The epoch 2 checkpoint (`checkpoint-100` — 800 examples ÷ effective batch 16 = 50 steps/
epoch, so step 100 is exactly the end of epoch 2) was selected over the final epoch 3
weights specifically because of this.

**Adapter location:** `models/qwen3-8b-automotive-complaint-lora/` (gitignored, not
committed — same as other model-weight artifacts in this repo).

**Completeness check performed on the extracted files** (not just assumed from the
download succeeding):
- `adapter_config.json` matches the locked hyperparameters exactly: `r=16`,
  `lora_alpha=32`, `use_dora=true`, `target_modules` = the 7 modules specified,
  `base_model_name_or_path=unsloth/Qwen3-8B-unsloth-bnb-4bit`.
- `adapter_model.safetensors` (180MB, 756 tensors, ~45M params) parsed with
  `safetensors.safe_open` — not corrupted/truncated. Covers all 36 transformer layers
  and all 7 target modules, with `lora_A`, `lora_B`, **and `lora_magnitude_vector`**
  tensors present for every one — concrete confirmation DoRA actually trained (the
  magnitude vector is DoRA-specific; plain LoRA wouldn't have it), not just configured.
- `trainer_state.json`'s own logged eval losses (epoch 1: 1.0770, epoch 2: 1.0595) match
  what was reported, and `global_step: 100` is exactly the end-of-epoch-2 step count —
  internally consistent with this genuinely being the epoch 2 checkpoint, not a mismatched
  or partial copy.
- Tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`) are
  present and non-trivial size.

**One informational note, not a defect:** the downloaded zip is a raw copy of the
Trainer's `checkpoint-100/` directory, not the notebook's dedicated `save_pretrained()`
export cell — it includes `optimizer.pt`, `scheduler.pt`, `scaler.pt`, and
`rng_state.pth` (~104MB combined), which are training-resumption state, not needed for
inference in Phase 3/4. Harmless to leave in place; only the `adapter_*` and tokenizer
files are actually needed going forward.
