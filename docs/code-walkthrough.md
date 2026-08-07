# Code Walkthrough

A guided tour through the actual pipeline, in the order it runs: raw NHTSA data → label
derivation → training set → trained model → evaluation → analysis. Every section points
to the real file and function — this is meant to be read next to the code, not instead
of it.

For *why* each piece works the way it does, see `docs/learning/` (the concepts, one file
per decision) and `docs/label-strategy.md` / `docs/training-hyperparameters.md` (the
full decision history, including bugs found and fixed along the way).

## 1. Data pipeline (`scripts/`) — runs locally, no GPU needed

```
scripts/nhtsa_schema.py         column layout for NHTSA's flat file (51 tab-delimited
                                 columns, no header row — this file just documents which
                                 column index means what)

scripts/label_rules.py          the label derivation logic:
                                   safety_risk(crash, fire, injured, deaths) -> "yes"/"no"
                                   severity(crash, fire, injured, deaths) -> "low"/"medium"/"high"
                                   defect_type(compdesc, narrative, fire_flag) -> one of 15 categories

scripts/component_taxonomy.py   bucket_component(raw_top_level) -> one of ~18 categories,
                                 built from real frequency counts (scripts/tally_components.py)

scripts/build_dataset.py        the main pipeline: streams the full NHTSA flat file
                                 (data/raw/FLAT_CMPL.zip) row by row with Python's csv
                                 module, applies label_rules.py + component_taxonomy.py
                                 to every row, reservoir-samples a stratified subset, and
                                 writes data/processed/train.jsonl + eval.jsonl

scripts/find_injury_only_high.py   targeted fix #1: scans the full flat file for the
                                    specific under-represented sub-pattern (injury/death,
                                    no crash/fire) and rebuilds train.jsonl with better
                                    representation inside the "high" tier

scripts/grow_medium_tier.py        targeted fix #2: adds more "medium"-pattern examples
                                    on top of the existing training set (purely additive,
                                    nothing removed)

scripts/check_seq_lengths.py       one-time check: measures real token length of every
                                    training example against Qwen3's actual tokenizer,
                                    to pick max_sequence_length with evidence, not a guess
```

**How a single row becomes a training example** (the core logic, `scripts/build_dataset.py`):
1. Read one tab-delimited row from the flat file.
2. Skip if it's not a vehicle complaint (`PROD_TYPE != "V"`) or the narrative is too
   short to be useful (`< 40` characters).
3. Compute `crash`, `fire`, `injured`, `deaths` from the row's own columns.
4. Call `label_rules.safety_risk(...)` and `label_rules.severity(...)` — pure functions,
   same input always gives the same output, no randomness, no model involved.
5. Call `component_taxonomy.bucket_component(...)` on the first-listed component, and
   `label_rules.defect_type(...)` on the component + narrative text.
6. Reservoir-sample into a `positives` or `negatives` pool (keyed by `ODINO`, NHTSA's
   per-complaint ID) to keep the class balance from Section 5a without loading the
   entire multi-million-row file into memory at once.

## 2. Training (`notebooks/train_qlora_dora.ipynb`) — runs on Colab/Kaggle GPU

Cell-by-cell, in order:

| Cell | What it does |
|---|---|
| 4 | Loads Qwen3-8B in 4-bit: `FastLanguageModel.from_pretrained(model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit", load_in_4bit=True, ...)` — see `docs/learning/01_qlora_dora_vs_alternatives.md` for what 4-bit loading buys us |
| 6 | Attaches the LoRA + DoRA adapter: `FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32, use_dora=True, target_modules=[...])` — see `docs/learning/01_qlora_dora_vs_alternatives.md` |
| 8 | Loads `train.jsonl`/`eval.jsonl` (auto-detects an attached Kaggle Dataset by filename, or prompts a Colab upload) |
| 10 | Formats each row into the training text: `SYSTEM_PROMPT` + the complaint narrative + the target JSON, run through `tokenizer.apply_chat_template(..., enable_thinking=False)` |
| 12 | Trains: `SFTTrainer` with `SFTConfig(learning_rate=2e-4, num_train_epochs=3, per_device_train_batch_size=4, gradient_accumulation_steps=4, eval_strategy="epoch", ...)` — see `docs/learning/05_hyperparameter_choices.md` for why each number |
| 14 | Overfitting check: prints train/eval loss per epoch, flags if eval loss increased on the last epoch |
| 16 | Saves the adapter with a timestamped folder name (never overwrites a prior run), zips it for download |

Run three times over the project (v1, v2, v3) as the Phase 3 diagnosis process found and
fixed real problems in the training data — same notebook, same locked hyperparameters
every time, only `train.jsonl`'s contents changed between rounds.

## 3. Evaluation (`notebooks/eval_baseline_vs_finetuned.ipynb`) — runs on Colab/Kaggle GPU

| Cell | What it does |
|---|---|
| 5 | `SYSTEM_PROMPT` copied verbatim from the training notebook, and `parse_json_output()` — extracts and validates a JSON object from raw model output, returns `None` (never crashes) if it doesn't parse |
| 7 | `run_eval(model, tokenizer, rows, label)` — the shared generation loop, greedy decoding (`do_sample=False`) for reproducibility, used for both models |
| 9 | Runs the **base model**, no adapter attached |
| 11 | Frees the base model from memory before loading the fine-tuned one (both are 8B — loading both at once would double VRAM use for no reason) |
| 13 | Runs the **fine-tuned model** — loads the adapter directly via `FastLanguageModel.from_pretrained(model_name=ADAPTER_DIR, ...)`, auto-detecting whichever Kaggle Dataset contains `adapter_config.json` |
| 16 | `compute_metrics(results, rows)` — accuracy per field, `safety_risk=yes` precision/recall, and the full severity confusion matrix (not just one blended number — see `docs/eval-report.md` for why this mattered) |
| 19 | `guess_why(row, pred)` — a data-grounded heuristic that explains *why* a failure likely happened, using the row's own metadata (multi-component strings, crash/fire/injury flags), not a fabricated explanation |
| 21 | Saves everything — full metrics, every prediction, failure examples — to `eval_results.json` |

Also run three times (once per training round), each time producing
`eval/eval_results_v1.json`, `eval_results_v2.json`, `eval_results_v3.json` — all
graded on the exact same 140-example `eval.jsonl`, confirmed byte-identical across every
round, so the three results are directly comparable.

## 4. Post-eval analysis (`scripts/`) — runs locally, no GPU, no retraining

```
scripts/boundary_review_analysis.py     tests the first proposed human-review-trigger
                                         rule against v2's actual saved predictions

scripts/boundary_review_analysis_v2.py  tests a broader version of the same rule, after
                                         the first one turned out to catch 0% of the
                                         cases it was meant to catch
```

Both scripts do the same kind of thing: load `eval/eval_results_v2.json` (already-saved
predictions, no model needed), apply a candidate rule (e.g. "flag for review if
predicted `severity=medium` and predicted `safety_risk=yes`"), and measure two numbers
against the real held-out set — how many real high-severity cases the rule would catch,
and how many complaints total get flagged (the review workload cost). This is how
`docs/eval-report.md` Section 4's production-mitigation recommendation ended up
*measured* against real predictions instead of just proposed and assumed to work.

## Where the four target fields actually get produced, end to end

```
NHTSA row (CRASH, FIRE, INJURED, DEATHS, COMPDESC columns)
    |
    v  scripts/label_rules.py + scripts/component_taxonomy.py
data/processed/train.jsonl  { component, defect_type, safety_risk, severity, ... }
    |
    v  notebooks/train_qlora_dora.ipynb
models/qwen3-8b-automotive-complaint-lora-FINAL/  (trained adapter)
    |
    v  notebooks/eval_baseline_vs_finetuned.ipynb
eval/eval_results_v2.json  (model's own predictions of the same 4 fields, graded)
    |
    v  scripts/boundary_review_analysis*.py
docs/eval-report.md Section 4  (measured production-mitigation recommendation)
```
