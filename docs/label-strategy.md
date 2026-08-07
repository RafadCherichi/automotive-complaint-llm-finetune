# Label Strategy (Phase 1 output)

How the four target fields — `component`, `defect_type`, `safety_risk`, `severity` —
are derived from NHTSA's own data, with zero manual labeling and zero LLM-generated
labels. See `blueprint.md` Section 5 / 5a for the locked decisions this implements.

## Source data

- **File:** `static.nhtsa.gov/odi/ffdd/cmpl/FLAT_CMPL.zip` — NHTSA's full public
  complaints database, all complaints since 1995. Downloaded whole (352MB zipped /
  1.5GB uncompressed) rather than queried per make/model, specifically to avoid
  selection bias from hand-picking which vehicles to include.
- **Format:** 51 tab-delimited columns, no header row. Column layout confirmed against
  NHTSA's own data dictionary (`static.nhtsa.gov/odi/ffdd/cmpl/CMPL.txt`) —
  see `scripts/nhtsa_schema.py`.
- **Processing:** read one row at a time directly from the zip using Python's built-in
  `csv` module (`scripts/build_dataset.py`), not pandas — required given this machine's
  RAM constraint (8GB total, and free memory was observed as low as ~0.4GB during this
  build, well under the "often ~2GB free" baseline in the blueprint). An earlier version
  used `pandas.read_csv(..., chunksize=...)`, which worked at first but hit a hard
  `MemoryError` on a later rerun once free RAM dropped further — pandas' C parser
  buffers more aggressively per chunk than a single-pass row read needs. Since the file
  is strictly tab-delimited with no real quoting, plain `csv.reader` line-by-line is a
  lighter-weight fit and has no chunk-size tuning to get wrong. One full pass over the
  file: 2,231,883 total rows, 2,159,966 of them vehicle complaints (`PROD_TYPE == "V"`),
  2,098,018 with a narrative long enough to be useful (≥40 characters after stripping).
- **Complaint identity:** NHTSA assigns one row per *(complaint, component)* pair — a
  single complaint that lists three components becomes three rows sharing the same
  `ODINO`. We treat `ODINO` as the true complaint identity and merge same-`ODINO` rows
  into one training example (see "Multi-component complaints" below).

## Field derivation

### `safety_risk` and `severity`

Directly from NHTSA's own flags — `CRASH`, `FIRE`, `INJURED`, `DEATHS` — no judgment
calls:

```
safety_risk = "yes" if crash or fire or injured>0 or deaths>0 else "no"

severity = "high"   if injured>0 or deaths>0
         = "medium" if crash or fire (and no injuries/deaths)
         = "low"    otherwise
```

### `component`

NHTSA's `COMPDESC` field is a colon-separated hierarchy path, e.g.
`POWER TRAIN:AUTOMATIC TRANSMISSION` or `SERVICE BRAKES, HYDRAULIC:ANTILOCK/TRACTION
CONTROL`. We take the top-level segment (before the first `:`) as the raw label, then
bucket it into a fixed taxonomy built from **real frequency counts** across all
2.16M vehicle rows (`scripts/tally_components.py`) — not guessed. Top ~18 categories by
volume, near-duplicate raw strings merged (e.g. `ENGINE` + `ENGINE AND ENGINE COOLING`
→ `ENGINE`; `SERVICE BRAKES` + `SERVICE BRAKES, HYDRAULIC` + `PARKING BRAKE` →
`BRAKES`), everything else → `OTHER`. Full mapping in `scripts/component_taxonomy.py`.

### `defect_type`

No NHTSA column maps to this directly — it's derived via deterministic keyword rules
(`scripts/label_rules.py`), matched in two tiers:

1. **Fire and unintended-acceleration** are checked first, against the free-text
   narrative — these are specific enough to be safe to detect from free text (see
   "What went wrong the first time" below for why that specificity matters).
2. **Everything else** is matched primarily against `COMPDESC` (NHTSA's own structured
   field), not the narrative. Narrative-only fallback rules exist for two categories
   with no reliable `COMPDESC` signature (software/infotainment, stalling language).

15 categories: `ENGINE/STALLING/POWER LOSS`, `UNINTENDED ACCELERATION`,
`BRAKE FAILURE`, `STEERING LOSS`, `TRANSMISSION FAILURE`, `SUSPENSION FAILURE`,
`TIRE/WHEEL FAILURE`, `FUEL SYSTEM LEAK`, `FIRE/SMOKE`, `ELECTRICAL FAULT`,
`AIRBAG NON-DEPLOYMENT`, `SEAT BELT FAILURE`, `STRUCTURAL/CORROSION`,
`SOFTWARE/INFOTAINMENT/ADAS`, `OTHER`.

#### What went wrong the first time (and why COMPDESC, not narrative, is primary)

The first version matched keywords against narrative text directly and put `BRAKE
FAILURE` as the single largest bucket (18.7%). Spot-checking real examples showed why:
a complaint about frozen windshield wipers got tagged `BRAKE FAILURE` because the
narrative said "when I took my foot off the brake" — describing what the driver was
doing, not what broke. Words like "brake" and "steering" show up constantly as
incidental scene-setting in complaint narratives regardless of the actual defect, so
free-text keyword matching produces a lot of false positives. Switching the primary
signal to `COMPDESC` (NHTSA's own curated categorization of what the complaint is
about) fixed this — narrative text is now only used for the two categories that need
it (fire/acceleration language) and as a last-resort fallback.

#### Taxonomy gap found during the OTHER-bucket check

Per Section 5a, `OTHER` becoming the largest bucket is a stop-and-flag signal. It did,
at 16.7%, after the COMPDESC fix above. Inspecting what was actually landing in `OTHER`
found two real coverage gaps, not just diverse unmatched text:

- `VEHICLE SPEED CONTROL` complaints (NHTSA's cruise-control component) were almost all
  narratives about unexpected acceleration/deceleration — just not using the literal
  phrase "unintended acceleration" the rule was looking for. Added as a direct
  `COMPDESC` signal for `UNINTENDED ACCELERATION`.
- `POWER TRAIN` complaints (shifting, gear, clutch problems) were falling through
  because the rule only matched the substring `TRANSMISSION`. Added `POWER TRAIN` as a
  second signal for `TRANSMISSION FAILURE`.

After both fixes: `OTHER` dropped to 10.2%, and `FIRE/SMOKE` (13.6%) is the largest
bucket — expected, since the sample is deliberately oversampled for safety-flagged
complaints. Remaining `OTHER` rows are genuinely outside the 15-category scope
(exterior lighting, wipers, seats, latches, and NHTSA's own `UNKNOWN OR OTHER` tag) —
comfort/convenience complaints that don't belong in a safety-defect taxonomy, not a
rule-coverage failure.

#### Bug #2, found by targeted spot-checking after the fact

A stratified ~20-row manual review (not a full re-read — a few examples from each of:
high/low severity, both taxonomy fixes above, multi-component complaints, and the
residual `OTHER` bucket) caught a second real problem: the `VEHICLE SPEED CONTROL` →
`UNINTENDED ACCELERATION` rule from bug #1 was too blunt. NHTSA's cruise-control
component covers malfunctions in *both* directions — unwanted speeding up **and**
unwanted slowing down/stalling — and two of three sampled examples were actually the
latter (e.g. "vehicle would no longer accelerate... could do nothing but coast and
brake"). Fixed by checking the narrative for power-loss language
(`lost power`, `would not accelerate`, `stall`, etc.) first, and only defaulting to
`UNINTENDED ACCELERATION` when that's absent. One residual known limitation: a
complaint describing "acceleration speed went from 70 to 25 mph" is a deceleration
event but doesn't use any of the power-loss phrases the rule looks for — detecting that
would require parsing and comparing the two numbers, which is disproportionate
engineering for a rule-based labeling pass. Left as a documented limitation rather than
"fixed" by adding number-parsing logic.

This is the value of the targeted (not exhaustive) spot-check: both real bugs found in
this project surfaced from a ~20-row stratified sample aimed at the newly-changed and
highest-risk categories, not from reading through the full dataset.

## Multi-component complaints

A complaint filed against multiple components (e.g. `POWER TRAIN,STEERING`) becomes
multiple rows sharing one `ODINO` in the source file. These are merged into a single
training example: the first-listed component is the `component` training target
(bucketed as above), and the **full raw multi-value string is retained** as a
`component_raw` metadata field (not part of the target) — so when the model gets a
multi-defect complaint "wrong" during Phase 3 error analysis, it's traceable to a
documented, honest ambiguity rather than a silent blind spot.

## Class balance

Real-world safety-flagged complaints are rare — 148,338 out of ~2.1M vehicle
complaints, ~7% (higher than the ~3% seen in the single-model exploration sample, now
confirmed against the full corpus). Training a model on that natural ratio risks it
learning to just predict `"no"` / `"low"` and still looking accurate — exactly the
failure mode the project's safety-triage story cares most about avoiding.

Per Section 5a: both the training and eval sets are deliberately stratified to ~32%
`safety_risk: yes` (within the locked 30–35% band) — real complaints only, nothing
fabricated, just weighted sampling toward the rare class. Sampling uses reservoir
sampling over the *entire* file (not filtered to hand-picked makes/models), so there's
no selection bias toward whichever vehicles happened to get queried.

## Phase 3 finding: `high` severity had a within-class blind spot (train.jsonl v2)

The Phase 3 before/after eval (`eval/eval_results_v1.json`, the v1 run) found the
fine-tuned model never predicted `severity: high` correctly — all 18 actual-`high` eval examples landed
on `medium` or `low`. Reading the missed examples showed a specific pattern, not random
noise: the model had learned **"crash/fire mentioned → medium"** as a shortcut, and
under-weighted injury-only language when no crash/fire flag was present — even complaints
with explicit injury text ("suffered heavy bruising," "sustained injuries... required
medical attention") got called `medium`.

The root cause was training-data composition, not the derivation logic: `severity: high`
splits into two real sub-patterns —

- **Sub-pattern A** — `injured>0` or `deaths>0`, but `crash=False` and `fire=False`
  (injury/death with no crash or fire language to key off of)
- **Sub-pattern B** — `injured>0`/`deaths>0` **and** `crash=True` or `fire=True`

In the original `train.jsonl`, the 99 `high` examples split 28 sub-pattern-A : 71
sub-pattern-B (28%) — sub-pattern B's crash/fire language dominated the tier, and the
model generalized from the majority pattern.

**Fix (within-class, not a dataset-wide rebalance):** re-scanned the full flat file for
sub-pattern-A candidates not already used in `train.jsonl` or `eval.jsonl`
(`scripts/find_injury_only_high.py`) and found **16,040 available** — no scarcity
problem, this was purely a sampling-luck issue in the original 800-row draw. Added 57 of
them to `train.jsonl` and removed 57 randomly-selected `medium` examples to compensate
(`scripts/rebuild_train_injury_fix.py`, seed=42), holding the total dataset size (800),
the overall `safety_risk: yes` rate (256, 32.0%), and `low` (544, completely untouched)
exactly as they were. Only the `high` tier's internal composition changed:

| | before | after |
|---|---|---|
| `high` total | 99 | 156 |
| — sub-pattern A (injury-only) | 28 (28.3%) | 85 (54.5%) |
| — sub-pattern B (crash/fire + injury) | 71 (71.7%) | 71 (45.5%) |
| `medium` | 157 | 100 |
| `low` | 544 | 544 (unchanged) |
| `safety_risk: yes` | 256 (32.0%) | 256 (32.0%) |

`eval.jsonl` was not read or written by this fix — confirmed byte-identical
(sha256 unchanged) before and after, so the before/after model comparison in Phase 3
stays apples-to-apples on the exact same 140 examples. The original `train.jsonl` is
preserved at `data/processed/train_v1_backup.jsonl` (gitignored, local only) in case this
needs to be reverted or compared against.

This is documented as a real finding, not smoothed over: the original stratification
(Section 5a) correctly targeted the `safety_risk`-level 97/3 imbalance, but didn't
anticipate that a *second*, subtler imbalance existed one level down, inside the `high`
tier itself. Worth remembering for any future label-strategy work — checking the
top-level class balance isn't sufficient when a field has internally distinct sub-patterns
that a model can shortcut between.

## Phase 3 finding #2: the v2 fix over-corrected (train.jsonl v3 — additive restore)

Retraining on v2's `train.jsonl` fixed the diagnosed problem — `severity: high` accuracy
went from 0% to 88.9% — but the Phase 3 rerun found a new regression: `severity: medium`
accuracy collapsed from 70.4% to 14.8%. The model's "I'm not sure" default shifted from
over-predicting `medium` (v1) to over-predicting `high` (v2), rather than actually
learning to tell the two apart. Full evidence with concrete misclassified examples,
narrative text, and the crash/fire/injury flags for each is in
`docs/eval-report.md` Sections 2-3 — roughly a third looked like genuine overcorrections
(narratives that explicitly said "no injuries" or "1 mph impact" still got called
`high`), and about half looked like defensible near-misses (dramatic language —
structural failure, fire, a child involved — without a recorded injury).

**Root cause, this time: absolute example count, not composition.** The v2 rebuild
*swapped* 57 `medium` examples out to make room for 57 new `high` examples — correct for
fixing the sub-pattern imbalance inside `high`, but it also dropped `medium`'s raw count
from 157 to 100, below the ~200–500-per-class range that's the standard target for
structured-extraction QLoRA fine-tunes on a 7-8B model. A class with too few examples is
prone to exactly this kind of unstable, over-generalized behavior.

**Fix (additive only — this is a restore, not another swap):** re-scanned the full flat
file for `medium`-pattern candidates (`crash=True` or `fire=True`, `injured=0`,
`deaths=0`) not already used anywhere in `train.jsonl` or `eval.jsonl`
(`scripts/grow_medium_tier.py`) and found **85,259 available**. Added 100 of them
straight on top of the existing 800 rows — **nothing was removed or swapped this time**,
unlike the v2 fix. `low` (544) and `high` (156, the tier the previous fix already
repaired) are completely untouched.

| | v2 | v3 (this fix) |
|---|---|---|
| `low` | 544 | 544 (unchanged) |
| `medium` | 100 | **200** |
| `high` | 156 | 156 (unchanged) |
| **total train** | 800 | **900** |
| `safety_risk: yes` | 256 (32.0%) | 356 (39.6%) |

**This deliberately moves outside Section 5a's original 30–35% `safety_risk: yes` band**
(now 39.6%) — a direct, expected consequence of adding only positive-class (`medium`)
examples without removing anything to compensate, as explicitly instructed. This is a
considered tradeoff, not an oversight: the per-class-count target (200–500 per class) is
the more relevant constraint for this specific fix, and the original 30–35% band was
calibrated before this second, class-count-specific problem was known. Still within
Section 6's locked ~500–1,000 total training-pairs range (900).

`eval.jsonl` untouched again — confirmed byte-identical (same sha256 as every prior
round) — so all three fine-tuned models (v1, v2, v3) remain comparable on the exact same
140 examples.

This was the last planned retrain round for this project, as decided going in — and the
result justified stopping. See below for what actually happened.

## Final decision: three rounds compared, v2 shipped (not v3)

v3 trained and evaluated cleanly, but the result argued against itself. Full metrics in
`docs/training-hyperparameters.md`; the data-relevant summary:

| | v1 (800 rows) | v2 (800 rows, swapped) | v3 (900 rows, grown) |
|---|---|---|---|
| `severity: high` accuracy | 0.0% | **88.9%** | 0.0% |
| `severity: medium` accuracy | 70.4% | 14.8% | 59.3% |
| `safety_risk` recall | 77.8% | **88.9%** | 68.9% |
| `safety_risk` precision | 68.6% | 71.4% | **86.1%** |

Growing `medium` from 100 to 200 examples (the v3 fix, aimed at the standard
200–500-per-class QLoRA range) improved `medium`'s own accuracy some (14.8% → 59.3%,
still below v1's 70.4%) but **undid v2's high-severity fix in the process** — `high`
accuracy fell straight back to 0%, the same failure v1 had. v3 does win on blended
accuracy and on `safety_risk` precision, but a model that never correctly identifies a
real high-severity complaint isn't an acceptable safety-triage tool regardless of how
clean its other numbers look.

**v2 shipped.** Its adapter is copied to `models/qwen3-8b-automotive-complaint-lora-FINAL/`
for Phase 4. v1 and v3 adapters are kept locally as superseded experiments, not deleted
— see `models/README.md`.

**Important distinction for anyone reading this later:** the `train.jsonl` currently on
disk holds **v3's** 900-row dataset (the most recent build), *not* the data that
produced the shipped v2 model. The data that actually trained the shipped model is
preserved at `data/processed/train_v2_backup.jsonl`. This is intentional — `train.jsonl`
tracks the latest data-side experiment, while the shipped model is pinned by its own
adapter files, independent of what `train.jsonl` currently contains. If Phase 4 or later
work ever needs to reproduce the shipped model's training data exactly, use
`train_v2_backup.jsonl`, not `train.jsonl`.

**Open question left for any future iteration, not resolved here:** why growing
`medium`'s raw count didn't cleanly fix `medium` accuracy, and why it interfered with
`high` at all, isn't fully understood — the three severity tiers may not be cleanly
separable by example count alone within this rule-derived label scheme. Recorded
honestly as a limitation rather than chased with a fourth round, per the decision to
stop after three.

## Dataset snapshot (v3 — most recent build; NOT what the shipped v2 model was trained on, see above)

| | rows | safety_risk: yes |
|---|---|---|
| `data/processed/train.jsonl` | 900 | 356 (39.6%) |
| `data/processed/eval.jsonl` | 140 | 45 (32.1%) |

No overlap between the two sets (checked by `ODINO`). Backups of every prior version are
kept locally (gitignored): `train_v1_backup.jsonl` (original Phase 1 build) and
`train_v2_backup.jsonl` (after the injury-only-high fix, before the medium restore).

**severity (train, v3):** 60.4% low / 22.2% medium / 17.3% high.
**severity (eval, unchanged throughout):** 67.9% low / 19.3% medium / 12.9% high.

**defect_type** (combined, v3, recomputed after the medium restore): FIRE/SMOKE 15.3%,
BRAKE FAILURE 11.5%, OTHER 10.4%, AIRBAG NON-DEPLOYMENT 9.3%, ELECTRICAL FAULT 8.5%,
TRANSMISSION FAILURE 7.9%, STEERING LOSS 7.4%, ENGINE/STALLING/POWER LOSS 7.1%,
UNINTENDED ACCELERATION 4.8%, FUEL SYSTEM LEAK 4.4%, STRUCTURAL/CORROSION 4.1%,
SUSPENSION FAILURE 3.7%, TIRE/WHEEL FAILURE 2.4%, SOFTWARE/INFOTAINMENT/ADAS 1.6%, SEAT
BELT FAILURE 1.5%. `OTHER` still not the largest bucket (FIRE/SMOKE is, by a wide margin
now that more crash/fire-flagged `medium` examples are in the mix) — no taxonomy-pass
flag triggered.

**component** (combined, v3): ELECTRICAL SYSTEM 16.2%, ENGINE 13.0%, AIR BAGS 12.5%,
BRAKES 8.5%, POWER TRAIN 7.4%, STEERING 6.0%, FUEL SYSTEM 4.9%, VEHICLE SPEED CONTROL
4.4%, STRUCTURE 4.4%, OTHER 3.8%, SUSPENSION 3.0%, EXTERIOR LIGHTING 2.6%, TIRES/WHEELS
2.4%, ADAS/DRIVER ASSIST 2.4%, SEATS 2.2%, VISIBILITY 2.1%, SEAT BELTS 1.5%, EQUIPMENT
1.5%, LATCHES/LOCKS/LINKAGES 1.1%.

## Record schema (`data/processed/{train,eval}.jsonl`)

```json
{
  "odino": "...",             // NHTSA complaint id
  "cmplid": "...",            // NHTSA row id (first row for this complaint)
  "make": "TOYOTA", "model": "TACOMA", "year": "2001",
  "narrative": "...",         // raw complaint text (CDESCR)
  "component": "SUSPENSION",       // training target
  "component_raw": "SUSPENSION",   // metadata: full raw multi-value string
  "defect_type": "SUSPENSION FAILURE",  // training target
  "safety_risk": "no",             // training target
  "severity": "low",               // training target
  "crash": false, "fire": false, "injured": 0, "deaths": 0  // metadata
}
```

Only `component`, `defect_type`, `safety_risk`, `severity` are the model's target —
everything else is metadata carried through for the demo, error analysis, and eval
harness.

## Known limitations (for the Phase 3 error-analysis writeup)

- **Multi-component complaints** are reduced to a single `component` label
  (first-listed). The raw multi-value string is kept as metadata precisely so these
  cases are traceable, not hidden.
- **Rule-based `defect_type` is coarse.** E.g. a dashboard-melting/heat complaint gets
  tagged `FIRE/SMOKE` via a "burns my fingers" keyword match — defensible, but not
  perfectly precise. A complaint spanning both `STEERING` and `VEHICLE SPEED CONTROL`
  components resolves to whichever category's keyword rule runs first, which can pick
  the less-central defect for that specific complaint.
- **No time filtering.** The corpus spans 1995–2026; some older complaints reference
  vehicle systems less relevant to modern cars (e.g. carburetors). Left in deliberately
  to avoid reintroducing manual-selection bias, per Section 5a.
- **Numeric-direction ambiguity in speed-control complaints.** A narrative like
  "acceleration went from 70 mph to 25 mph" describes a deceleration event, but doesn't
  contain any of the power-loss phrases the rule checks for, so it can fall through to
  `UNINTENDED ACCELERATION` by default. Reliably catching this would require parsing
  and comparing the two numbers in the sentence — out of scope for a keyword-rule
  system. Found during the Phase 1 stratified spot-check; documented rather than fixed.
- **`severity` tiers may not be cleanly separable by training-example count alone.**
  Growing `medium` from 100 to 200 examples (the v3 fix) improved `medium` accuracy some
  but unexpectedly regressed `high` accuracy back to 0% — the same failure the v2 fix
  had solved. Why adding more `medium` data interferes with the model's `high`-tier
  behavior isn't understood; this shipped as an open question rather than a fourth
  retrain round. See "Final decision" above for the full three-round comparison.
