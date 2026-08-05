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

## Final dataset

| | rows | safety_risk: yes |
|---|---|---|
| `data/processed/train.jsonl` | 800 | 256 (32.0%) |
| `data/processed/eval.jsonl` | 140 | 45 (32.1%) |

No overlap between the two sets (checked by `ODINO`).

**severity:** 68.0% low / 19.6% medium / 12.4% high (combined).

**defect_type** (combined, final, after both bug fixes): FIRE/SMOKE 13.6%, BRAKE
FAILURE 11.6%, OTHER 10.2%, AIRBAG NON-DEPLOYMENT 9.6%, ELECTRICAL FAULT 9.0%,
TRANSMISSION FAILURE 8.2%, STEERING LOSS 8.0%, ENGINE/STALLING/POWER LOSS 7.4%, FUEL
SYSTEM LEAK 4.8%, UNINTENDED ACCELERATION 4.8%, STRUCTURAL/CORROSION 3.8%, SUSPENSION
FAILURE 3.4%, TIRE/WHEEL FAILURE 2.3%, SOFTWARE/INFOTAINMENT/ADAS 1.8%, SEAT BELT
FAILURE 1.4%.

**component** (combined): ELECTRICAL SYSTEM 15.9%, ENGINE 13.1%, AIR BAGS 12.9%,
BRAKES 8.0%, POWER TRAIN 7.1%, STEERING 6.3%, FUEL SYSTEM 5.5%, STRUCTURE 4.4%,
VEHICLE SPEED CONTROL 4.3%, OTHER 3.5%, EXTERIOR LIGHTING 3.1%, SUSPENSION 2.8%,
TIRES/WHEELS 2.6%, ADAS/DRIVER ASSIST 2.6%, SEATS 2.1%, VISIBILITY 2.1%, EQUIPMENT
1.8%, SEAT BELTS 1.4%, LATCHES/LOCKS/LINKAGES 0.7%.

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
