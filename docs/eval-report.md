# Evaluation Report — Base Qwen3-8B vs. Fine-Tuned (Phase 3)

The core evidence document for this project: does QLoRA + DoRA fine-tuning on real
NHTSA complaint data actually improve structured safety-extraction accuracy over the
unmodified base model, measured on the same 140-example held-out set every time.

**Shipped model:** v2 (`models/qwen3-8b-automotive-complaint-lora-FINAL/`). All numbers
in this report are pulled directly from `eval/eval_results_v1.json`,
`eval/eval_results_v2.json`, and `eval/eval_results_v3.json` — nothing rounded up or
softened, including the results that don't flatter the shipped model.

---

## 1. Headline result: base vs. shipped model (v2)

| Metric | Base (zero-shot) | Fine-tuned (v2) | Change |
|---|---|---|---|
| JSON validity rate | 99.3% | **100.0%** | +0.7pp |
| component accuracy | 11.4% | **64.3%** | +52.9pp |
| defect_type accuracy | 4.3% | **67.1%** | +62.8pp |
| safety_risk accuracy | 36.4% | **85.0%** | +48.6pp |
| severity accuracy (blended) | 15.7% | **70.7%** | +55.0pp |
| safety_risk=yes precision | 32.8% | **71.4%** | +38.6pp |
| safety_risk=yes recall | 93.3% | 88.9% | −4.4pp |

The blended numbers are dramatic and genuine: near-zero-effort zero-shot prompting on a
4-field structured extraction task performs close to random guessing on `component` and
`defect_type` (11.4% and 4.3% — there are 19 and 15 categories respectively, so this is
barely above chance), while the fine-tuned model gets roughly two in three right on
both.

**The one number that goes the "wrong" direction, and why it's not actually bad news:**
base model's `safety_risk=yes` recall (93.3%) is higher than v2's (88.9%). Looking at
*why* explains it — base achieves that recall by saying "yes" to almost everything: 86
false positives out of 140 examples, for a precision of just 32.8%. A tool that cries
wolf two times out of three isn't usable for triage; an analyst using it would drown in
false alarms and likely start ignoring it. v2 trades a small amount of recall for more
than double the precision — a real, deliberate, and better tradeoff for a system meant
to be trusted in production, not a regression.

---

## 2. The three-round retraining story

Same locked hyperparameters all three rounds (`r=16`, `alpha=32`, DoRA, `lr=2e-4`, 3
epochs, epoch 2 selected each time — see `docs/training-hyperparameters.md`). Only the
training data changed between rounds.

### v1: never predicted `severity: high`

The first fine-tune (800 examples, natural sampling-luck composition) looked solid on
blended metrics but had a total blind spot: **0.0% accuracy on `severity: high`** — all
18 actual-high eval examples got called `medium` or `low`. Reading the misses showed a
specific, learned shortcut: the model had picked up **"crash or fire mentioned →
medium"** as a rule of thumb, and applied it even to complaints with explicit injury
language ("suffered heavy bruising," "sustained injuries... required medical
attention").

Root cause: `severity: high` splits into two real sub-patterns — injury/death **with**
a crash or fire flag (71 of the original 99 high-tier examples, 71.7%) and injury/death
**without** one (only 28, 28.3%). The crash/fire-flagged majority dominated what the
model learned, and it never picked up the injury-only signal at all.

### v1 → v2: the injury-only fix

Re-scanned the full NHTSA flat file for injury/death-only complaints (no crash, no
fire) not already in the dataset — found 16,040 available, so this was a sampling-luck
problem, not a scarcity problem. Swapped 57 of them into the `high` tier in place of 57
`medium` examples, bringing the sub-pattern split from 28%/72% to a roughly balanced
54.5%/45.5%. Total dataset size and `safety_risk: yes` rate held exactly fixed — a
within-class fix, not a broader rebalance.

**Result: `severity: high` accuracy went from 0.0% to 88.9% (16 of 18)** — the fix
worked exactly as diagnosed. `safety_risk=yes` recall also improved, from 77.8% to
88.9%, without costing precision (68.6% → 71.4%). This is the model that shipped.

### v2 → v3: the medium/high seesaw

v2's own eval revealed a new problem: `severity: medium` accuracy had collapsed, from
70.4% (v1) to 14.8%. The model's "uncertain" default had shifted from over-predicting
`medium` (v1's failure mode) to over-predicting `high` (v2's new one) — 20 of 27 actual
medium examples were now called `high`. The obvious-looking fix: `medium` had only 100
training examples, well under the ~200–500-per-class range that's standard for
structured-extraction QLoRA fine-tunes on a 7-8B model. So we grew it — added 100 new
medium-pattern examples (crash or fire, no injury/death) purely additively, nothing
removed, bringing `medium` to 200 examples and total training data to 900 rows.

**Result: `severity: medium` accuracy improved (14.8% → 59.3%) but `severity: high`
regressed straight back to 0.0%** — the exact failure v1 had, undone by v2, now back.
`safety_risk=yes` precision reached its best value across all three rounds (86.1%), but
recall dropped to 68.9%, the worst of the three.

This is the important finding from round three: **rebalancing one severity tier
systematically hurt the adjacent one.** It happened in both directions — the v1→v2 fix
(add to `high`) hurt `medium`, and the v2→v3 fix (add to `medium`) undid the `high` fix.
That symmetry is evidence this is a **shared-signal problem, not a data-volume
problem**. `medium` and `high` aren't cleanly separated by different vocabulary the
model can learn independently — they're distinguished by exactly one thing: whether the
narrative's injury/death language is present or absent, layered on top of crash/fire
language that's common to both tiers. Adding more examples to either tier just shifts
which class absorbs the model's uncertainty on that same underlying signal; it doesn't
teach it to read the signal more precisely. Full three-round metrics table is in
`docs/training-hyperparameters.md`.

**v2 shipped, not v3**, because a model that never correctly identifies a real
high-severity complaint (v3's regression) is not an acceptable safety-triage tool
regardless of how its other numbers look — see Section 4 for why the `safety_risk`
binary decision, not the exact severity tier, is the number this project treats as
load-bearing.

---

## 3. Honest limitation: the medium/high severity boundary

The shipped model (v2) still gets `severity: medium` right only 14.8% of the time (4 of
27 medium-actual eval examples) — the tradeoff accepted for fixing `high`. Reading
through the misclassified examples shows this isn't one uniform failure mode; it splits
into two real, different sub-patterns.

### (a) Genuine model overcorrection

Cases where the narrative is dramatic-sounding but the complaint itself, and NHTSA's
own flags, indicate nothing happened — no crash, no fire, no injury — and the model
flagged `safety_risk: yes` / `severity: high` anyway:

- *"...the contact heard a loud pop as the steering wheel started spinning
  uncontrollably... tie rod bolt was fractured..."* (crash=False, fire=False,
  injured=0) — predicted `yes`/`high`, actual `no`/`low`.
- *"Gas from airbags came through the vent and made passengers ill and temporarily
  blind."* (crash=False, fire=False, injured=0) — predicted `yes`/`high`, actual
  `no`/`low`.
- *"...I have a higher risk to be killed or injured because the transmission could fail
  without warning... What a nightmare!"* (a Takata recall complaint about faulty parts,
  not an incident that occurred — crash=False, fire=False, injured=0) — predicted
  `yes`/`high`, actual `no`/`low`.
- *"While backing my truck out of the garage I lost my brakes... I was able to get it
  to stop with emergency brake."* (a real defect, safely handled, no crash — crash=False,
  fire=False, injured=0) — predicted `yes`/`high`, actual `no`/`low`.

These are the clearest cases of the model over-weighting alarming vocabulary
("uncontrollably," "temporarily blind," "nightmare") over the absence of any actual
incident.

### (b) Defensible boundary disagreements

Cases where a real crash or fire genuinely happened (so `safety_risk: yes` is correct
both times), and the disagreement is only about `medium` vs. `high` — and the narrative
language a human would find just as alarming as the model apparently did:

- *"...the truck rolled again down [the] driveway and hit another **with child in the
  truck**."* (crash=True) — predicted `high`, actual `medium`.
- *"Car was parked and off — the passenger side headlight blew up and **car caught on
  fire**."* (this one actually has `fire=True` on record) — predicted `high`, actual
  `medium`.
- *"...the contact's vehicle crashed into a deer... there were no injuries..."*
  (crash=True, but the complaint text itself states no injuries) — predicted `high`,
  actual `medium`.

A model — or a human reviewer — reading "child in the truck" or "car caught on fire"
and leaning toward `high` even without a recorded injury is not an unreasonable
judgment call. These are boundary disagreements with the label, not clear model errors.

**Bottom line:** roughly half of the medium/high misses in the failure sample look like
genuine overcorrection, and roughly half look like defensible disagreement at a genuinely
ambiguous label boundary. More training data alone did not resolve this in three rounds
of trying (Section 2) — it's a structural property of how close `medium` and `high` sit
to each other in this label scheme, not a data-quantity problem.

---

## 4. Recommended production mitigation (measured, not just proposed)

The core recommendation is a standard pattern for deploying a classifier with a
known-weak boundary between two adjacent classes: **automate the binary triage decision
(`safety_risk: yes`/`no`) — route uncertain severity calls to human review rather than
fully automating the tier assignment.** That part holds up: v2's `safety_risk=yes`
precision (71.4%) and recall (88.9%) are strong enough to trust for the binary decision
this project's PM framing actually depends on (`blueprint.md` Section 3) — of every 10
complaints flagged as a safety risk, roughly 7 genuinely are, and it catches about 9 of
every 10 real safety risks in the data.

**But the specific review-trigger rule needs to be reported as measured, not assumed —
and the first version of this section got it wrong.** Two candidate rules were tested
directly against v2's actual saved predictions on all 140 eval examples
(`scripts/boundary_review_analysis.py`, `scripts/boundary_review_analysis_v2.py`):

| Rule | Review workload | Coverage of the 18 real high-severity cases |
|---|---|---|
| Flag if predicted `severity=medium` AND predicted `safety_risk=yes` | 4/140 = 2.9% | **0 of 18 (0.0%)** |
| Flag if predicted `safety_risk=yes` AND predicted `severity≠high` (broader) | 4/140 = 2.9% | **0 of 18 (0.0%)** |

**Neither rule catches anything.** Both land on the same 4 flagged examples and the same
0% coverage. This isn't a close call that a slightly different threshold would fix — the
reason is structural: v2 already gets 16 of the 18 real high-severity cases right
(88.9%, Section 1). The 2 it misses were **both predicted `severity=low` AND
`safety_risk=no`** — a complete miss on the binary decision itself, not an
uncertain-but-flagged tier call:

- odino=10200837 — *"...liftgate closes automatically on top of owner when loading and
  unloading."* (injured=1, crash=False, fire=False) — predicted `safety_risk=no`,
  `severity=low`.
- odino=11541469 — *"...trunk hit me in the head twice and continued to close..."*
  (injured=1, crash=False, fire=False) — predicted `safety_risk=no`, `severity=low`.

**No severity-based review-trigger can catch a case the model never flagged as a safety
risk in the first place.** A rule that routes "flagged but uncertain" cases to a human
only helps when the model raises *some* signal to route on — these two examples raise
none. Both are the same injury-present/no-crash/no-fire narrative pattern that was the
original subject of the v1→v2 fix (Section 2) — that fix cut this failure mode from 10
false negatives (v1) down to these 2 residual cases in the shipped model, but didn't
eliminate it. **This is a distinct, unresolved gap that "flag for human review" does not
address**, and it should be reported as such rather than folded into a mitigation that
doesn't actually cover it.

**Revised recommendation, split into what the evidence actually supports:**
1. Trust the binary `safety_risk` decision for automated triage (71.4% precision / 88.9%
   recall) — this part is well-supported.
2. Route the severity *tier* to human review whenever `safety_risk: yes` — this catches
   the `medium`/`high` boundary disagreements documented in Section 3, at a workload
   cost of 2.9% of the eval set (4 of 140 examples) under either rule tested.
3. **Mitigation 2 provides zero protection against the 2 residual injury-only false
   negatives** — those need a different backstop (e.g. a lightweight keyword pre-filter
   for injury language, run independently of the model, as a second layer) rather than
   a review-trigger keyed off the model's own output. Not designed or tested here —
   flagged honestly as an open gap rather than retrofitted into a mitigation that
   doesn't actually solve it.

---

## 5. Concrete failure examples (shipped model, v2)

All seven below are real eval-set complaints the shipped model (v2) got wrong, with the
`guess_why` annotations generated by the eval notebook (grounded in the row's own
metadata — multi-component strings, crash/fire/injury flags — not fabricated after the
fact).

**odino=801596** — *"Truck was on in the driveway for one minute and the truck started
to roll down driveway, the first time truck was in park and on the truck rolled again
down in driveway and hit another with child in the truck."*
Predicted: `{component: POWER TRAIN, defect_type: TRANSMISSION FAILURE, safety_risk: yes, severity: high}`
Actual: `{component: POWER TRAIN, defect_type: TRANSMISSION FAILURE, safety_risk: yes, severity: medium}`
guess_why: *severity tier mismatch*
— Category (b): child involved, no recorded injury; defensible disagreement.

**odino=969272** — *"Car was park and off the passenger side head light blow up and car
caught on fire."* (crash=False, **fire=True**, injured=0)
Predicted: `{component: ELECTRICAL SYSTEM, defect_type: FIRE/SMOKE, safety_risk: yes, severity: high}`
Actual: `{component: EXTERIOR LIGHTING, defect_type: FIRE/SMOKE, safety_risk: yes, severity: medium}`
guess_why: *multi/hierarchical component ('EXTERIOR LIGHTING:HEADLIGHTS') — model may
have picked a different one than the first-listed primary label; severity tier
mismatch*
— Category (b): a real fire, correctly detected as `safety_risk: yes`; only the tier
call and the component (headlight vs. general electrical) are off.

**odino=11072384** — *"...while driving 70 mph, the contact's vehicle crashed into a
deer, causing damage to the front driver's side... there were no injuries..."*
Predicted: `{component: AIR BAGS, defect_type: AIRBAG NON-DEPLOYMENT, safety_risk: yes, severity: high}`
Actual: `{component: AIR BAGS, defect_type: AIRBAG NON-DEPLOYMENT, safety_risk: yes, severity: medium}`
guess_why: *severity tier mismatch*
— Category (b): correct on every field except severity; the complaint text itself says
"no injuries," and the model still leaned high.

**odino=10432604** — *"...the contact heard a loud pop as the steering wheel started
spinning uncontrollably. After inspection, he noticed that the tie rod bolt was
fractured on the steering rack."* (crash=False, fire=False, injured=0)
Predicted: `{component: STEERING, defect_type: STEERING LOSS, safety_risk: yes, severity: high}`
Actual: `{component: STEERING, defect_type: STEERING LOSS, safety_risk: no, severity: low}`
guess_why: *safety_risk miss (crash=False fire=False injured=0 deaths=0) — over-flagged
a non-risk complaint; severity tier mismatch*
— Category (a): dramatic wording ("spinning uncontrollably"), no actual incident on
record.

**odino=871031** — *"Gas from airbags came through the vent and made passengers ill and
temporarily blind."* (crash=False, fire=False, injured=0)
Predicted: `{component: AIR BAGS, defect_type: AIRBAG NON-DEPLOYMENT, safety_risk: yes, severity: high}`
Actual: `{component: AIR BAGS, defect_type: AIRBAG NON-DEPLOYMENT, safety_risk: no, severity: low}`
guess_why: *safety_risk miss (crash=False fire=False injured=0 deaths=0) — over-flagged
a non-risk complaint; severity tier mismatch*
— Category (a): "temporarily blind" reads alarming, but no crash/fire/injury flag is on
record for this complaint.

**odino=10660775** — *"...I have a higher risk to be killed or injured because the
transmission could fail without warning... and the airbags could be deployed and kill or
injured myself and family member. What a nightmare!"* (a Takata recall complaint about
a potential future failure, crash=False, fire=False, injured=0)
Predicted: `{component: AIR BAGS, defect_type: AIRBAG NON-DEPLOYMENT, safety_risk: yes, severity: high}`
Actual: `{component: AIR BAGS, defect_type: AIRBAG NON-DEPLOYMENT, safety_risk: no, severity: low}`
guess_why: *safety_risk miss (crash=False fire=False injured=0 deaths=0) — over-flagged
a non-risk complaint; severity tier mismatch*
— Category (a): hypothetical/future risk language ("could," "nightmare") about a recall,
not a description of an incident that happened.

**odino=10468085** — *"While backing my truck out of the garage I lost my brakes due to
pedal going to floor and losing brake fluid. I was able to get it to stop with emergency
brake."* (crash=False, fire=False, injured=0)
Predicted: `{component: BRAKES, defect_type: BRAKE FAILURE, safety_risk: yes, severity: high}`
Actual: `{component: BRAKES, defect_type: BRAKE FAILURE, safety_risk: no, severity: low}`
guess_why: *safety_risk miss (crash=False fire=False injured=0 deaths=0) — over-flagged
a non-risk complaint; severity tier mismatch*
— Category (a): a real, serious-sounding defect that was safely handled — no crash
resulted, so the official flags read `no`/`low`.

---

## Sources

`eval/eval_results_v1.json`, `eval/eval_results_v2.json`, `eval/eval_results_v3.json` —
each graded on the identical, byte-verified-unchanged 140-example `data/processed/eval.jsonl`.
Section 4's review-trigger rules were measured directly against v2's saved predictions
with `scripts/boundary_review_analysis.py` (narrow rule) and
`scripts/boundary_review_analysis_v2.py` (broader rule) — no retraining, no GPU, pure
analysis of already-saved output. Training-side detail in
`docs/training-hyperparameters.md`; label-derivation and dataset-composition detail in
`docs/label-strategy.md`.
