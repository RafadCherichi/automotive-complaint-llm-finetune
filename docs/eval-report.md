# Evaluation Report — Base Qwen3-8B vs. Fine-Tuned (Phase 3)

The core evidence document for this project: does QLoRA + DoRA fine-tuning on real
NHTSA complaint data actually improve structured safety-extraction accuracy over the
unmodified base model, measured on the same 140-example held-out set every time.

**Current shipped model: v4** (`models/qwen3-8b-automotive-complaint-lora-v4-FINAL/`,
Round 4 — see Section 7). Sections 1-6 below are the original three-round investigation
that shipped **v2** (`models/qwen3-8b-automotive-complaint-lora-FINAL/`) — left as
written, an accurate record of that investigation at the time, not rewritten in light of
Round 4. All numbers in this report are pulled directly from `eval/eval_results_v1.json`
through `eval/eval_results_v4_epoch2.json` — nothing rounded up or softened, including
results that don't flatter whichever model was shipped at the time.

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

## 6. Text-support audit: how much of the measured error is label noise vs. real model error

Sections 2 and 3 documented the `severity: medium`/`high` seesaw across three training
rounds, purely from retraining outcomes. A separate follow-up question: since
`safety_risk`/`severity` labels are derived deterministically from NHTSA's *structured*
flags (`CRASH`/`FIRE`/`INJURED`/`DEATHS` — see `docs/label-strategy.md`), not from the
narrative text the model actually sees, could some of the measured error be the model
correctly reading text that simply doesn't support its own label, rather than the model
being wrong?

**Methodology:** a deterministic, fully-inspectable keyword detector (4 alarm categories
— injury, crash, fire, control-loss — plus negation handling and a separate "hedge"
flag for hypothetical/recall language), built and hand-validated against two rounds of a
stratified, human-checked sample before being run at scale. Deliberately *not*
LLM-based — using a model to judge whether text supports a label would be circular (see
`docs/learning/04_label_derivation_from_existing_flags.md`). Every pattern is in
`scripts/text_support_audit.py`; the full run is `scripts/full_text_support_audit.py`.
Measurement only — no retraining, no changes to `train.jsonl` or `eval.jsonl`.

### safety_risk: real, measurable label noise

Cross-referencing the detector against v2's actual wrong predictions
(`eval/eval_results_v2.json`):

| | count | % of wrong predictions |
|---|---|---|
| model wrong, but label contradicted by narrative text | 12 / 21 | **57.1%** |
| model wrong, text ambiguous (hedge-only) | 3 / 21 | 14.3% |
| model wrong, label text-supported (genuine model error) | 6 / 21 | 28.6% |

Across the full eval set, `safety_risk: no` labels are the weaker side: only 52.6% of
"no" rows (50/95) have narrative text with no alarm language at all; 21.1% (20/95)
describe something a reader would flag as concerning anyway; 26.3% (25/95) are ambiguous
(hedge words like "could"/"recall"/"nearly" with no other signal). `safety_risk: yes`
labels are much better supported — 77.8% (eval, 35/45) and 84.6% (train, 301/356). The
same pattern holds on the 900-row training set, so it isn't a small-sample artifact.

**Conclusion: v2's measured 85.0% `safety_risk` accuracy likely understates the model's
real skill.** More than half of its "wrong" predictions are on rows where the label
itself doesn't match what the narrative describes — the model may be reading the text
correctly and disagreeing with a label a human would also question.

### severity medium/high confusion: confirmed NOT a labeling artifact

This is the more important result, because it tests the seesaw from Section 2/3 directly
— and it points the opposite way. Two independent checks, both against the 20 eval rows
where the actual label is `medium` and v2 predicted `high`:

1. **Automated keyword check:** only 1 of the 20 had any INJURY-category hit in the
   text — and that one case ("...seat fell backward, **could have caused injuries**...")
   is a hedge/hypothetical, not a real injury. Effectively 0/20.
2. **Full manual read, ignoring the lexicon entirely** — every one of the 20 narratives
   read directly, judged for *any* wording describing an actual injury: **0/20.**
   - 3 rows explicitly state no injury occurred (*"THERE WERE NO INJURIES"*, ×2 more
     with the same wording).
   - 2 rows describe a near-miss or hypothetical, not a real injury (*"could have
     caused injuries"*, *"quick thinking... so no lives were lost"*).
   - 15 rows never mention an injury at all — real crash/mechanical-failure narratives
     (hit a pole, hit a fence, brake failure, tire-tread separation, airbag
     non-deployment) with no injury outcome stated.

**Conclusion: the `medium`/`high` seesaw is a genuine, unresolved model limitation, not
something more data or relabeling would fix.** The model over-predicts `high` on rows
that are real and alarming (crash/collision language is genuinely present, correctly
matching `safety_risk: yes`) but carry no injury signal in the text — and the `medium`
label is correct given that text. This directly confirms the "shared-signal problem"
theory from the three-round retraining story
(`docs/learning/06_class_imbalance_three_rounds.md`) by direct human reading, not just
by the pattern of what rebalancing did and didn't fix.

---

## 7. Round 4: text-supported label correction + atomic decomposition

Section 6 found two things: `safety_risk` had real, measurable label noise (57% of v2's
errors traced to labels the text itself contradicted), and the `severity` medium/high
seesaw did not (0/20 confused rows had any injury language, confirmed by both keyword
audit and full manual read). Round 4 acts on the first finding and tests whether giving
the model explicit, auditable "what does the text actually say" signals helps with both.

**What changed** (full methodology, exact row counts, and the two borderline-call
writeups are in `docs/label-strategy.md`'s Round 4 section):
- **42 of 900 `train.jsonl` rows** (4.7%) had `safety_risk`/`severity` corrected to match
  the narrative text — via an automated audit, two regex fix rounds, then hand review
  once the automated pool hit a ~25% false-positive ceiling regex patching couldn't push
  past. (Originally 38/900 at the time v4 was trained; a later forensic pass found and
  fixed one more `HEDGE` bug, raising it to 42 — this changed the reporting-layer ground
  truth only, not the already-trained model, see the recall bullet below.) `eval.jsonl`
  was never touched — still the fixed comparison point across all four rounds.
- **Target schema expanded from 4 fields to 7**: `component`, `defect_type`,
  `safety_risk`, `severity` (unchanged) plus `crash_described`, `fire_described`,
  `injury_described` — booleans derived from the same validated text lexicon, not NHTSA
  metadata.
- Hyperparameters unchanged (`r=16`, `alpha=32`, `use_dora=True`, `lr=2e-4`, 3 epochs);
  `MAX_SEQ_LENGTH` raised 768→896, a mechanical consequence of the longer 7-field target
  (checked against the real tokenizer, not assumed — 16/1040 rows exceeded 768).

**Disclosed gap, stated plainly rather than buried in a parenthetical: the shipped
v4-FINAL model was actually trained on the 38-correction version of `train_v4.jsonl`,
not the 42-correction version described throughout this section.** The forensic
recall-drop investigation below found and fixed a `HEDGE` bug *after* training had
already completed, which raised the correction count from 38 to 42 (4 rows, a 0.44
percentage-point shift). **No retrain was run for this delta** — at 4/900 rows, and
given the rows involved were already being handled conservatively (left unchanged
rather than mis-corrected) rather than actively wrong, it wasn't judged worth the
Colab/Kaggle GPU cost. Full reasoning in `docs/training-hyperparameters.md`'s Round 4
section. This means every accuracy number in this section reflects the model as
actually trained and shipped, evaluated against the *current* (42-correction)
text-consistent ground truth — a real, small mismatch between training-time and
reporting-time label versions, disclosed here rather than hidden.

### Checkpoint decision: epoch2 vs. epoch3, decided on task metrics, not loss alone

v4's epoch2→epoch3 eval loss regression was small (0.7723 → 0.7772, **0.63%** — smaller
than v1's 0.83% and v2's 0.79%), small enough to plausibly be validation noise on a
140-example eval set rather than confirmed overfitting. So this round didn't inherit the
v1-v3 default of "epoch 2 wins" — both checkpoints were fully evaluated and compared on
real downstream metrics.

**Epoch2 won, 11 metrics to 3 (3 ties):**

| metric | epoch3 | epoch2 | winner |
|---|---|---|---|
| component accuracy (strict) | 70.0% | 67.1% | epoch3 (+2.9pp) |
| component accuracy (relaxed) | 72.1% | 70.7% | epoch3 (+1.4pp) |
| defect_type accuracy | 71.4% | 72.1% | epoch2 (+0.7pp) |
| safety_risk accuracy (official) | 90.0% | 91.4% | epoch2 (+1.4pp) |
| safety_risk accuracy (text-consistent) | 95.0% | 96.4% | epoch2 (+1.4pp) |
| safety_risk=yes precision | 87.8% | 88.4% | epoch2 (+0.6pp) |
| safety_risk=yes recall | 80.0% | 84.4% | epoch2 (+4.4pp) |
| severity accuracy (official) | 77.9% | 79.3% | epoch2 (+1.4pp) |
| severity accuracy (text-consistent) | 82.9% | 83.6% | epoch2 (+0.7pp) |
| severity: medium | 29.6% | 33.3% | epoch2 (+3.7pp) |
| severity: high | 61.1% | 66.7% | epoch2 (+5.6pp) |
| crash_described accuracy | 93.6% | 92.9% | epoch3 (−0.7pp) |
| fire_described / injury_described | tie | tie | — |

Epoch2 won every `safety_risk` metric (both label framings) and every `severity` metric,
including the two historically hardest tiers. Epoch3 only won on `component` — a
lower-priority field, since `safety_risk` is this project's load-bearing number
(`blueprint.md` Section 3) — and `crash_described`, by a margin (0.7pp, ~1 example)
too thin to weigh against a near-clean sweep elsewhere. **Epoch2 ships as v4.** This
validates the same direction v1-v3 always took by default, but this time on evidence
instead of an assumption carried over from loss alone.

### v1 → v2 → v3 → v4: full comparison, official labels

| metric | v1 | v2 (Rounds 1-3 shipped) | v3 | **v4 (shipped)** |
|---|---|---|---|---|
| component accuracy | 68.6% | 64.3% | 66.4% | **67.1%** |
| defect_type accuracy | 69.3% | 67.1% | 71.4% | **72.1%** |
| safety_risk accuracy | 81.4% | 85.0% | 86.4% | **91.4%** |
| severity accuracy | 70.0% | 70.7% | 75.7% | **79.3%** |
| safety_risk=yes precision | 68.6% | 71.4% | 86.1% | **88.4%** |
| safety_risk=yes recall | 77.8% | **88.9%** | 68.9% | 84.4% |
| severity: low | 83.2% | 83.2% | 94.7% | **94.7%** |
| severity: medium | 70.4% | 14.8% | 59.3% | 33.3% |
| severity: high | 0.0% | **88.9%** | 0.0% | 66.7% |

**v4 beats v2 on every headline metric except one uncontested regression** (`severity:
high`) **and one that looks like a regression in the aggregate number but isn't one on
forensic inspection** (`safety_risk` recall):

- **`safety_risk` recall, forensically verified row-by-row**: the raw confusion matrices
  (computed directly from `eval_results_v2.json`/`eval_results_v4_epoch2.json`, not
  recalled from the summary numbers) are v2 TP=40/FP=16/FN=5/TN=79 and v4
  TP=38/FP=5/FN=7/TN=90 — both reproduce the reported 88.9%/84.4% recall exactly.
  Pulling every row where the two models disagree gives the full picture: **14 rows
  where v4 fixed a mistake v2 made, 0 rows where v2 fixed a mistake v4 made, and 2 rows
  where v4 newly misses what v2 caught.** All 14 fixes are false positives (v2 wrongly
  said `yes`; every one has `crash_described`/`fire_described`/`injury_described` all
  `False` in v4's output, and several — 10432604, 10468085, 10660775, 10536141,
  11742196, 11056804, 10097937 — are exact matches to the "text contradicts label" rows
  already identified in Section 6). **Neither of the 2 new misses is a case of the model
  reading real danger signal worse than v2 did:**
  - Odino 964875 ("...did not engage vehicle in reverse. causing loss of control" — no
    described impact) has NHTSA `crash=True` but its own audit-corrected entry in
    `data/processed/eval_text_consistent.json` is `label_source:
    text_corrected_downgrade` — **the project's own Round 4 audit already determined
    this row's official `yes` label is itself text-unsupported.** v4 predicting `no`
    matches the corrected label; this isn't a miss by the text-consistent standard.
  - Odino 876084 (tire tread separated at highway speed, driver pulled over safely, no
    collision described) traced to a spurious hedge match — rerunning the detector on
    the raw text directly gave `hits=[]`, `hedge='RECALL'`, and the "recall" trigger
    fired on an unrelated sentence ("tires were not on recall or on advisory list"), not
    a hypothetical-risk mention. **Update since this was first written:** that specific
    bug (bare "recall" as a hedge trigger, with no risk/hypothetical co-occurrence
    requirement) was found to affect 10 rows total and was fixed in `text_support_audit.py`
    (v5) — see `docs/label-strategy.md`'s Round 4 section for the full forensic pass. With
    the fix applied, 876084's own audit-corrected entry is now also
    `label_source: text_corrected_downgrade` (`safety_risk: no`), matching v4's
    prediction. **Both of the 2 new misses are now audit-confirmed non-misses**, not one
    confirmed and one merely ambiguous.

  **So the actual, verified picture is 14 clean wins and 0 clean losses** — the −4.5pp
  aggregate recall number is real (both new misses keep their official `yes` label,
  since `eval.jsonl` is never modified, so they count against v4 in the official-label
  metric regardless of how contested the underlying evidence is), but it should not be
  read as "v4 got worse at catching real safety risks." It's the same phenomenon behind
  the 14 fixes — trusting the text over an ambiguous or contradicted metadata flag —
  landing as a miss instead of a fix on 2 edge cases that the project's own audit now
  says shouldn't count as misses at all.
- **`severity: high`**: v2's 88.9% vs. v4's 66.7% (−22.2pp) — the one metric where v2 is
  still clearly better. v2 was specifically built to solve high-severity detection (the
  v1→v2 injury-only-high fix, Section 2), and v4's training data wasn't rebuilt around
  that same fix — the 38 Round 4 corrections target label-text mismatches, not the
  sub-pattern imbalance v2 solved. The atomic fields didn't erase this gap: giving the
  model `injury_described` as an explicit signal improved `severity: medium` a lot
  (33.3% vs. v2's 14.8%) but didn't fully recover v2's `high`-tier strength. **This is a
  real, honest limitation of v4, not a result to obscure** — the shared-signal problem
  documented in `docs/learning/06_class_imbalance_three_rounds.md` still isn't fully
  solved; Round 4 improved the label quality feeding it without resolving the underlying
  medium/high separability issue.

### Text-consistent numbers (adjusted ceiling) and new atomic-field metrics

New in v4 — graded against `data/processed/eval_text_consistent.json` (7/140 rows
adjusted as of a v5 fix to a false "recall"-keyword hedge trigger — see the recall
bullet above; originally 5/140 — same audit+hand-review standard as the training
corrections, reporting only, `eval.jsonl` itself untouched):

| metric | official labels | text-consistent (adjusted) |
|---|---|---|
| safety_risk accuracy | 91.4% | **96.4%** |
| safety_risk=yes precision | 88.4% | 93.0% |
| safety_risk=yes recall | 84.4% | 95.2% |
| severity accuracy | 79.3% | 83.6% |

Atomic-field accuracy (crash/fire/injury described, vs. the same reporting layer):
`crash_described` 92.9%, `fire_described` 97.9%, `injury_described` 93.6%. Relaxed
`component` accuracy (matches any co-occurring valid label, not just the strict
first-listed target): 70.7% vs. 67.1% strict, on 31/140 multi-component complaints.

### Is this near the realistic ceiling?

Before Round 4 ran, the label-noise audit (Section 6) implied a rough prediction: with
~15-20% confirmed label noise on `safety_risk`, a realistic ceiling for that field sits
around 90-92%, not higher — and general noisy-label text classification research
(Clothing1M, WebVision-style benchmarks) tops out around 75-86% even with dedicated
noise-robust methods, for comparison.

**v4's `safety_risk` accuracy landed at 91.4% (official) / 96.4% (text-consistent) —
right at, and above, the predicted ceiling.** That's a clean result to report
plainly: **further iteration on `safety_risk` specifically is not expected to help much
more**, because the model is now performing close to what the label quality itself
allows, not being held back by a fixable model deficiency. The text-consistent number
exceeding the predicted ceiling (96.4% > 92%) makes sense given it's graded against
labels the audit itself cleaned — the official-label number (91.4%) is the fairer one to
judge the ceiling against, and it lands squarely inside the predicted 90-92% band.

`severity` accuracy (79.3% official / 83.6% text-consistent) sits inside the general
75-86% noisy-label benchmark range too, but the tier-level breakdown shows this isn't a
uniform ceiling story: `low` is well past it (94.7%), `medium` is well under (33.3%),
`high` regressed from v2 rather than improving. **This is the honest, undissolved part of
Round 4's result**: the safety_risk-level label noise fix worked as evidenced (landed at
its predicted ceiling), but the medium/high severity boundary — already confirmed in
Section 6 to be a genuine model-separability problem, not a labeling one — is not
something this round's intervention (better labels, an explicit text-signal field) was
ever positioned to fix, and it didn't. **No new diagnosable cause surfaced for the
medium/high gap in Round 4** the way each prior round surfaced one (v1's sub-pattern
imbalance, v2's absolute-count problem) — so unlike those, this isn't a lead into an
obvious Round 5. Per the shared-signal theory in `docs/learning/06_...md`, closing it
would need a structurally different intervention (e.g., loss reweighting toward
injury-specific vocabulary, or a two-stage "is there risk" / "how severe" architecture)
rather than another data-quality pass — flagged as an open question for future work, not
a default next round.

---

## Sources

`eval/eval_results_v1.json`, `eval/eval_results_v2.json`, `eval/eval_results_v3.json` —
each graded on the identical, byte-verified-unchanged 140-example `data/processed/eval.jsonl`.
Section 4's review-trigger rules were measured directly against v2's saved predictions
with `scripts/boundary_review_analysis.py` (narrow rule) and
`scripts/boundary_review_analysis_v2.py` (broader rule) — no retraining, no GPU, pure
analysis of already-saved output. Section 6's text-support audit used
`scripts/text_support_audit.py` (the detector), `scripts/pull_validation_sample.py`
(hand-validation sampling), `scripts/full_text_support_audit.py` (the full-scale run),
`scripts/isolate_hit_pattern_cases.py` and `scripts/medium_high_injury_check.py`
(follow-up checks) — also no retraining, no GPU, measurement only against files already
in the repo. Section 7's Round 4 comparison used `eval/eval_results_v4_epoch3.json` and
`eval/eval_results_v4_epoch2.json` (both graded on the same unchanged `eval.jsonl`, plus
the new `data/processed/eval_text_consistent.json` reporting layer for the
text-consistent numbers) — real Kaggle GPU runs, not simulated; the checkpoint
comparison table was computed directly from both files, not eyeballed. Training-side
detail in `docs/training-hyperparameters.md`; label-derivation, dataset-composition, and
the full Round 4 label-correction methodology in `docs/label-strategy.md`.
