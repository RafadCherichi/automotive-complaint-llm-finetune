# 7. Atomic Decomposition: Splitting a Derived Judgment into Its Checkable Evidence

## (1) The concept

A **compound label** is a target that's actually a judgment computed from several
underlying facts — `severity: high` isn't a single observation, it's the output of a rule
(`injured>0 or deaths>0`) applied to more basic facts. When a model is trained only on
the compound label, it has to learn both things at once: what the underlying evidence
looks like in the input, *and* how to combine that evidence into the final judgment. If
the compound label is ever noisy or wrong, the model has no way to signal *why* it
disagrees — it can only output a different compound guess.

**Atomic decomposition** means adding the underlying atomic facts as their own
targets, alongside (not instead of) the compound one. The model now has to predict
`crash_described: true/false` on its own merits, checkable independently against the
text — which does two things: it forces the model to ground its compound judgment in
something verifiable, and it gives *the person evaluating the model* a way to tell
whether a wrong compound answer came from misreading the evidence or from applying the
rule incorrectly to evidence it read fine.

## (2) How this project uses it

Round 4 added three atomic boolean fields — `crash_described`, `fire_described`,
`injury_described` — to the 4-field target (`component`, `defect_type`, `safety_risk`,
`severity`), derived from the same validated text lexicon used in the Section 6 label-noise
audit (`scripts/text_support_audit.py`), not from NHTSA's structured flags. The motivation
was directly evidential: Section 6 found 57% of the shipped model's `safety_risk` errors
were on rows where the label itself contradicted the narrative text. Giving the model an
explicit "what does the text say" field, trained alongside the compound judgment, tests
whether that grounding improves the compound judgment too — and lets the atomic fields be
graded on their own, separately from `safety_risk`/`severity`.

**Result, reported honestly both ways:** `safety_risk` accuracy improved from v2's 85.0%
to v4's 91.4% (95.0% against the text-consistent, audit-adjusted labels) — landing right
at the ~90-92% ceiling the label-noise audit predicted going in, evidence the atomic
fields (or at minimum, the label corrections they came bundled with) helped close a real,
identifiable gap. The atomic fields themselves came back accurate on their own terms too
(92.9%-97.9% depending on the field). But decomposition didn't fix everything it might
have: `severity: high` accuracy *regressed* from v2's 88.9% to v4's 66.7%, even with
`injury_described` available as an explicit signal — because the medium/high boundary was
already confirmed (Section 6, via full manual read of 20 confused rows) to be a genuine
model-separability problem, not a label-quality one. Atomic decomposition only helps the
part of the problem that actually was a grounding/label issue; it isn't a general fix for
a model failing to weigh evidence it's already reading correctly.

## (3) When the simpler 4-field approach would have been enough

- **When the compound label isn't actually noisy.** Decomposition is a fix for "the model
  can't tell why a compound label disagrees with the evidence" — if an audit like
  Section 6's hadn't found real label-text contradictions, there'd be no noise for the
  atomic fields to help ground, and the extra target fields would just be added output
  surface with no corresponding benefit (as happened here for `severity: high`, where the
  underlying issue was never a labeling problem).
- **When the atomic facts aren't independently checkable.** This only works because
  `crash_described`/`fire_described`/`injury_described` can be validated against the text
  by the same deterministic lexicon used to build the label correction in the first
  place — a shared, auditable source of truth for both. Decomposing a judgment into
  "atomic" fields that are themselves just as subjective or unverifiable as the compound
  one (e.g., splitting "severity" into invented sub-scores with no independent evidence
  to check them against) adds target-schema complexity without adding grounding.
- **When training data or sequence budget is tight enough that more target fields cost
  more than they return.** Every additional field is more for the model to get right per
  example and a longer target sequence per row (Round 4 needed `MAX_SEQ_LENGTH` raised
  768→896 for exactly this reason). On a dataset already at the small end for QLoRA
  fine-tuning (900 rows), that's a real tradeoff, not a free upgrade — worth it here
  because Section 6 had already identified a specific, addressable noise source to ground
  against; not automatically worth it for every compound label.
