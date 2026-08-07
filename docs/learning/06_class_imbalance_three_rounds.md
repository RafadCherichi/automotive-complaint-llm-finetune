# 6. Class Imbalance Across Three Training Rounds — the Project's Core Technical Finding

## (1) The concept

**Class imbalance** is when a dataset has far more examples of one category than
another. If a model trains on data that mirrors a heavily imbalanced real-world split
directly, it can reach a deceptively high overall accuracy just by always guessing the
majority class — like a smoke detector that's "99% accurate" only because it almost
never goes off, which is exactly backwards from what a smoke detector is for.
**Oversampling** means deliberately including *more* examples of the rare/minority class
than its natural frequency would suggest, specifically so the model gets enough exposure
to actually learn it.

A subtler version of the same problem: sometimes what looks like *one* category from the
outside actually contains distinct **sub-patterns**, and a model needs separate exposure
to each sub-pattern to learn the category properly — oversampling the category as a
whole isn't enough if all the added examples happen to be the same sub-pattern.

An even subtler failure mode, and the one this project actually ran into twice: a
**shared-signal problem**. If two different labels are distinguished by only *one* piece
of information in the text, and that same underlying signal governs both labels, then
adding more training examples to either side doesn't teach the model to read that signal
more precisely — it just shifts which label the model defaults to when uncertain.

## (2) How this project uses it — with real numbers, across three rounds

**Round 0 (dataset design, Phase 1):** real-world NHTSA complaints with any safety flag
(crash/fire/injury/death) are only ~7% of the corpus (148,338 of ~2.1M). Trained
naturally, a model could hit a high accuracy score while missing almost every real
safety signal — precisely the failure mode this project's safety-triage framing exists
to prevent. Fixed before any training happened, by stratifying both the training and
eval sets to ~32% `safety_risk: yes` (`docs/label-strategy.md`, "Class balance").

**v1 → v2 (the sub-pattern problem):** even after that top-level fix, `severity: high`
split into two sub-patterns that weren't evenly represented: complaints with injury/death
*and* a crash/fire flag (71 of 99 high-tier examples, 71.7%), versus injury/death *with
no* crash/fire flag (only 28, 28.3%). The model, trained mostly on the first sub-pattern,
learned "crash or fire mentioned → medium" as a shortcut and got **0% accuracy on real
`severity: high` complaints** — it never learned to recognize injury language on its
own, without crash/fire language alongside it. Fixed by rebalancing the sub-pattern split
within the `high` tier specifically (57 injury-only examples added in place of 57
`medium` examples) — accuracy on `severity: high` went to 88.9%
(`docs/eval-report.md` Section 2).

**v2 → v3 (the shared-signal problem):** fixing `high` revealed a new problem —
`severity: medium` accuracy had collapsed from 70.4% to 14.8%. The model's "I'm not
sure" default had simply moved: from over-guessing `medium` (v1) to over-guessing `high`
(v2). The seemingly obvious fix — give `medium` more raw training examples (100 → 200,
following the standard 200-500-per-class guidance for this model size) — improved
`medium` somewhat (14.8% → 59.3%) but **undid the `high` fix in the process**, dropping
`high` back to 0.0% — the exact same failure v1 had (`docs/training-hyperparameters.md`'s
three-round table).

**Why "just add more data" doesn't always work — the actual lesson:** `medium` and
`high` are distinguished by exactly one signal in the text: whether injury/death
language is present, layered on top of crash/fire language that's common to *both*
tiers. Both fixes tried to teach the model to read that one signal by rebalancing raw
example counts between the two classes — but because real training data is finite,
that's a zero-sum move: examples added to one tier's count are effectively examples not
added to the other's. Rebalancing shifts which class the model's uncertainty lands on;
it doesn't teach the model to read the deciding signal more precisely. That would likely
need a different kind of intervention entirely — e.g. explicitly weighting the training
loss toward injury-specific vocabulary, or a two-stage design that separates "is there
any risk" from "how severe is it" using different signals for each stage. Neither was
attempted here; both are flagged as open questions for future work rather than a fourth
retraining round (`docs/label-strategy.md`'s "Known limitations").

## (3) When the alternative would win

- **Simple oversampling (no sub-pattern analysis)** is enough when a class's internal
  examples are actually homogeneous — no hidden sub-patterns. It failed here specifically
  because `severity: high` wasn't homogeneous; injury-only and crash+injury complaints
  needed separate representation.
- **"Add more data" class-count rebalancing** works when two classes are separated by
  *multiple, largely independent* signals in the data — in that case, more examples of
  either class genuinely do teach the model more about that class's distinguishing
  features, without a shared-signal tradeoff pulling against the other class. It failed
  here because `medium` and `high` share almost all of their surface signal (crash/fire
  language) and differ on exactly one dimension (injury presence) — the shared-signal
  case is the one this fix can't solve, and needed to be documented as an honest,
  unresolved limitation instead.
