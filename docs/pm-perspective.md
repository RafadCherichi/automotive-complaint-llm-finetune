# PM Perspective — Safety-Triage Business Case

This project isn't just "a fine-tuned model exists" — it's a specific business problem
with a specific tradeoff, measured with real numbers. This doc lays out that framing:
who this is for, what it saves them, what it costs when it's wrong, and what a real
production version would need beyond what's built here.

All numbers below are the current shipped model's (**v4**, Round 4) real, measured
results — see `docs/eval-report.md` for the full evidence, including where this model
is still weak. v4 replaced the earlier v2 model after a follow-up investigation found
that some of v2's measured errors were actually caused by noisy training/eval labels
(cases where the official record disagreed with what the complaint text itself said),
not model mistakes — v4 corrects that noise and adds three explicit "did the complaint
text actually describe X" fields the model must ground its judgment in. Full story in
`docs/eval-report.md` Sections 6-7.

## The problem

An OEM (auto manufacturer) quality/safety team receives thousands of free-text customer
complaints. Every one needs to be read and tagged: what part failed, what kind of
failure, and — critically — is this a genuine safety risk that needs priority review,
or a routine quality issue that can wait in the normal queue?

Doing this by hand doesn't scale, and it isn't even consistent — two different reviewers
reading the same complaint can reasonably disagree on how urgent it is, especially under
time pressure with a large backlog. The risk isn't hypothetical: a real safety complaint
sitting in an unsorted queue behind hundreds of routine ones is exactly the failure mode
that turns into a missed early warning sign for a defect that later becomes a recall.

## What this tool does

Given a raw complaint (real, unedited customer text), it outputs four structured triage
fields: which vehicle component is involved, what kind of defect it is, whether it's a
safety risk (`yes`/`no`), and how severe (`low`/`medium`/`high`) — plus three supporting
fields (`crash_described`/`fire_described`/`injury_described`) that state what the
complaint text itself actually describes, so a reviewer can see the evidence the model's
safety_risk/severity call is grounded in, not just trust the compound judgment blind.
Paste in text, get structured triage data back — instantly, consistently, and at
whatever volume the queue needs.

## The number that actually matters: `safety_risk` precision and recall

Two ways a triage tool can fail, and they cost completely different things:

- **False negative** (misses a real safety risk, calls it "no risk"): the complaint
  sits in the normal queue. If it's an early sign of a real defect, this is how a
  preventable recall becomes a bigger one — the cost of a miss here can be severe and,
  in the worst case, genuinely dangerous.
- **False positive** (flags a routine complaint as a safety risk): an analyst spends a
  few extra minutes reviewing something that turns out to be fine. Annoying, and it
  wastes review capacity, but nobody gets hurt by it.

These costs are wildly asymmetric — a missed real risk is categorically worse than an
extra few minutes of review time. That's exactly why **precision and recall**, not
overall accuracy, are the numbers that matter here:

- **Recall** ("of all the real safety risks in the data, how many did the model catch?")
  — v4's recall is **84.4%**. It misses about 1 in 6 real safety risks.
- **Precision** ("of everything the model flagged as a safety risk, how many actually
  were?") — v4's precision is **88.4%**. Roughly 9 of every 10 flagged complaints are
  genuine.

For comparison, the *unmodified* base model (before any fine-tuning) had 93.3% recall
but only 32.8% precision — it achieved that higher recall by flagging almost
everything as risky (86 false alarms out of 140 test complaints). A tool that cries wolf
two times out of three trains its own users to stop trusting it — that's not a safer
system, it's a system nobody uses correctly after the first week.

**The recall number is lower than the previous shipped model (v2, 88.9% → v4, 84.4%) —
but a row-by-row forensic check (not just the aggregate percentages) shows this is not
v4 getting worse at recognizing real danger.** Pulling every case where the two models
disagree: v4 fixes 14 of v2's mistakes and introduces 0 new clean mistakes of its own.
The 2 rows behind the −4.5pp recall number are contested edge cases, not new model
weaknesses — and **both are now confirmed by this project's own audit**, not just
one of the two as first thought:

- One (a "loss of control" complaint with no described crash) has its official label
  flagged by the audit as text-unsupported — the audit's corrected label agrees with
  what v4 predicted.
- The other (a tire failure with no collision described) was first traced to the
  audit's keyword detector misfiring on an unrelated use of the word "recall." That bug
  turned out to affect 10 rows total and was fixed — and once fixed, this row's own
  audit-corrected label *also* flips to agree with v4's prediction. There's no real
  evidence of danger in the text for either model to have caught.

So the honest picture is: **14 clean wins, 0 clean losses, and 2 rows where the
project's own audit — not just informal judgment — now agrees with v4 over the
official label.** That's a materially better result than "v4 got worse at catching
real risk," which is what the raw recall number alone would suggest. It doesn't erase
the aggregate number — the official metric still counts both edge cases against v4,
since the eval set's labels are deliberately never touched — but it means the case for
treating this as a genuine regression is weak. **The human-review safety net (below) and the still-unbuilt
independent injury-checker (see "What a next iteration would need") remain worth
building regardless** — not because v4 is worse at reading danger signal, but because
even a model with zero clean regressions still won't catch every case where the record
itself is ambiguous, and a second, independent layer is the only way to cover that.

## Where the tool still needs a human — stated plainly, not hidden

The severity *tier* (`low`/`medium`/`high`) is still measurably weaker than the binary
safety-risk decision, and v4 did not uniformly improve it. `severity: medium` improved
to 33.3% (up from v2's 14.8%, still weak in absolute terms), but **`severity: high`
actually got worse — 66.7%, down from v2's 88.9%.** This is a real regression, not an
oversight: the medium/high boundary was separately investigated
(`docs/eval-report.md` Section 6) and confirmed to be a genuine model-separability
problem, not a labeling one — Round 4's fixes targeted label quality, which helped
`safety_risk` and `medium` but doesn't reach the underlying issue for `high`. Practical
consequence: **treat `severity: high` predictions as less trustworthy under v4 than they
were under v2**, and weigh that specifically if severity tier drives any routing or
prioritization logic downstream.

**The recommendation is unchanged from v2, and still measured, not a guess:** automate
the binary `safety_risk: yes`/`no` decision, and route the severity tier to a human
whenever `safety_risk: yes`.

**One gap this mitigation does not cover, and it's still there in v4, checked directly
rather than assumed carried over:** the same two complaints in the test set that v2
missed entirely are *still* missed by v4 — not mis-tiered, never flagged as a safety
risk at all (a liftgate closing on someone, a trunk lid hitting someone in the head;
both real injuries, no crash or fire involved). A "review anything the model flags"
process can't catch a complaint the model never flagged in the first place. Combined
with v4's lower recall (above), this reinforces rather than resolves the case for an
independent safety net — see below.

## What a next iteration of this tool would need

Not built here — flagged as the honest next steps, grounded in what the evidence from
this project actually points at:

- **A second-opinion checker for injury language, genuinely independent of the model.**
  Round 4 added an `injury_described` field, but it's predicted by the *same* model as
  `safety_risk` — it grounds the model's own reasoning, but it isn't an independent
  check, and the same two residual injury-only misses persist in v4 exactly as they did
  in v2 (above). What's still missing is a lightweight, separate keyword-based scan that
  runs *alongside* the model, not as another output from it, so a miss on one layer
  doesn't mean a miss overall.
- **A structurally different fix for the `severity: medium`/`high` boundary** —
  Round 4's evidence rules out "better labels" or "give the model more explicit signal"
  as the fix, since both were tried and `high` still regressed. What's untested is a
  genuinely different approach: loss reweighting toward injury-specific vocabulary
  during training, or splitting into two separate decisions ("is there risk" and "how
  severe," each learned from different signal) instead of one combined tier prediction.
  Flagged as an open question, not a default next round — see `docs/eval-report.md`
  Section 7's ceiling assessment for why more of the same approach isn't expected to help.
- **Multi-label defect tagging.** Right now each complaint gets exactly one
  `component` and one `defect_type`, even when the real complaint describes more than
  one problem (`docs/label-strategy.md` documents this as a known limitation — the raw
  multi-component data is kept, just not used as a training target yet).
- **Severity calibrated against real outcome data, not just NHTSA's own flags.** The
  current `severity` field is derived from whether a complaint already involved a
  crash/fire/injury — useful, but it's a proxy for "how bad did this already get," not
  necessarily "how likely is this to escalate." A production version would ideally
  incorporate downstream outcomes (recalls issued, repeat complaints on the same part)
  to calibrate severity against what actually predicts a real safety escalation.
