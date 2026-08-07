# PM Perspective — Safety-Triage Business Case

This project isn't just "a fine-tuned model exists" — it's a specific business problem
with a specific tradeoff, measured with real numbers. This doc lays out that framing:
who this is for, what it saves them, what it costs when it's wrong, and what a real
production version would need beyond what's built here.

All numbers below are the shipped model's (v2) real, measured results — see
`docs/eval-report.md` for the full evidence, including where this model is still weak.

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

Given a raw complaint (real, unedited customer text), it outputs four structured fields:
which vehicle component is involved, what kind of defect it is, whether it's a safety
risk (`yes`/`no`), and how severe (`low`/`medium`/`high`). Paste in text, get structured
triage data back — instantly, consistently, and at whatever volume the queue needs.

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
  — the model's shipped recall is **88.9%**. It misses about 1 in 9 real safety risks.
- **Precision** ("of everything the model flagged as a safety risk, how many actually
  were?") — the model's shipped precision is **71.4%**. Roughly 7 of every 10 flagged
  complaints are genuine.

For comparison, the *unmodified* base model (before any fine-tuning) had 93.3% recall
but only 32.8% precision — it achieved that higher recall by flagging almost
everything as risky (86 false alarms out of 140 test complaints). A tool that cries wolf
two times out of three trains its own users to stop trusting it — that's not a safer
system, it's a system nobody uses correctly after the first week. The fine-tuned model
trades a small amount of recall for more than double the precision, which is the right
direction for a tool meant to actually be relied on, not just look good on one metric in
isolation.

## Where the tool still needs a human — stated plainly, not hidden

The severity *tier* (`low`/`medium`/`high`) is measurably weaker than the binary
safety-risk decision — the model gets `severity: medium` right only 14.8% of the time.
Reading through the actual misclassified examples (`docs/eval-report.md` Section 3)
shows this splits into two different situations: about half look like the model
overreacting to dramatic-sounding language when nothing actually happened, and about
half look like genuinely defensible disagreements — cases where a reasonable human
reviewer might also lean toward a higher severity than the strict official record
technically supports.

**The recommendation, and it's a measured one, not a guess:** automate the binary
`safety_risk: yes`/`no` decision (the numbers above support trusting it), and route the
severity tier to a human whenever `safety_risk: yes` — that's a workload cost of about
2.9% of all complaints (measured directly against the model's real predictions, not
estimated), for meaningfully better severity judgment on the complaints that matter
most.

**One gap this mitigation does not cover, stated honestly:** two complaints in the test
set were missed entirely — not mis-tiered, but never flagged as a safety risk at all.
Both described a real injury with no crash or fire involved (a liftgate closing on
someone, a trunk lid hitting someone in the head). A "review anything the model
flags" process can't catch a complaint the model never flagged in the first place. This
is a real, unresolved limitation of the current model, not something the human-review
step patches over — see `docs/eval-report.md` Section 4 for the full reasoning and what
a real fix would need (likely an independent keyword-based safety net, not another
review-trigger keyed off this same model's output).

## What a v2 of this tool would need

Not built here — flagged as the honest next steps, grounded in what the evidence from
this project actually points at:

- **A second-opinion checker for injury language**, independent of the model, to catch
  the residual false-negative pattern above — e.g. a lightweight keyword scan for
  injury-related words that runs *in addition to* the model, not instead of it, so a
  miss on one layer doesn't mean a miss overall.
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
