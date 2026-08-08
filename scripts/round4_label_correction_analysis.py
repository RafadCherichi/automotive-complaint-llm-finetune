"""Round 4, Phase 1: DRY-RUN analysis of training-label corrections (measurement +
proposal only -- writes NOTHING to data/processed/*.jsonl). Per the Round 4 design:
apply text_support_audit.py's validated lexicon to train.jsonl, and identify rows
where the label has a CLEAR, NON-HEDGE contradiction with the narrative text.

Correction rule (binary alarm framing, matching the already-validated
text_support_audit.classify() methodology used in the two prior audits):
  - crash_described / fire_described / injury_described = the 3 corresponding
    text_support_audit categories, negation-aware.
  - text_alarm = crash_described or fire_described or injury_described.
  - UPGRADE: label says no-alarm (safety_risk=no / severity=low) but text_alarm is
    True AND no hedge co-occurs -> a real, unhedged category fired. Correct to
    alarm. New severity = "high" if injury_described else "medium" (same formula
    as the original NHTSA-flag-based derivation, see docs/label-strategy.md, just
    fed by text-derived flags instead of NHTSA flags for this row).
    v2 of this script: UPGRADE now requires no-hedge, same as DOWNGRADE always
    has -- the first dry-run showed hedge-co-occurring "upgrades" were mostly
    near-miss/hypothetical language ("almost caused a crash," "could result in,"
    a recall notice's boilerplate risk language), not a described real event.
    Rewriting ground truth needs a higher precision bar than the original
    audit's softer "does this sound alarming enough to question a label" one.
  - DOWNGRADE: label says alarm but text_alarm is False AND no hedge language
    either (a genuinely clean narrative). Correct to no-alarm / low.
    SAFETY EXEMPTION: never downgrade a row where NHTSA's own injured>0 or
    deaths>0 -- these are the highest-consequence, must-not-miss cases, and the
    injury lexicon has no dedicated death-language detection (e.g. "killed,"
    "fatal," "deceased" aren't in the word list). Trusting an absent keyword
    match to override a documented fatality/injury would be reckless. Logged
    separately, not silently dropped.
  - Hedge-flagged ambiguous rows (label=alarm, text_alarm=False, hedge=True) are
    left untouched entirely, per the Round 4 instruction not to guess on
    genuinely unclear cases.
  - Only the binary alarm/no-alarm boundary is corrected (matching how the whole
    audit has always framed "text-supported"); medium-vs-high TIER mismatches
    within an already-alarm-consistent row are NOT touched here -- see the
    design-decision note printed at the end of this script's output.
"""
import json
import random
import sys

sys.path.insert(0, "scripts")
from text_support_audit import CRASH, FIRE, INJURY, HEDGE, _is_negated


def fires(text, pattern):
    for m in pattern.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return m.group(0)
    return None


def analyze(rows):
    upgrades, downgrades, exempted, unchanged, hedge_left = [], [], [], [], []
    for r in rows:
        text = r["narrative"]
        crash_d = fires(text, CRASH)
        fire_d = fires(text, FIRE)
        injury_d = fires(text, INJURY)
        hedge_m = HEDGE.search(text)
        text_alarm = bool(crash_d or fire_d or injury_d)
        current_alarm = r["safety_risk"] == "yes"
        # hospital-only tracking: was INJURY's only trigger the ambiguous bare
        # "hospital" word (location vs. treatment), with nothing else firing?
        hospital_only = (injury_d is not None and injury_d.lower() == "hospital"
                          and not crash_d and not fire_d)

        rec = {
            "row": r, "crash_described": bool(crash_d), "fire_described": bool(fire_d),
            "injury_described": bool(injury_d), "hedge": bool(hedge_m),
            "hospital_only_driver": hospital_only,
            "snippets": {"crash": crash_d, "fire": fire_d, "injury": injury_d, "hedge": hedge_m.group(0) if hedge_m else None},
        }

        if not current_alarm and text_alarm and not hedge_m:
            new_severity = "high" if injury_d else "medium"
            rec["old"] = {"safety_risk": r["safety_risk"], "severity": r["severity"]}
            rec["new"] = {"safety_risk": "yes", "severity": new_severity}
            upgrades.append(rec)
        elif current_alarm and not text_alarm and not hedge_m:
            if r["injured"] > 0 or r["deaths"] > 0:
                rec["reason"] = f"injured={r['injured']} deaths={r['deaths']} -- exempted from downgrade"
                exempted.append(rec)
            else:
                rec["old"] = {"safety_risk": r["safety_risk"], "severity": r["severity"]}
                rec["new"] = {"safety_risk": "no", "severity": "low"}
                downgrades.append(rec)
        elif not current_alarm and text_alarm and hedge_m:
            hedge_left.append(rec)  # would-be upgrade, but hedge co-occurs -- left alone
        elif current_alarm and not text_alarm and hedge_m:
            hedge_left.append(rec)  # would-be downgrade, but hedge co-occurs -- left alone
        else:
            unchanged.append(rec)
    return upgrades, downgrades, exempted, unchanged, hedge_left


train_rows = [json.loads(l) for l in open("data/processed/train.jsonl", encoding="utf-8")]
upgrades, downgrades, exempted, unchanged, hedge_left = analyze(train_rows)

print("=" * 70)
print("ROUND 4 PHASE 1 -- label-correction DRY RUN on train.jsonl (900 rows)")
print("(no files modified -- this is a proposal only)")
print("=" * 70)
print(f"\nUPGRADE (no/low -> yes/alarm, clear text contradiction):  {len(upgrades)}")
print(f"DOWNGRADE (yes/alarm -> no/low, clear text contradiction): {len(downgrades)}")
print(f"EXEMPTED from downgrade (injured>0 or deaths>0 per NHTSA): {len(exempted)}")
print(f"LEFT UNTOUCHED -- hedge-only ambiguous:                    {len(hedge_left)}")
print(f"LEFT UNTOUCHED -- already text-consistent:                 {len(unchanged)}")
print(f"TOTAL:                                                     {len(train_rows)}")
print(f"\nNet rows changed: {len(upgrades) + len(downgrades)} / 900 "
      f"({100*(len(upgrades)+len(downgrades))/900:.1f}%)")

rng = random.Random(7)


def show(recs, label, n):
    print(f"\n{'-'*70}\n{label} -- showing {min(n, len(recs))} of {len(recs)}\n{'-'*70}")
    sample = recs if len(recs) <= n else rng.sample(recs, n)
    for rec in sample:
        r = rec["row"]
        print(f"\nodino={r['odino']}")
        print(f"narrative: {r['narrative'][:400]}{'...' if len(r['narrative'])>400 else ''}")
        print(f"NHTSA flags: crash={r['crash']} fire={r['fire']} injured={r['injured']} deaths={r['deaths']}")
        print(f"text signals: crash_described={rec['crash_described']} fire_described={rec['fire_described']} "
              f"injury_described={rec['injury_described']} hedge={rec['hedge']}  snippets={rec['snippets']}")
        if "old" in rec:
            print(f"CORRECTION: {rec['old']} -> {rec['new']}")
        elif "reason" in rec:
            print(f"NOT corrected: {rec['reason']}")


show(upgrades, "UPGRADE sample", 12)
show(downgrades, "DOWNGRADE sample", 12)
show(exempted, "EXEMPTED (full list -- should be small)", 20)

hospital_only_upgrades = [rec for rec in upgrades if rec["hospital_only_driver"]]
print(f"\n{'-'*70}")
print(f"'hospital'-only-driven UPGRADE corrections (no other injury/crash/fire "
      f"signal): {len(hospital_only_upgrades)} / {len(upgrades)}")
print(f"{'-'*70}")
for rec in hospital_only_upgrades:
    r = rec["row"]
    print(f"\nodino={r['odino']}")
    print(f"narrative: {r['narrative'][:400]}{'...' if len(r['narrative'])>400 else ''}")

print(f"\n{'-'*70}")
print("DESIGN-DECISION NOTE (flagging for review, not silently decided):")
print(f"{'-'*70}")
print("""This correction only touches the binary low<->alarm boundary (matching how
text_support_audit.classify() has always framed "text-supported" throughout both
prior audits). It does NOT attempt to correct medium<->high TIER mismatches
within rows that are already alarm-consistent (e.g. a row correctly flagged as
alarm by both text and label, but where the label says "medium" and the text's
injury_described would suggest "high", or vice versa). Rationale: the medium/high
boundary was the subject of the severity-seesaw investigation and confirmed to be
a genuine, hard-to-separate distinction -- not a simple lexicon-driven relabel.
Silently reaching into that boundary here would blur this investigation's own
finding. Flagging this scope choice explicitly for your review before finalizing.""")
