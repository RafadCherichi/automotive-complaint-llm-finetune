"""Round 4: builds the "text-consistent" REPORTING layer for eval.jsonl.

This does NOT modify data/processed/eval.jsonl -- that file stays byte-identical,
the fixed comparison point across all four rounds. This script produces a
separate side file, data/processed/eval_text_consistent.json, used only by the
eval notebook to report a second, "adjusted ceiling" accuracy number alongside
the official one. Same methodology as train_v4.jsonl's correction (automated
audit + hand review), applied here for reporting only, never for training.

5 of 140 rows corrected in the original pass, hand-reviewed individually (small
enough pool to review every candidate, not just a sample):
  - 2 UPGRADE (odino 10081273: real explosion: "coil was blown out of the
    cylinder"; odino 871031: real described symptoms, "made passengers ILL and
    temporarily BLIND" -- one of the 3 original motivating examples for this
    whole investigation)
  - 3 DOWNGRADE (odino 721649: link-only complaint, no narrative content to
    support the crash flag; odino 11468953 and 964875: both describe a rollaway
    stopped before any impact)
  - 1 downgrade candidate REJECTED: odino 11685196's "the car gently RAN INTO
    the car in front of me" is a real collision the CRASH lexicon doesn't catch
    ("ran into" isn't in the pattern) -- a lexicon gap, not label noise. Caught
    by cross-referencing an earlier read of this same row in a prior exercise.
  - 4 upgrade candidates rejected: near-miss/hypothetical/feature-name matches,
    same categories found in the train.jsonl review.

v5 update: removing HEDGE's bare "recall" trigger (text_support_audit.py) freed
up 2 more downgrade rows (odino 11339982: a bare listing of recall numbers with
no narrative content; odino 876084: tire tread separation, no collision
described -- both previously blocked by a spurious "recall" hedge match) and
surfaced 2 more upgrade CANDIDATES that were hand-reviewed and REJECTED (odino
11624858: "potentially...vulnerable to a rear-end collision," hypothetical, not
a real event; odino 11183247: "the recall for the Souls catching on fire"
describes OTHER vehicles in a recall notice, not this complainant's own car).
Net change: +2 downgrades (5 -> 7 corrected rows total), 0 new upgrades.
"""
import json
import sys

sys.path.insert(0, "scripts")
from text_support_audit import CRASH, FIRE, INJURY, HEDGE, _is_negated

HAND_REVIEWED_KEEP = {
    "10081273": {"crash": False, "fire": True, "injury": False},   # coil explosion, real
    "871031":   {"crash": False, "fire": False, "injury": True},   # "made passengers ill and blind", real
}
HAND_REVIEWED_UPGRADE_REJECT = {
    "11487942", "11515420", "11735257", "10882196",  # original 4
    "11624858", "11183247",  # v5: hypothetical / other-vehicle recall mention
}
HAND_REVIEWED_DOWNGRADE_REJECT = {"11685196"}  # "ran into" -- real collision, lexicon gap


def fires(text, pattern):
    for m in pattern.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return m.group(0)
    return None


rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
assert len(rows) == 140

out = {}
counts = {"downgrade": 0, "upgrade_handreviewed": 0, "exempted": 0, "hedge_left": 0, "unchanged": 0}

for r in rows:
    odino = r["odino"]
    text = r["narrative"]
    entry = {
        "safety_risk": r["safety_risk"],
        "severity": r["severity"],
    }

    if odino in HAND_REVIEWED_KEEP:
        v = HAND_REVIEWED_KEEP[odino]
        entry["crash_described"] = v["crash"]
        entry["fire_described"] = v["fire"]
        entry["injury_described"] = v["injury"]
        entry["safety_risk"] = "yes"
        entry["severity"] = "high" if v["injury"] else "medium"
        entry["label_source"] = "text_corrected_upgrade_handreviewed"
        counts["upgrade_handreviewed"] += 1
        out[odino] = entry
        continue

    crash_d = fires(text, CRASH)
    fire_d = fires(text, FIRE)
    injury_d = fires(text, INJURY)
    if injury_d is not None and injury_d.lower() == "hospital" and not crash_d and not fire_d:
        injury_d = None
    hedge_m = HEDGE.search(text)
    text_alarm = bool(crash_d or fire_d or injury_d)
    current_alarm = r["safety_risk"] == "yes"

    entry["crash_described"] = bool(crash_d)
    entry["fire_described"] = bool(fire_d)
    entry["injury_described"] = bool(injury_d)

    if odino in HAND_REVIEWED_DOWNGRADE_REJECT:
        entry["label_source"] = "original_handreviewed_downgrade_reject_lexicon_gap"
        # override: text DOES support the label (lexicon missed "ran into") --
        # force crash_described True to reflect the real event
        entry["crash_described"] = True
        counts["unchanged"] += 1
    elif odino in HAND_REVIEWED_UPGRADE_REJECT:
        entry["label_source"] = "original_handreviewed_upgrade_reject"
        entry["crash_described"] = entry["fire_described"] = entry["injury_described"] = False
        counts["unchanged"] += 1
    elif current_alarm and not text_alarm and not hedge_m:
        if r["injured"] > 0 or r["deaths"] > 0:
            entry["label_source"] = "original_exempted_high_stakes"
            counts["exempted"] += 1
        else:
            entry["safety_risk"] = "no"
            entry["severity"] = "low"
            entry["label_source"] = "text_corrected_downgrade"
            counts["downgrade"] += 1
    elif hedge_m and (text_alarm != current_alarm):
        entry["label_source"] = "original_hedge_ambiguous"
        counts["hedge_left"] += 1
    else:
        entry["label_source"] = "original_text_consistent"
        counts["unchanged"] += 1
    out[odino] = entry

assert len(out) == 140
print("Row-disposition counts (eval.jsonl, reporting layer only):")
for k, v in counts.items():
    print(f"  {k}: {v}")
print(f"  TOTAL: {sum(counts.values())}")
print(f"\nNet corrected: {counts['downgrade'] + counts['upgrade_handreviewed']} / 140 "
      f"({100*(counts['downgrade']+counts['upgrade_handreviewed'])/140:.1f}%)")

with open("data/processed/eval_text_consistent.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print("\nWrote data/processed/eval_text_consistent.json (140 entries, odino -> "
      "text-consistent safety_risk/severity + atomic fields, REPORTING ONLY)")
print("data/processed/eval.jsonl itself was not opened for writing -- still byte-identical.")
