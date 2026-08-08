"""Round 4: builds data/processed/eval_v4.jsonl -- needed because the v4 training
notebook's per-epoch validation loop formats eval rows the same way as train rows,
which now requires all 7 target fields, not just 4. eval.jsonl only has the original 4.

NOT the same thing as eval_text_consistent.json:
  - eval_text_consistent.json (Phase 4 reporting layer): safety_risk/severity are
    CORRECTED where text-inconsistent, used only for the "adjusted ceiling" metric.
  - eval_v4.jsonl (this file, training-loop input): component/defect_type/
    safety_risk/severity are byte-for-byte the ORIGINAL eval.jsonl values --
    completely untouched. Only the 3 new atomic fields are added.

Confirmed first (required before trusting detector-only labeling on all 140 rows):
none of eval.jsonl's 140 odinos overlap with the 32 rows from train.jsonl's hand
review (verified directly against scripts/build_train_v4.py's HAND_REVIEWED_KEEP /
HAND_REVIEWED_REJECT sets -- zero overlap, train/eval have never overlapped by
odino per docs/label-strategy.md's original dataset build).

Atomic fields use the same methodology as train_v4.jsonl: hand-reviewed verdict for
the 8 rows in eval.jsonl that were themselves hand-reviewed during the
eval_text_consistent.json build (2 KEEP upgrade candidates + 6 upgrade-candidate
rejects + 1 downgrade-reject lexicon-gap row), automated (v5-fixed) detector for
the other 132.

v5 update: 2 more reject odinos added (11624858, 11183247) -- both were upgrade
candidates that only surfaced after text_support_audit.py's HEDGE fix removed a
false "recall" trigger; both hand-reviewed and rejected (hypothetical framing /
a different vehicle's recall mention, not this complaint's own car). They MUST
be in the reject set here, not left to the automated branch below -- without the
override, the automated computation would show crash_described=True for
11624858 ("rear-end") and fire_described=True for 11183247 ("fire"), both of
which were specifically rejected as not describing a real event.
"""
import json
import sys

sys.path.insert(0, "scripts")
from text_support_audit import CRASH, FIRE, INJURY, HEDGE, _is_negated

# Same 2 KEEP verdicts as eval_text_consistent.json's hand review.
HAND_REVIEWED_KEEP = {
    "10081273": {"crash": False, "fire": True, "injury": False},
    "871031":   {"crash": False, "fire": False, "injury": True},
}
HAND_REVIEWED_UPGRADE_REJECT = {
    "11487942", "11515420", "11735257", "10882196",  # original 4
    "11624858", "11183247",  # v5
}
HAND_REVIEWED_DOWNGRADE_REJECT_LEXICON_GAP = {"11685196"}  # "ran into" -- real collision


def fires(text, pattern):
    for m in pattern.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return m.group(0)
    return None


rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
assert len(rows) == 140

out = []
counts = {"hand_reviewed": 0, "automated": 0}

for r in rows:
    odino = r["odino"]
    text = r["narrative"]
    new_r = dict(r)  # component/defect_type/safety_risk/severity untouched, verbatim

    if odino in HAND_REVIEWED_KEEP:
        v = HAND_REVIEWED_KEEP[odino]
        new_r["crash_described"] = v["crash"]
        new_r["fire_described"] = v["fire"]
        new_r["injury_described"] = v["injury"]
        counts["hand_reviewed"] += 1
    elif odino in HAND_REVIEWED_UPGRADE_REJECT:
        new_r["crash_described"] = False
        new_r["fire_described"] = False
        new_r["injury_described"] = False
        counts["hand_reviewed"] += 1
    elif odino in HAND_REVIEWED_DOWNGRADE_REJECT_LEXICON_GAP:
        crash_d = fires(text, CRASH)
        fire_d = fires(text, FIRE)
        injury_d = fires(text, INJURY)
        new_r["crash_described"] = True  # overridden: real collision, lexicon missed "ran into"
        new_r["fire_described"] = bool(fire_d)
        new_r["injury_described"] = bool(injury_d)
        counts["hand_reviewed"] += 1
    else:
        crash_d = fires(text, CRASH)
        fire_d = fires(text, FIRE)
        injury_d = fires(text, INJURY)
        if injury_d is not None and injury_d.lower() == "hospital" and not crash_d and not fire_d:
            injury_d = None
        new_r["crash_described"] = bool(crash_d)
        new_r["fire_described"] = bool(fire_d)
        new_r["injury_described"] = bool(injury_d)
        counts["automated"] += 1

    out.append(new_r)

assert len(out) == 140
print(f"hand-reviewed rows: {counts['hand_reviewed']}  automated rows: {counts['automated']}  total: {sum(counts.values())}")

# sanity: confirm the 4 official target fields are byte-identical to eval.jsonl
for orig, new in zip(rows, out):
    for f in ["component", "defect_type", "safety_risk", "severity"]:
        assert orig[f] == new[f], f"MISMATCH on {f} for odino={orig['odino']}"
print("confirmed: component/defect_type/safety_risk/severity are byte-identical to eval.jsonl for all 140 rows")

with open("data/processed/eval_v4.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("wrote data/processed/eval_v4.jsonl")
