"""Targeted rebuild of train.jsonl per the Phase 3 diagnosis: the fine-tuned model
learned "crash/fire mentioned -> medium" as a shortcut and under-weighted injury-only
(no crash/fire) language for severity=high, even when injury text was explicit.

This is a WITHIN-CLASS fix, not a dataset-wide rebalance:
  - safety_risk:yes total stays at 256/800 (32%) -- same overall balance as before,
    within Section 5a's locked 30-35% band.
  - low (544) is completely untouched -- not part of the diagnosed problem.
  - Only the composition *within* the positive class shifts: sub-pattern A
    (injury/death present, crash=False, fire=False) is added specifically to the
    "high" tier, and "medium" shrinks by the same count to hold the totals fixed.

Before: high=99 (28 sub-pattern-A : 71 sub-pattern-B), medium=157.
After:  high=156 (85 sub-pattern-A : 71 sub-pattern-B), medium=100.
Sub-pattern A goes from 28% to 54.5% of the high tier -- roughly balanced against
sub-pattern B instead of being a small minority drowned out by crash/fire language.

eval.jsonl is never read or written here.
"""
import json
import random

from component_taxonomy import bucket_component
from label_rules import safety_risk, severity, defect_type

RANDOM_SEED = 42
N_TO_ADD = 57  # 28 -> 85 sub-pattern-A examples in the high tier

rng = random.Random(RANDOM_SEED)

train = [json.loads(l) for l in open("data/processed/train.jsonl", encoding="utf-8")]
eval_odinos = {json.loads(l)["odino"] for l in open("data/processed/eval.jsonl", encoding="utf-8")}

candidates = json.load(open("data/raw/injury_only_high_candidates.json", encoding="utf-8"))
print(f"candidate pool (sub-pattern A, not already in train/eval): {len(candidates):,}")
assert not (set(c["odino"] for c in candidates) & eval_odinos), "candidate pool leaked eval ODINOs"

sampled = rng.sample(candidates, N_TO_ADD)


def to_record(c):
    component_raw_list = sorted(c["component_raw"])
    primary_raw_top_level = component_raw_list[0].split(":")[0].strip()
    joined_raw = " ".join(component_raw_list)
    return {
        "odino": c["odino"],
        "cmplid": c["cmplid"],
        "make": c["make"],
        "model": c["model"],
        "year": c["year"],
        "narrative": c["narrative"],
        "component": bucket_component(primary_raw_top_level),
        "component_raw": ",".join(component_raw_list),
        "defect_type": defect_type(joined_raw, c["narrative"], c["fire"]),
        "safety_risk": safety_risk(c["crash"], c["fire"], c["injured"], c["deaths"]),
        "severity": severity(c["crash"], c["fire"], c["injured"], c["deaths"]),
        "crash": c["crash"],
        "fire": c["fire"],
        "injured": c["injured"],
        "deaths": c["deaths"],
    }


new_high_a_records = [to_record(c) for c in sampled]
assert all(r["severity"] == "high" and not r["crash"] and not r["fire"] for r in new_high_a_records)

medium_records = [r for r in train if r["severity"] == "medium"]
non_medium_records = [r for r in train if r["severity"] != "medium"]
assert len(medium_records) == 157

kept_medium = rng.sample(medium_records, len(medium_records) - N_TO_ADD)
dropped_medium = [r for r in medium_records if r not in kept_medium]

new_train = non_medium_records + kept_medium + new_high_a_records
rng.shuffle(new_train)

assert len(new_train) == 800, len(new_train)
assert len(set(r["odino"] for r in new_train)) == 800, "duplicate ODINO in rebuilt train set"
assert not (set(r["odino"] for r in new_train) & eval_odinos), "rebuilt train set leaked eval ODINOs"

with open("data/processed/train.jsonl", "w", encoding="utf-8") as f:
    for r in new_train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
sev_dist = Counter(r["severity"] for r in new_train)
high = [r for r in new_train if r["severity"] == "high"]
sub_a = sum(1 for r in high if not r["crash"] and not r["fire"])
sub_b = len(high) - sub_a

print()
print("REBUILT train.jsonl:")
print(f"  total: {len(new_train)}")
print(f"  safety_risk=yes: {sum(1 for r in new_train if r['safety_risk']=='yes')} ({sum(1 for r in new_train if r['safety_risk']=='yes')/len(new_train):.1%})")
print(f"  low: {sev_dist['low']}  medium: {sev_dist['medium']}  high: {sev_dist['high']}")
print(f"  high sub-pattern A (injury-only, no crash/fire): {sub_a} ({sub_a/len(high):.1%} of high)")
print(f"  high sub-pattern B (crash/fire + injury/death): {sub_b} ({sub_b/len(high):.1%} of high)")
print(f"  dropped {len(dropped_medium)} medium examples to hold total/ratio fixed")
