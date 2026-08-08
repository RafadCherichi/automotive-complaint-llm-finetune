"""Isolates eval.jsonl rows where the ONLY reason the detector fires at all is the
broadened active "hit <object>" sub-pattern added to CRASH -- not "hit by", not any
other crash trigger, not injury/fire/control_loss. Read-only, no data file edits.
"""
import json
import re
import sys

sys.path.insert(0, "scripts")
from text_support_audit import INJURY, FIRE, CONTROL_LOSS, _is_negated

HIT_X = re.compile(
    r"\bhit\b.{0,20}(car|vehicle|truck|wall|pole|tree|curb|building|fence|person|pedestrian|another)\b",
    re.I,
)
# every OTHER crash trigger, i.e. the CRASH pattern minus the hit-X branch
OTHER_CRASH = re.compile(
    r"\bcrash|collis|\bwreck|\bstruck\b|\bstrike\b|hit by|\baccidents?\b|"
    r"rear.?end|t-?boned|totaled|sideswip",
    re.I,
)


def fires(text, pattern):
    for m in pattern.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return m.group(0), m.start(), m.end()
    return None


eval_rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]

isolated = []
for r in eval_rows:
    text = r["narrative"]
    hit_match = fires(text, HIT_X)
    if not hit_match:
        continue
    if fires(text, OTHER_CRASH) or fires(text, INJURY) or fires(text, FIRE) or fires(text, CONTROL_LOSS):
        continue  # something else also fired -- not isolated
    isolated.append((r, hit_match))

print(f"Isolated count (hit-X is the ONLY trigger in the whole narrative): {len(isolated)}")
print()
for r, (snippet, start, end) in isolated:
    text = r["narrative"]
    ctx_start = max(0, start - 100)
    ctx_end = min(len(text), end + 60)
    prefix = "..." if ctx_start > 0 else ""
    suffix = "..." if ctx_end < len(text) else ""
    print(f"--- odino={r['odino']} ---")
    print(f"matched sentence: {prefix}{text[ctx_start:ctx_end]}{suffix}")
    print(f"exact match: {snippet!r}")
    print(f"label: safety_risk={r['safety_risk']} severity={r['severity']} "
          f"(crash={r['crash']} fire={r['fire']} injured={r['injured']} deaths={r['deaths']})")
    print()
