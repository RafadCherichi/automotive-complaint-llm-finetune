"""Narrow follow-up check (measurement only -- no data file edits, no GPU):
tests whether the medium/high severity confusion is real model error or the
model correctly picking up on injury-sounding text that the official
injured=0 flag missed.

Uses the SAME validated lexicon/negation logic as text_support_audit.py --
specifically isolates the INJURY category only (not crash/fire/control_loss,
since those are exactly what triggered severity=medium in the first place and
aren't the question here).
"""
import json
import sys

sys.path.insert(0, "scripts")
from text_support_audit import INJURY, _is_negated

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def injury_hit(text):
    for m in INJURY.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return m.group(0)
    return None


eval_rows = load_jsonl("data/processed/eval.jsonl")
train_rows = load_jsonl("data/processed/train.jsonl")

v2 = json.load(open("eval/eval_results_v2.json", encoding="utf-8"))
pred_by_odino = {p["odino"]: p["parsed"] for p in v2["finetuned"]["predictions"]}

# --- Part 1: eval rows actual=medium, predicted=high -----------------------
medium_predicted_high = []
for r in eval_rows:
    if r["severity"] != "medium":
        continue
    pred = pred_by_odino.get(r["odino"])
    if pred is None or pred.get("severity") != "high":
        continue
    medium_predicted_high.append(r)

print(f"PART 1: eval rows actual=medium, model predicted=high: n={len(medium_predicted_high)}")
n_injury_hit = 0
part1_hits = []
for r in medium_predicted_high:
    snippet = injury_hit(r["narrative"])
    flag = f"injury flag(NHTSA)={r['injured']}"
    if snippet:
        n_injury_hit += 1
        part1_hits.append((r, snippet))
    print(f"  odino={r['odino']}  {flag}  crash={r['crash']} fire={r['fire']}  "
          f"INJURY_TEXT_HIT={'YES: ' + repr(snippet) if snippet else 'no'}")

print(f"\n  => {n_injury_hit}/{len(medium_predicted_high)} "
      f"({100*n_injury_hit/len(medium_predicted_high):.1f}%) of medium->high-confused rows "
      f"have injury-category language in the text despite injured=0.\n")

# --- Part 2: all train medium rows, same injury check -----------------------
train_medium = [r for r in train_rows if r["severity"] == "medium"]
n_train_injury_hit = 0
part2_hits = []
for r in train_medium:
    snippet = injury_hit(r["narrative"])
    if snippet:
        n_train_injury_hit += 1
        part2_hits.append((r, snippet))

print(f"PART 2: ALL train.jsonl medium rows: n={len(train_medium)}")
print(f"  => {n_train_injury_hit}/{len(train_medium)} "
      f"({100*n_train_injury_hit/len(train_medium):.1f}%) have injury-category language "
      f"in text despite the medium tier being derived without an injury flag "
      f"(injured=0 for all severity=medium rows, by the label-derivation rule).\n")

# sanity check: confirm injured==0 for all severity=medium rows (per label derivation)
non_zero_injured_medium = [r for r in train_medium if r["injured"] != 0]
print(f"  sanity check -- train medium rows with injured!=0: {len(non_zero_injured_medium)} "
      f"(should be 0 if medium always implies injured=0)\n")

# --- Part 3: show full narrative text for every injury-hit case ------------
print("=" * 70)
print("PART 3a: full narrative text -- eval medium->high-confused rows WITH injury hit")
print("=" * 70)
for r, snippet in part1_hits:
    print(f"\n--- odino={r['odino']} ---")
    print(f"narrative: {r['narrative']}")
    print(f"injury match: {snippet!r}")
    print(f"NHTSA flags: crash={r['crash']} fire={r['fire']} injured={r['injured']} deaths={r['deaths']}")
    print(f"model predicted: {pred_by_odino[r['odino']]}")

print("\n" + "=" * 70)
print("PART 3b: full narrative text -- train.jsonl medium rows WITH injury hit")
print("=" * 70)
for r, snippet in part2_hits:
    print(f"\n--- odino={r['odino']} ---")
    print(f"narrative: {r['narrative']}")
    print(f"injury match: {snippet!r}")
    print(f"NHTSA flags: crash={r['crash']} fire={r['fire']} injured={r['injured']} deaths={r['deaths']}")
