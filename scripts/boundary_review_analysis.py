"""Measures the "route medium+yes to human review" safety net proposed in
docs/eval-report.md Section 4, against the shipped model's (v2) actual predictions.
No retraining, no GPU -- reuses eval/eval_results_v2.json's already-saved predictions.

Flag rule: predicted severity == "medium" AND predicted safety_risk == "yes"
  -> "boundary case, recommend human review"

Reports:
  1. Of the 18 real high-severity eval examples, how many get flagged by this rule
     (whether or not the model's raw severity label was correct)?
  2. Total review workload: how many of the 140 eval examples get flagged?
  3. As a fraction of everything the model labeled "medium" (the denominator the
     report's target sentence asks for).
"""
import json

eval_rows = {r["odino"]: r for r in (json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8"))}
v2 = json.load(open("eval/eval_results_v2.json", encoding="utf-8"))
preds = {r["odino"]: r for r in v2["finetuned"]["predictions"]}


def norm(s):
    return (s or "").strip().upper() if isinstance(s, str) else ""


n_total = len(eval_rows)
actual_high = [odino for odino, r in eval_rows.items() if r["severity"] == "high"]
n_actual_high = len(actual_high)

flagged = []
predicted_medium = []
for odino, row in eval_rows.items():
    pred = preds[odino]["parsed"]
    if pred is None:
        continue
    pred_severity = norm(pred.get("severity")).lower()
    pred_safety_risk = norm(pred.get("safety_risk")).lower()
    if pred_severity == "medium":
        predicted_medium.append(odino)
        if pred_safety_risk == "yes":
            flagged.append(odino)

flagged_set = set(flagged)

# Of the 18 actual-high examples, how many got flagged (regardless of whether the
# raw severity prediction was correct)?
high_caught = [o for o in actual_high if o in flagged_set]
# Break down where the actual-high examples' predictions actually landed, for context.
high_breakdown = {"low": 0, "medium": 0, "high": 0, "other/invalid": 0}
for o in actual_high:
    pred = preds[o]["parsed"]
    if pred is None:
        high_breakdown["other/invalid"] += 1
        continue
    sev = norm(pred.get("severity")).lower()
    high_breakdown[sev if sev in ("low", "medium", "high") else "other/invalid"] += 1

print(f"eval set size: {n_total}")
print(f"actual severity=high count: {n_actual_high}")
print(f"model predicted severity=medium (any safety_risk): {len(predicted_medium)}")
print(f"flagged for review (predicted medium AND predicted safety_risk=yes): {len(flagged)}")
print()
print(f"review workload: {len(flagged)}/{n_total} = {len(flagged)/n_total:.1%} of the full eval set")
print(f"review workload as share of all predicted-medium complaints: "
      f"{len(flagged)}/{len(predicted_medium)} = {len(flagged)/len(predicted_medium):.1%}")
print()
print(f"of the {n_actual_high} real high-severity cases, where did the model's severity prediction actually land?")
for k, v in high_breakdown.items():
    print(f"  predicted {k}: {v} ({v/n_actual_high:.1%})")
print()
print(f"of the {n_actual_high} real high-severity cases, caught by the medium+yes flag: "
      f"{len(high_caught)} ({len(high_caught)/n_actual_high:.1%})")
if high_caught:
    print(f"  odinos: {high_caught}")
