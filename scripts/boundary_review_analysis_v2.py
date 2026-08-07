"""Option 2: broader review-trigger rule test, against v2's actual saved predictions.
No retraining, no GPU -- reuses eval/eval_results_v2.json.

New rule: flag for review if predicted safety_risk == "yes" AND predicted severity != "high"
(i.e. the model called it a safety risk but didn't call it high -- covers both the
"predicted low" and "predicted medium" miss patterns, not just medium).
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
for odino, row in eval_rows.items():
    pred = preds[odino]["parsed"]
    if pred is None:
        continue
    pred_severity = norm(pred.get("severity")).lower()
    pred_safety_risk = norm(pred.get("safety_risk")).lower()
    if pred_safety_risk == "yes" and pred_severity != "high":
        flagged.append(odino)

flagged_set = set(flagged)
high_caught = [o for o in actual_high if o in flagged_set]

print(f"eval set size: {n_total}")
print(f"actual severity=high count: {n_actual_high}")
print(f"flagged for review (predicted safety_risk=yes AND predicted severity != high): {len(flagged)}")
print()
print(f"review workload: {len(flagged)}/{n_total} = {len(flagged)/n_total:.1%} of the full eval set")
print()
print(f"of the {n_actual_high} real high-severity cases, caught by the broader flag: "
      f"{len(high_caught)} ({len(high_caught)/n_actual_high:.1%})")
print(f"  odinos caught: {high_caught}")

# Specifically inspect the 2 known misses (actual high, predicted low) to see whether
# THEY individually get caught -- this is the direct test of whether "flag for review"
# can work at all for this failure pattern, or whether it needs a different mitigation.
print()
print("the 2 known actual-high misses (predicted severity=low) -- what did the model predict for safety_risk on these specifically?")
for odino in actual_high:
    pred = preds[odino]["parsed"]
    sev = norm(pred.get("severity")).lower() if pred else None
    if sev == "low":
        sr = norm(pred.get("safety_risk")).lower() if pred else None
        caught = odino in flagged_set
        print(f"  odino={odino}: predicted severity={sev}, predicted safety_risk={sr}, caught by broader flag={caught}")
        row = eval_rows[odino]
        print(f"    narrative: {row['narrative'][:200]}")
