"""Full-scale text-support audit (measurement only -- no retraining, no data file
edits, no GPU). Uses text_support_audit.classify()/score_narrative() -- the same,
already hand-validated detector -- as the single source of truth.

Four parts, per the user's scope:
  1. Coverage/contradiction percentages over eval.jsonl (140) and the full
     train.jsonl (900 -- used whole rather than subsampled, since this is a
     programmatic pass, not a manual read; more accurate than a sample).
  2. Broken down by label class: safety_risk yes/no, and severity per tier
     (low/medium/high).
  3. Cross-referenced against eval/eval_results_v2.json's actual finetuned-model
     predictions: of the model's wrong predictions (per dimension), what fraction
     land on rows where the label itself is contradicted or unsupported by text.
  4. Hedge-flagged rows reported as their own bucket, never folded into
     "supported" or "contradicted."
"""
import json
import sys

sys.path.insert(0, "scripts")
from text_support_audit import classify, score_narrative, CATEGORIES

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def audit_rows(rows):
    """Returns per-row records: odino, hits, hedge, fired, plus both label framings."""
    out = []
    for r in rows:
        hits, hedge = score_narrative(r["narrative"])
        out.append({
            "odino": r["odino"],
            "safety_risk": r["safety_risk"],
            "severity": r["severity"],
            "hits": hits,
            "hedge": hedge,
            "fired": len(hits) > 0,
        })
    return out


def bucket_for(rec, label_is_alarm):
    fired, hedge = rec["fired"], rec["hedge"]
    if not fired and hedge:
        return "hedge_only"
    if fired and label_is_alarm:
        return "clear_hit"
    if not fired and not label_is_alarm:
        return "clear_clean"
    if fired and not label_is_alarm:
        return "contradiction_label_no_text_yes"
    return "contradiction_label_yes_text_no"


def pct(n, d):
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else f"{n}/0 (n/a)"


def report_safety_risk(recs, dataset_name):
    print(f"\n### safety_risk breakdown -- {dataset_name} ###")
    for label_val in ("yes", "no"):
        subset = [r for r in recs if r["safety_risk"] == label_val]
        label_is_alarm = label_val == "yes"
        buckets = [bucket_for(r, label_is_alarm) for r in subset]
        n = len(subset)
        if label_val == "yes":
            supported = buckets.count("clear_hit")
            contradicted = buckets.count("contradiction_label_yes_text_no")
        else:
            supported = buckets.count("clear_clean")
            contradicted = buckets.count("contradiction_label_no_text_yes")
        hedge_n = buckets.count("hedge_only")
        print(f"  safety_risk={label_val}  (n={n})")
        print(f"    text supports label:       {pct(supported, n)}")
        print(f"    text contradicts label:    {pct(contradicted, n)}")
        print(f"    hedge-only (ambiguous):    {pct(hedge_n, n)}")


def report_severity(recs, dataset_name):
    print(f"\n### severity breakdown -- {dataset_name} ###")
    for tier in ("low", "medium", "high"):
        subset = [r for r in recs if r["severity"] == tier]
        label_is_alarm = tier in ("medium", "high")
        buckets = [bucket_for(r, label_is_alarm) for r in subset]
        n = len(subset)
        if label_is_alarm:
            supported = buckets.count("clear_hit")
            contradicted = buckets.count("contradiction_label_yes_text_no")
        else:
            supported = buckets.count("clear_clean")
            contradicted = buckets.count("contradiction_label_no_text_yes")
        hedge_n = buckets.count("hedge_only")
        print(f"  severity={tier}  (n={n})")
        print(f"    text supports label:       {pct(supported, n)}")
        print(f"    text contradicts label:    {pct(contradicted, n)}")
        print(f"    hedge-only (ambiguous):    {pct(hedge_n, n)}")


def report_category_coverage(recs, dataset_name):
    print(f"\n### keyword-category coverage -- {dataset_name} (n={len(recs)}) ###")
    for name, _ in CATEGORIES:
        n_fired = sum(1 for r in recs if any(h[0] == name for h in r["hits"]))
        print(f"  {name}: {pct(n_fired, len(recs))}")
    n_hedge = sum(1 for r in recs if r["hedge"])
    print(f"  hedge (any, incl. co-occurring): {pct(n_hedge, len(recs))}")
    n_any = sum(1 for r in recs if r["fired"])
    print(f"  ANY alarm category fired:        {pct(n_any, len(recs))}")


# ---------------------------------------------------------------------------
eval_rows = load_jsonl("data/processed/eval.jsonl")
train_rows = load_jsonl("data/processed/train.jsonl")

eval_recs = audit_rows(eval_rows)
train_recs = audit_rows(train_rows)
by_odino_eval = {r["odino"]: r for r in eval_recs}

print("=" * 70)
print("PART 1 + 2: coverage / contradiction, by dataset and label class")
print("=" * 70)

report_category_coverage(eval_recs, "eval.jsonl (140)")
report_safety_risk(eval_recs, "eval.jsonl (140)")
report_severity(eval_recs, "eval.jsonl (140)")

report_category_coverage(train_recs, "train.jsonl (900, full set)")
report_safety_risk(train_recs, "train.jsonl (900, full set)")
report_severity(train_recs, "train.jsonl (900, full set)")

print("\n" + "=" * 70)
print("PART 3: cross-reference against eval_results_v2.json (finetuned model)")
print("=" * 70)

v2 = json.load(open("eval/eval_results_v2.json", encoding="utf-8"))
preds = v2["finetuned"]["predictions"]
pred_by_odino = {p["odino"]: p["parsed"] for p in preds}

for dim, label_field, alarm_fn in [
    ("safety_risk", "safety_risk", lambda v: v == "yes"),
    ("severity", "severity", lambda v: v in ("medium", "high")),
]:
    wrong = []
    for r in eval_rows:
        odino = r["odino"]
        pred = pred_by_odino.get(odino)
        if pred is None:
            continue
        actual_val = r[label_field]
        pred_val = pred.get(label_field)
        if pred_val != actual_val:
            wrong.append((r, actual_val))

    n_wrong = len(wrong)
    n_contradicted = 0
    n_hedge = 0
    n_supported_but_wrong = 0
    contradicted_examples = []
    for r, actual_val in wrong:
        rec = by_odino_eval[r["odino"]]
        label_is_alarm = alarm_fn(actual_val)
        b = bucket_for(rec, label_is_alarm)
        if b in ("contradiction_label_no_text_yes", "contradiction_label_yes_text_no"):
            n_contradicted += 1
            contradicted_examples.append(r["odino"])
        elif b == "hedge_only":
            n_hedge += 1
        else:
            n_supported_but_wrong += 1

    print(f"\n--- {dim} ---")
    print(f"  model wrong predictions (finetuned, on eval.jsonl): {n_wrong}")
    print(f"  of those, label contradicted-by-text (probable label noise): {pct(n_contradicted, n_wrong)}")
    print(f"  of those, hedge-only / ambiguous text: {pct(n_hedge, n_wrong)}")
    print(f"  of those, label text-supported (real model error): {pct(n_supported_but_wrong, n_wrong)}")
    print(f"  contradicted-label odinos: {contradicted_examples}")

print("\n" + "=" * 70)
print("PART 4: hedge-flagged rows, listed separately (not folded into either bucket)")
print("=" * 70)
hedge_rows_eval = [r for r in eval_recs if r["hedge"]]
print(f"\neval.jsonl hedge-flagged rows: {len(hedge_rows_eval)} / {len(eval_recs)}")
for r in hedge_rows_eval:
    co = " [co-occurring with a confirmed hit]" if r["fired"] else " [hedge is the ONLY signal]"
    print(f"  odino={r['odino']}  safety_risk={r['safety_risk']} severity={r['severity']}  "
          f"hedge={r['hedge']!r}{co}")
