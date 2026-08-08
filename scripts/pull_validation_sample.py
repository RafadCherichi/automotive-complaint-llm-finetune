"""Pulls a stratified ~20-example validation sample from eval.jsonl for hand-checking
the text_support_audit.py detector, per the user's request: not just clear hits/misses,
but boundary cases where the detector is most likely to be wrong.

Uses text_support_audit.classify() as the single source of truth for bucketing, so this
script and the eventual full-scale audit can never silently disagree on what a bucket
means (v1 had its own separate, now-incorrect hedge-override logic inline here).

Read-only: does not modify eval.jsonl or train.jsonl, no model, no GPU.
"""
import json
import random

from text_support_audit import classify

eval_rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
rng = random.Random(11)


def label_is_alarm(r):
    return r["safety_risk"] == "yes" or r["severity"] in ("medium", "high")


buckets = {
    "clear_hit": [],
    "clear_clean": [],
    "contradiction_label_no_text_yes": [],
    "contradiction_label_yes_text_no": [],
    "hedge_only": [],
}
by_odino = {}

for r in eval_rows:
    bucket, hits, hedge = classify(r["narrative"], label_is_alarm(r))
    buckets[bucket].append((r, hits, hedge))
    by_odino[r["odino"]] = (bucket, r, hits, hedge)

print("Bucket sizes (whole 140-example eval set):")
for name, items in buckets.items():
    print(f"  {name}: {len(items)}")
print()

# Known borderline / detector-miss candidates (like the control-arm example already
# flagged) -- narratives describing a failure "while driving/operating" using words
# outside the current lexicon. Manually curated by scanning clear_clean for plausible
# near-misses, since this pattern is exactly the kind of paraphrase the keyword
# approach is weakest on -- not reliably auto-detectable by the same keyword method
# being audited.
borderline_candidates_odinos = []
for r, hits, hedge in buckets["clear_clean"] + buckets["hedge_only"]:
    text_u = r["narrative"].upper()
    if ("WHILE DRIVING" in text_u or "WHILE OPERATING" in text_u or "WHILE TRAVELING" in text_u) and \
       any(w in text_u for w in ["DETACHED", "FELL OFF", "BROKE", "SEPARATED", "GAVE OUT", "SNAPPED"]):
        borderline_candidates_odinos.append(r["odino"])

print(f"borderline candidates found (failure word + 'while driving', no keyword hit): {len(borderline_candidates_odinos)}")
print(f"  odinos: {borderline_candidates_odinos}")
print()

sample = []
sample += [("clear_hit", x) for x in rng.sample(buckets["clear_hit"], min(4, len(buckets["clear_hit"])))]
sample += [("clear_clean", x) for x in rng.sample(buckets["clear_clean"], min(4, len(buckets["clear_clean"])))]
sample += [("contradiction_label_no_text_yes", x) for x in rng.sample(
    buckets["contradiction_label_no_text_yes"], min(5, len(buckets["contradiction_label_no_text_yes"])))]
sample += [("contradiction_label_yes_text_no", x) for x in rng.sample(
    buckets["contradiction_label_yes_text_no"], min(4, len(buckets["contradiction_label_yes_text_no"])))]
sample += [("hedge_only", x) for x in rng.sample(buckets["hedge_only"], min(3, len(buckets["hedge_only"])))]

for o in borderline_candidates_odinos[:3]:
    _, r, hits, hedge = by_odino[o]
    sample.append(("borderline_detector_miss", (r, hits, hedge)))

rng.shuffle(sample)

print(f"=== VALIDATION SAMPLE ({len(sample)} examples) ===\n")
for bucket_name, (r, hits, hedge) in sample:
    print(f"--- odino={r['odino']}  [bucket: {bucket_name}] ---")
    print(f"narrative: {r['narrative']}")
    print(f"label: safety_risk={r['safety_risk']} severity={r['severity']} "
          f"(crash={r['crash']} fire={r['fire']} injured={r['injured']} deaths={r['deaths']})")
    print(f"detector hits: {hits if hits else 'NONE'}")
    print(f"hedge flag: {hedge if hedge else 'none'}"
          + ("  [co-occurring with a confirmed hit]" if hedge and hits else ""))
    print()
