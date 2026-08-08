"""Round 4, Phase 1 (v5 update): builds the final corrected train_v4.jsonl.

Applies exactly what was reviewed and approved:
  - 29+2=31 DOWNGRADE rows (29 from the original two automated review rounds, plus
    2 more (11443985, 10838757) that became eligible after the v5 HEDGE fix
    removed a false "recall" trigger that was blocking them)
  - 9+2=11 UPGRADE rows: 9 from the original 32-row hand-reviewed pool, plus 2
    more (10375275, 10305181) found in a second 10-row candidate pool that only
    existed because the recall-hedge bug had been hiding them from the automated
    scan entirely -- see HAND_REVIEWED_KEEP_V5 below.
  - 23+8=31 UPGRADE candidates hand-reviewed and REJECTED as near-miss/
    hypothetical/feature-name/unrelated-backstory matches (23 from the original
    pool, 8 more from the v5 pool) -- see docs/label-strategy.md's Round 4
    section for the full per-row reasoning.
  - 2 rows (11387507, 11702659) hand-overridden to stay UNCHANGED despite the v5
    fix making them look like downgrade candidates -- both describe a real
    collision, masked by two separate, unrelated bugs (a negation-window
    over-reach and a CRASH lexicon coverage gap), not by the recall-hedge issue
    this pass was fixing. See HAND_REVIEWED_DOWNGRADE_REJECT below.
  - Everything else keeps its original label

Also adds the 3 new atomic fields for the Round 4 7-field schema:
crash_described, fire_described, injury_described.
  - For the 32 hand-reviewed odinos: uses the HAND-REVIEWED verdict, not a fresh
    automated run -- the whole point of hand review was that the automated
    detector was wrong on these specific rows; re-running it here would let the
    same bugs back in through the new fields.
  - For all other 868 rows: uses the (v4-fixed) automated detector directly.
    This is a known, documented limitation -- see docs/label-strategy.md's
    "keyword matching for auditing vs. ground truth" finding. These rows were
    not cheap enough to hand-review at full scale.

Does NOT touch eval.jsonl. Writes data/processed/train_v4.jsonl (new file --
train.jsonl, the v3 dataset, is left in place untouched for reference).
"""
import json
import sys

sys.path.insert(0, "scripts")
from text_support_audit import CRASH, FIRE, INJURY, HEDGE, _is_negated

# Hand-reviewed verdicts for the 32-row UPGRADE candidate pool. Only these 9 are
# real, confirmed events (all fire/crash -- none had a confirmed real injury).
HAND_REVIEWED_KEEP = {
    "706927":  {"crash": False, "fire": True,  "injury": False},  # smoke at steering column, real
    "10048316":{"crash": True,  "fire": False, "injury": False},  # struck a mud flap, real
    "11737629":{"crash": False, "fire": True,  "injury": False},  # smoke + shattered window, real
    "11555993":{"crash": False, "fire": True,  "injury": False},  # mirror exploded, real
    "10641082":{"crash": False, "fire": True,  "injury": False},  # sunroof exploded, real
    "11618407":{"crash": False, "fire": True,  "injury": False},  # smoke into cabin, real
    "10129605":{"crash": False, "fire": True,  "injury": False},  # smoke incident in garage, real
    "11489761":{"crash": False, "fire": True,  "injury": False},  # white smoke from tailpipe, real
    "10586370":{"crash": False, "fire": True,  "injury": False},  # window exploded, real
}
# The other 23 of the 32 (21 clear rejects + 2 borderline, both rejected):
# hand-review confirmed the matched language was a near-miss, hypothetical, a
# safety-feature NAME (e.g. "collision warning"), or -- for the 2 borderline
# cases -- real but not describing a genuine current event for this complaint.
# All get crash/fire/injury_described = False (label unchanged, matching what
# NHTSA's own flags already said).
HAND_REVIEWED_REJECT = {
    "10553016", "10467171", "10430388", "11000349", "11615319", "11432039",
    "11354507", "10401870", "836172", "10658913", "725821", "10555894",
    "11500755", "780471", "10564570", "10102395", "10679567", "10553686",
    "11687984", "845148", "10206724", "11184847", "732229",
}
assert len(HAND_REVIEWED_KEEP) == 9
assert len(HAND_REVIEWED_REJECT) == 23
HAND_REVIEWED_ALL = set(HAND_REVIEWED_KEEP) | HAND_REVIEWED_REJECT
assert len(HAND_REVIEWED_ALL) == 32

# v5 addition: a forensic v2-vs-v4 recall-drop investigation found 8 train rows
# where HEDGE's now-removed bare "recall" trigger was blocking a downgrade (see
# text_support_audit.py's v5 changelog). Re-scanning post-fix, 4 are exempted
# regardless (injured>0) and 2 (11443985, 10838757) genuinely and correctly
# downgrade now. The other 2 surfaced SEPARATE bugs that would cause a WRONG
# downgrade if left to the automated path, so they're hand-overridden to stay
# unchanged here rather than silently mis-corrected:
#   - 11387507: real crash ("...DID NOT IMMEDIATELY STOP AND CRASHED INTO THE
#     REAR OF A SECOND VEHICLE") -- CRASH doesn't fire because the negation
#     window's "NOT" (negating "stop," a different clause) over-reaches across
#     the "AND" into "crashed." A negation-window bug, not label noise.
#   - 11702659: real minor collision ("caused me to backup into a mailbox") --
#     CRASH's object-noun list doesn't cover "back(ed)/backup into" phrasing. A
#     lexicon coverage gap, not label noise.
# Neither underlying bug is fixed here -- both need their own regression pass
# before a blanket change can be trusted (same reasoning as the v4 "hit"-gap
# revert) -- so these 2 rows are hand-overridden instead.
HAND_REVIEWED_DOWNGRADE_REJECT = {"11387507", "11702659"}

# v5 addition, part 2: removing the "recall" hedge trigger also unblocked 10
# rows that were previously UPGRADE candidates (label=no, but a real category
# fired) sitting in the hedge-ambiguous bucket -- these were never part of the
# original 32-row pool since "recall" was hiding them from the automated scan
# entirely. Hand-reviewed individually, same standard as the original 32:
HAND_REVIEWED_KEEP_V5 = {
    "10375275": {"crash": False, "fire": False, "injury": True},  # "makes me SICK" -- real, stated as fact, same standard as odino 871031
    "10305181": {"crash": True,  "fire": False, "injury": False}, # "FAILED TO STOP...GOT INTO AN ACCIDENT" -- real, dated
}
# The other 8: hypothetical/future-risk framing ("my concern is," "how many
# more...have to be," "can result in," "would be responsible for"), unrelated
# backstory (a previous accident mentioned only as context for a recall-delay
# complaint), or a genuinely negated case a too-short negation window missed
# ("I was NOT in any danger or in a crash" -- "crash" is 7 words after "NOT,"
# past the 5-word window; a third negation-window bug, opposite direction from
# 11387507's over-reach, also not fixed here).
HAND_REVIEWED_REJECT_V5 = {
    "11694212", "11586899", "10944582", "11580021",
    "11619586", "10703093", "11396670", "10595204",
}
assert len(HAND_REVIEWED_KEEP_V5) == 2
assert len(HAND_REVIEWED_REJECT_V5) == 8

# fold the v5 additions into the same KEEP/ALL sets the main loop already checks
HAND_REVIEWED_KEEP.update(HAND_REVIEWED_KEEP_V5)
HAND_REVIEWED_ALL |= set(HAND_REVIEWED_KEEP_V5) | HAND_REVIEWED_REJECT_V5


def fires(text, pattern):
    for m in pattern.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return m.group(0)
    return None


rows = [json.loads(l) for l in open("data/processed/train.jsonl", encoding="utf-8")]
assert len(rows) == 900

out = []
counts = {"downgrade": 0, "upgrade_handreviewed": 0, "exempted": 0,
          "downgrade_hedge_left": 0, "upgrade_hedge_left": 0, "unchanged": 0}

for r in rows:
    odino = r["odino"]
    text = r["narrative"]
    new_r = dict(r)

    if odino in HAND_REVIEWED_ALL:
        if odino in HAND_REVIEWED_KEEP:
            verdict = HAND_REVIEWED_KEEP[odino]
            new_r["crash_described"] = verdict["crash"]
            new_r["fire_described"] = verdict["fire"]
            new_r["injury_described"] = verdict["injury"]
            new_r["safety_risk"] = "yes"
            new_r["severity"] = "high" if verdict["injury"] else "medium"
            new_r["label_source"] = "text_corrected_upgrade_handreviewed"
            counts["upgrade_handreviewed"] += 1
        else:
            new_r["crash_described"] = False
            new_r["fire_described"] = False
            new_r["injury_described"] = False
            new_r["label_source"] = "original_handreviewed_reject"
            counts["unchanged"] += 1
        out.append(new_r)
        continue

    crash_d = fires(text, CRASH)
    fire_d = fires(text, FIRE)
    injury_d = fires(text, INJURY)
    hedge_m = HEDGE.search(text)
    # hospital-only exclusion, scoped to this correction pass only (per approved
    # decision) -- NOT a change to the shared INJURY regex, so the already
    # published eval-report.md Section 6 numbers stay reproducible.
    if injury_d is not None and injury_d.lower() == "hospital" and not crash_d and not fire_d:
        injury_d = None
    text_alarm = bool(crash_d or fire_d or injury_d)
    current_alarm = r["safety_risk"] == "yes"

    new_r["crash_described"] = bool(crash_d)
    new_r["fire_described"] = bool(fire_d)
    new_r["injury_described"] = bool(injury_d)

    if odino in HAND_REVIEWED_DOWNGRADE_REJECT:
        # real crash/collision described in text; automated pipeline would
        # otherwise downgrade due to the two separate bugs noted above -- override
        new_r["crash_described"] = True
        new_r["label_source"] = "original_handreviewed_downgrade_reject_lexicon_or_negation_gap"
        counts["unchanged"] += 1
    elif current_alarm and not text_alarm and not hedge_m:
        if r["injured"] > 0 or r["deaths"] > 0:
            new_r["label_source"] = "original_exempted_high_stakes"
            counts["exempted"] += 1
        else:
            new_r["safety_risk"] = "no"
            new_r["severity"] = "low"
            new_r["label_source"] = "text_corrected_downgrade"
            counts["downgrade"] += 1
    elif current_alarm and not text_alarm and hedge_m:
        new_r["label_source"] = "original_downgrade_hedge_ambiguous"
        counts["downgrade_hedge_left"] += 1
    elif not current_alarm and text_alarm and hedge_m:
        new_r["label_source"] = "original_upgrade_hedge_ambiguous"
        counts["upgrade_hedge_left"] += 1
    else:
        new_r["label_source"] = "original_text_consistent"
        counts["unchanged"] += 1
    out.append(new_r)

assert len(out) == 900
print("Row-disposition counts:")
for k, v in counts.items():
    print(f"  {k}: {v}")
print(f"  (unchanged total includes 23 hand-reviewed rejects)")
print(f"  TOTAL: {sum(counts.values())}")
print(f"\nNet corrected: {counts['downgrade'] + counts['upgrade_handreviewed']} / 900 "
      f"({100*(counts['downgrade']+counts['upgrade_handreviewed'])/900:.2f}%)")

with open("data/processed/train_v4.jsonl", "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("\nWrote data/processed/train_v4.jsonl (900 rows, 7-field schema + label_source audit trail)")
