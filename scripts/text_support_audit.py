"""Diagnostic audit (measurement only -- no retraining, no data file edits, no GPU):
quantifies what fraction of safety_risk/severity labels are actually supported by the
complaint NARRATIVE TEXT alone, as opposed to only by NHTSA's structured flags
(CRASH/FIRE/INJURED/DEATHS) that the model never sees.

Methodology: Option B from the options-first proposal -- expanded deterministic keyword
detection (4 alarm categories + a "hedge" co-occurring flag for hypothetical/recall
context), extending the same keyword-matching approach already used in
scripts/label_rules.py. Not an LLM: using a model to judge whether text supports a label
would be circular (see docs/learning/04_label_derivation_from_existing_flags.md).

Every pattern below is inspectable by design -- this file IS the methodology, not a
black box around it.

v2 changes, made after a 21-example hand-validation found two real bugs and one design
flaw in v1:
  1. Negation handling added (NegEx-style: a negation word within ~5 words of a match,
     before OR after it, cancels the hit). Fixes "an accident was AVOIDED" -- v1
     false-positived on "accident" with no awareness the sentence says the opposite.
  2. CRASH broadened to catch active "hit <object>" phrasing ("hit the consumer's
     vehicle," "hit another"), not just the passive "hit by" from v1.
  3. hedge changed from an override to a co-occurring flag. v1 let hedge silently
     swallow real hits (a confirmed crash that also mentioned "risk of" elsewhere got
     bucketed as merely "ambiguous," hiding a genuine match). Now: hedge only means
     "ambiguous, flag for review" when it's the ONLY signal that fired. If an alarm
     category ALSO fires, that's reported as a confirmed hit, with hedge noted
     alongside it as a co-occurring flag, not a replacement.

v3 changes, found during the full-scale audit and a follow-up medium/high severity
check (train/eval hand-read, not just the automated pass):
  1. INJURY's bare "burn" restricted to require a nearby person-word -- it was
     matching property/vehicle fire language ("wire burning," "car burned"),
     which belongs to FIRE, not a personal injury.
  2. INJURY's "wound" excluded when followed by "up" -- was matching the "wound
     up" idiom, unrelated to an injury.
  3. Negation now recognizes common contractions ("DON'T," "ISN'T," ...) and
     "nobody"/"no one" -- \bn't\b alone never matches inside a contraction since
     there's no word boundary before the "n."

v4 changes, found during Round 4's training-label-correction hand-review (a much
higher-precision bar than the original audit's "does this sound alarming enough
to question a label" one -- rewriting ground truth needs actual described
events, not near-misses or hypotheticals):
  1. "avoided"/"prevented" added to NEGATION_PRE (previously POST-only) -- catches
     "narrowly AVOIDED an accident" (trigger before the match), not just "an
     accident was AVOIDED" (trigger after). Kept.
  2. CRASH's "hit <object>" gap: a tightening fix was attempted and REVERTED
     after regression-testing showed it broke 6 genuine collision matches for
     every 1 false positive it fixed -- see the CRASH pattern's own comment for
     the full story. Left as-is (known residual limitation, handled by hand
     review on the one row it actually affects, not a pattern change).
  NOTE: two more precision issues surfaced during Round 4 (ADAS/safety-feature
  *names* like "forward collision warning" matching CRASH; hypothetical phrasing
  like "would/can result in," "in order to prevent" not fully covered by HEDGE)
  but were NOT fixed here either -- they were resolved by hand-reviewing the
  affected rows individually instead, since regex was hitting diminishing
  returns (see docs/label-strategy.md, Round 4 section: "keyword matching for
  auditing vs. ground truth"). Left as known residual gaps in this module.

v5 change, found during a forensic v2-vs-v4 recall-drop investigation (see
docs/eval-report.md Section 7's safety_risk recall bullet):
  1. HEDGE's bare "recall" removed -- see HEDGE's own comment for the full
     count/reasoning (7 of 10 affected rows were administrative recall mentions
     with no hypothetical framing; the word added no unique detection value for
     genuine cases, only false-triggered on administrative ones).
  NOTE: this same investigation found two more, separate bugs it did NOT fix:
  (a) the negation window can over-reach across an unrelated clause joined by
  "and" -- "the vehicle did NOT immediately stop AND crashed into..." reads as
  negating "crashed," which it doesn't grammatically negate (train odino
  11387507); (b) CRASH's object-noun list doesn't cover "back(ed)/backup into"
  phrasing (train odino 11702659, "caused me to backup into a mailbox"). Both
  are handled by hand-review overrides in scripts/build_train_v4.py rather than
  a blanket pattern change -- same reasoning as the v4 "hit"-gap revert: a
  general fix to either needs its own regression pass before being trusted
  across the whole dataset, not a same-day patch. Left as known residual gaps.
"""
import re

# --- Category 1: direct injury/harm language ---------------------------------
# v3: "burn" restricted to require a nearby person-word -- bare \bburn matched
# property/vehicle fire language ("wire burning," "car burned") which belongs to
# FIRE, not a personal injury (found via full-scale audit, train odino 11110795
# and 22 others). "wound" excluded when followed by "up" -- was matching the
# "wound up" idiom (odino 10839595), unrelated to an injury.
INJURY = re.compile(
    r"\binjur|\bhurt\b|\bpain\b|hospital|ambulance|\bblind\b|\bill\b|\bsick\b|"
    r"burn(ed|s|ing)?\b(?=.{0,25}\b(hand|arm|leg|skin|face|finger|fingers|body|"
    r"myself|me|driver|passenger|consumer|person|him|her|them)\b)|"
    r"bruis|\bwound(?!\s*up)\w*|\bbleed|unconscious|\bchoke|\bfaint|\bdizzy\b|nausea|\bvomit",
    re.I,
)

# --- Category 2: crash/collision language -------------------------------------
# v2: added active "hit <object>" -- v1 only caught passive "hit by", missing
# "hit the consumer's vehicle" (10024802) and "hit another [vehicle]" (801596).
# Restricted to a concrete object list (not bare "hit \w+") specifically to avoid
# matching idioms like "hit the brakes" / "hit the gas."
# v4 attempted fix, REVERTED: the "hit <object>" gap is a generic .{0,20}
# character span, which can bridge OVER an unrelated intervening noun and
# misattribute a later word as the object -- found during Round 4 hand-review,
# train odino 11500755: "HIT A BUMP THE VEHICLE'S FRONT SUSPENSION..." matched
# as if "vehicle" were hit, when the real object was "a bump" and "the
# vehicle's..." starts an unrelated next clause. Tried replacing the free gap
# with a structured (article/possessive-only) one -- but regression-testing
# that fix against real "hit" collisions found it broke 6 genuine matches
# ("hit the BACK OF another vehicle," "hit a TELEPHONE pole," "hit a PARKED
# car," "hit the GARAGE wall," "hit the SIDE OF the truck," "hit a SIGN then a
# wall") for every 1 false positive it fixed -- ordinary descriptive phrases
# ("the back of," "a parked," compound nouns) don't fit an article-only gap.
# Reverted: the loose gap's real-world false-positive rate is low (this is the
# only case found across ~1,040 rows scanned so far) and a general fix that
# breaks 6x more than it repairs is a worse trade. The one known affected row
# is handled by hand review instead (docs/label-strategy.md, Round 4 section),
# not a blanket pattern change. Left as a documented residual limitation.
CRASH = re.compile(
    r"\bcrash|collis|\bwreck|\bstruck\b|\bstrike\b|hit by|"
    r"\bhit\b.{0,20}(car|vehicle|truck|wall|pole|tree|curb|building|fence|"
    r"person|pedestrian|another)\b|"
    r"\baccidents?\b|rear.?end|t-?boned|totaled|sideswip",
    re.I,
)

# --- Category 3: fire/smoke language ------------------------------------------
FIRE = re.compile(
    r"\bfire\b|\bflame|\bsmoke\b|explod|explos|caught.{0,15}fire",
    re.I,
)

# --- Category 4: critical control-loss / dangerous-failure language ----------
CONTROL_LOSS = re.compile(
    r"lost (all |my |the )?(control|brakes|steering|power)\b|no brakes|"
    r"could not stop|would not stop|unable to stop|uncontrollably|spun out|\bswerv|"
    r"brakes? (fail|went to|going to).{0,15}floor|"
    r"steering.{0,15}(lock|spin|fail)|"
    r"airbag.{0,15}(did n|didn.t|failed to).{0,10}deploy",
    re.I,
)

CATEGORIES = [("injury", INJURY), ("crash", CRASH), ("fire", FIRE), ("control_loss", CONTROL_LOSS)]

# --- Hedge: hypothetical / future-risk / near-miss context -------------------
# A co-occurring flag, not an override (v2) -- see module docstring point 3.
# v5: bare "recall" removed. Found via a forensic recall-vs-v2 comparison: 7 of 10
# rows where "recall" was the ONLY hedge trigger were administrative mentions
# ("were not on recall," "no recall associated with the VIN," a bare listing of
# recall numbers) with no risk/hypothetical framing at all -- the word alone
# doesn't distinguish "this part was recalled" (a fact) from "there's a recall
# because it could catch fire" (a hypothetical). Checked this doesn't lose real
# detection: the one genuine hedge/recall case already in the record (odino
# 10660775's Takata complaint -- "the transmission COULD fail... What a
# NIGHTMARE!") still fires via "could" and "nightmare" independently, so
# "recall" was never doing unique work for a true positive, only false-triggering
# on administrative ones.
HEDGE = re.compile(
    r"\bcould\b|\bmight\b|\bpotential\b|risk of|in case|\bwarned\b|"
    r"\bnearly\b|\balmost\b|"
    r"\bif\b.{0,30}(fail|deploy|crash)|\bnightmare\b",
    re.I,
)

# --- Negation handling (NegEx-style) ------------------------------------------
# Pre-triggers: negation word appears BEFORE the matched phrase ("no accident",
# "not injured", "without incident").
# v3: \bn't\b never matches inside a contraction like "DON'T" -- there's no word
# boundary between the "N" and the preceding letter, so the bare token never
# fires. Fixed by listing common contractions explicitly (found via full-scale
# audit, train odino 10257384: "SHE DON'T HAVE ANY INJURIES" wasn't caught).
# "nobody"/"no one" added too (odino 11746042: "nobody was hurt" wasn't caught).
# v4: "avoided"/"prevented" added to PRE too -- found via Round 4 label-correction
# review, train odino 10948421: "narrowly AVOIDED an accident" has the trigger
# word BEFORE the match, which only NEGATION_POST (after-only) used to check.
# "no/not/contractions" deliberately stay PRE-only -- checking them in the AFTER
# window too would risk new false negatives (e.g. "the crash happened, no doubt"
# would wrongly cancel a real crash if "no" were checked after the match).
NEGATION_PRE = re.compile(
    r"\b(no|not|n't|without|don't|doesn't|didn't|isn't|wasn't|weren't|aren't|"
    r"hasn't|haven't|hadn't|wouldn't|couldn't|shouldn't|won't|can't|cannot|"
    r"nobody|no one|noone|avoided|prevented)\b",
    re.I,
)
# Post-triggers: negation/cancellation word appears AFTER the matched phrase
# ("an accident was AVOIDED", "a crash was PREVENTED"). This is the direction that
# catches 11657901 -- "avoided" comes after "accident," not before it.
NEGATION_POST = re.compile(r"\b(avoided|prevented)\b", re.I)
NEGATION_WINDOW_WORDS = 5


def _words(text):
    return re.findall(r"\S+", text)


def _is_negated(text, start, end):
    """Checks a ~5-word window on both sides of a match for negation triggers."""
    before = " ".join(_words(text[:start])[-NEGATION_WINDOW_WORDS:])
    after = " ".join(_words(text[end:])[:NEGATION_WINDOW_WORDS])
    return bool(NEGATION_PRE.search(before)) or bool(NEGATION_POST.search(after))


def _category_fires(text, pattern):
    """Scans ALL occurrences (not just the first) -- fires if at least one is
    non-negated, since a long narrative can mention the same word both negated and
    for real in different places."""
    for m in pattern.finditer(text):
        if not _is_negated(text, m.start(), m.end()):
            return m.group(0)
    return None


def score_narrative(text):
    """Returns (hits, hedge_matched).
    hits: list of (category, matched_snippet) for non-negated matches only.
    hedge_matched: the hedge match text if present, else None -- always reported
    regardless of whether hits also fired (v2: co-occurring, not overriding)."""
    hits = []
    for name, pattern in CATEGORIES:
        snippet = _category_fires(text, pattern)
        if snippet:
            hits.append((name, snippet))
    hedge_m = HEDGE.search(text)
    return hits, (hedge_m.group(0) if hedge_m else None)


def text_supports_alarm(text):
    hits, _ = score_narrative(text)
    return len(hits) > 0


def classify(text, label_is_alarm):
    """Single source of truth for how a (hits, hedge, label) triple maps to a
    verdict bucket -- used by both the validation-sample puller and the full audit,
    so the two can never silently disagree on what a "confirmed hit" means.

    v2 hedge behavior: hedge alone (no category hit) -> 'hedge_only' (ambiguous,
    needs a human). hedge alongside a real hit -> still a confirmed hit, hedge
    reported as an extra flag, not a bucket of its own.
    """
    hits, hedge = score_narrative(text)
    fired = len(hits) > 0
    if not fired and hedge:
        bucket = "hedge_only"
    elif fired and label_is_alarm:
        bucket = "clear_hit"
    elif not fired and not label_is_alarm:
        bucket = "clear_clean"
    elif fired and not label_is_alarm:
        bucket = "contradiction_label_no_text_yes"
    else:  # not fired and label_is_alarm
        bucket = "contradiction_label_yes_text_no"
    return bucket, hits, hedge


if __name__ == "__main__":
    print("Categories:")
    for name, pat in CATEGORIES:
        print(f"\n{name}:\n  {pat.pattern}")
    print(f"\nhedge (co-occurring flag, not an override):\n  {HEDGE.pattern}")
    print(f"\nnegation pre-triggers (checked before a match):\n  {NEGATION_PRE.pattern}")
    print(f"negation post-triggers (checked after a match):\n  {NEGATION_POST.pattern}")
