# 4. Label Derivation: Deriving Labels from Existing Structured Fields, Not Manual or Synthetic Labeling

## (1) The concept

**Supervised learning** means training a model on labeled examples — input paired with
the correct answer — so it learns to predict the answer for new, unseen inputs. Labels
conventionally come from one of two places: **manual annotation** (a human reads each
example and assigns a label — accurate but slow, expensive, and prone to inconsistency
between different annotators), or **synthetic/LLM-generated labels** (another AI model
generates the labels quickly and cheaply, but risks being wrong, biased, or just
hallucinated, since it's a guess by a different model, not verified ground truth).

There's a third option, used here: **deriving labels from other real structured data
already collected about the same event.** If a database already records "this complaint
involved a crash: yes" as a verified fact, there's no need for a human or an AI to
separately *guess* whether the situation sounds dangerous — the fact is already known
and can be turned into a label with simple, deterministic logic.

## (2) How this project uses it

NHTSA complaint records include free-text narrative *plus* structured columns:
`CRASH`, `FIRE`, `INJURED`, `DEATHS` (Y/N and counts), and `COMPDESC` (a structured
component-category field). `scripts/label_rules.py` derives two of the four target
fields directly and deterministically from these:

```python
def safety_risk(crash, fire, injured, deaths):
    return "yes" if (crash or fire or injured > 0 or deaths > 0) else "no"

def severity(crash, fire, injured, deaths):
    if injured > 0 or deaths > 0: return "high"
    if crash or fire: return "medium"
    return "low"
```

`component` is derived by bucketing NHTSA's own `COMPDESC` field into a taxonomy built
from real frequency counts (`scripts/component_taxonomy.py`, `scripts/tally_components.py`)
— not invented categories. `defect_type` is the one field with no direct NHTSA column,
so it's matched via deterministic keyword rules primarily against `COMPDESC`
(`scripts/label_rules.py`'s `defect_type()` function) — still a fixed, inspectable rule
set, not an LLM guessing.

This kept every training label traceable to a real, verified fact about the complaint
(an NHTSA-recorded flag or category), with zero manual annotation and zero synthetic
generation — directly satisfying the "existing data only, zero budget" constraint from
`docs/blueprint.md`.

## (3) When the alternative would win

- **Manual labeling** is the right (or only) choice when the target concept genuinely
  can't be derived from any existing structured data — e.g. a subjective judgment like
  "how frustrated does this customer sound," which has no corresponding database column
  to derive it from.
- **LLM-generated labels** make sense when there's no reliable structured signal at all
  to derive from, speed/cost matters more than verification, and some label noise is an
  acceptable tradeoff. This project deliberately avoided that path so every training
  label stays grounded in a verified real-world fact instead of another model's guess.
