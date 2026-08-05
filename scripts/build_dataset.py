"""Phase 1 dataset build (blueprint.md Section 5 / 5a / 6).

Streams NHTSA's full FLAT_CMPL flat file once (~1.5GB uncompressed, read in chunks so
memory stays bounded -- this machine often has only ~2GB RAM free). For every vehicle
complaint row, derives the four target fields from NHTSA's own structured columns:

  component    <- COMPDESC, first-listed value, bucketed into a fixed taxonomy
  defect_type  <- rule-based keyword match over COMPDESC + narrative
  safety_risk  <- crash/fire/injury/death flags
  severity     <- tiered from the same flags

No hand labeling, no LLM-generated labels -- everything here is a deterministic
function of real NHTSA metadata.

Sampling: two reservoirs (safety_risk yes / no), keyed by ODINO (NHTSA's per-complaint
id -- one complaint can have multiple rows, one per listed component). This scans the
*whole* file rather than hand-picking makes/models, so there's no selection bias, while
keeping memory bounded to the reservoir size rather than the full corpus. Multiple rows
sharing an ODINO are merged into one record; every raw COMPDESC value seen for that
complaint is kept as metadata (component_raw) even though only the first is the training
target, per Section 5a.

Class balance: real-world safety-flagged complaints are ~3% of the corpus. Per Section
5a we deliberately oversample: pull (effectively) all matching positives, fill the rest
with negatives, target ~32% safety_risk:yes in both the training and eval sets (eval is
also stratified away from the natural 97/3 split so precision/recall on the rare class
is measurable, not noise).
"""
import csv
import io
import json
import random
import zipfile

from component_taxonomy import bucket_component
from label_rules import safety_risk, severity, defect_type

RANDOM_SEED = 42
ZIP_PATH = "data/raw/FLAT_CMPL.zip"
INNER_NAME = "FLAT_CMPL.txt"
MIN_NARRATIVE_LEN = 40

# Column positions (0-indexed) in FLAT_CMPL.txt -- see scripts/nhtsa_schema.py for the
# full 51-column layout. Reading row-by-row with the csv module instead of pandas: this
# machine has been observed with under 400MB free RAM, and pandas' C parser buffers more
# aggressively per chunk than a plain line-by-line read of a tab-split file needs.
IDX_CMPLID, IDX_ODINO, IDX_MAKE, IDX_MODEL, IDX_YEAR = 0, 1, 3, 4, 5
IDX_CRASH, IDX_FIRE, IDX_INJURED, IDX_DEATHS, IDX_COMPDESC = 6, 8, 9, 10, 11
IDX_CDESCR, IDX_PROD_TYPE = 19, 45


def field(row, i):
    return row[i] if i < len(row) else ""

NEG_RESERVOIR_CAP = 5000  # negatives are common; keep a bounded, uniformly-sampled pool

TRAIN_TOTAL = 800
EVAL_TOTAL = 140
POS_RATIO = 0.32  # blueprint.md Section 5a: target ~30-35% safety_risk:yes

TRAIN_POS = round(TRAIN_TOTAL * POS_RATIO)
TRAIN_NEG = TRAIN_TOTAL - TRAIN_POS
EVAL_POS = round(EVAL_TOTAL * POS_RATIO)
EVAL_NEG = EVAL_TOTAL - EVAL_POS

rng = random.Random(RANDOM_SEED)


class Reservoir:
    """Algorithm-R reservoir sampling keyed by ODINO, with merge-on-duplicate-key so
    multi-component complaints (same ODINO, several COMPDESC rows) accumulate their
    raw component list instead of being treated as separate candidates."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.pool = {}
        self.order = []
        self.n_seen = 0

    def offer(self, key, record):
        if key in self.pool:
            self.pool[key]["component_raw"] |= record["component_raw"]
            return
        self.n_seen += 1
        if self.capacity is None or len(self.pool) < self.capacity:
            self.pool[key] = record
            self.order.append(key)
        else:
            j = rng.randrange(self.n_seen)
            if j < self.capacity:
                del self.pool[self.order[j]]
                self.pool[key] = record
                self.order[j] = key


positives = Reservoir(capacity=None)  # rare enough to keep them all
negatives = Reservoir(capacity=NEG_RESERVOIR_CAP)

total_rows = 0
vehicle_rows = 0
kept_rows = 0

with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(INNER_NAME) as fb:
        f = io.TextIOWrapper(fb, encoding="cp1252", errors="replace", newline="")
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            total_rows += 1
            if field(row, IDX_PROD_TYPE) != "V":
                continue
            vehicle_rows += 1

            cdescr = field(row, IDX_CDESCR).strip()
            odino = field(row, IDX_ODINO).strip()
            if len(cdescr) < MIN_NARRATIVE_LEN or not odino:
                continue
            kept_rows += 1

            injured_raw = field(row, IDX_INJURED).strip()
            deaths_raw = field(row, IDX_DEATHS).strip()
            injured = int(injured_raw) if injured_raw.isdigit() else 0
            deaths = int(deaths_raw) if deaths_raw.isdigit() else 0
            crash = field(row, IDX_CRASH).strip().upper() == "Y"
            fire = field(row, IDX_FIRE).strip().upper() == "Y"
            pos = crash or fire or injured > 0 or deaths > 0

            record = {
                "odino": odino,
                "cmplid": field(row, IDX_CMPLID).strip(),
                "make": field(row, IDX_MAKE).strip(),
                "model": field(row, IDX_MODEL).strip(),
                "year": field(row, IDX_YEAR).strip(),
                "crash": crash,
                "fire": fire,
                "injured": injured,
                "deaths": deaths,
                "narrative": cdescr,
                "component_raw": {field(row, IDX_COMPDESC).strip()},
            }
            (positives if pos else negatives).offer(odino, record)

print(f"total rows scanned: {total_rows:,}")
print(f"vehicle rows (PROD_TYPE=V): {vehicle_rows:,}")
print(f"rows passing narrative-length filter: {kept_rows:,}")
print(f"unique positive complaints (safety_risk=yes) found: {positives.n_seen:,}")
print(f"unique negative complaints seen (reservoir cap {NEG_RESERVOIR_CAP:,}): {negatives.n_seen:,}")
print()

need_pos = TRAIN_POS + EVAL_POS
need_neg = TRAIN_NEG + EVAL_NEG
pos_keys = list(positives.pool.keys())
neg_keys = list(negatives.pool.keys())

if len(pos_keys) < need_pos:
    print(f"WARNING: only {len(pos_keys)} positive complaints available, needed {need_pos}. "
          f"Using all available positives; train/eval positive counts will be scaled down.")
if len(neg_keys) < need_neg:
    print(f"WARNING: only {len(neg_keys)} negative complaints available, needed {need_neg}.")

sel_pos = rng.sample(pos_keys, min(need_pos, len(pos_keys)))
sel_neg = rng.sample(neg_keys, min(need_neg, len(neg_keys)))
rng.shuffle(sel_pos)
rng.shuffle(sel_neg)

# keep the actual train/eval positive split proportional to what's available
actual_train_pos = round(len(sel_pos) * TRAIN_POS / need_pos) if need_pos else 0
actual_train_neg = round(len(sel_neg) * TRAIN_NEG / need_neg) if need_neg else 0

train_keys = sel_pos[:actual_train_pos] + sel_neg[:actual_train_neg]
eval_keys = sel_pos[actual_train_pos:] + sel_neg[actual_train_neg:]
rng.shuffle(train_keys)
rng.shuffle(eval_keys)


def to_record(key):
    src = positives.pool.get(key) or negatives.pool.get(key)
    component_raw_list = sorted(src["component_raw"])
    primary_raw_top_level = component_raw_list[0].split(":")[0].strip()
    joined_raw = " ".join(component_raw_list)
    return {
        "odino": src["odino"],
        "cmplid": src["cmplid"],
        "make": src["make"],
        "model": src["model"],
        "year": src["year"],
        "narrative": src["narrative"],
        "component": bucket_component(primary_raw_top_level),
        "component_raw": ",".join(component_raw_list),
        "defect_type": defect_type(joined_raw, src["narrative"], src["fire"]),
        "safety_risk": safety_risk(src["crash"], src["fire"], src["injured"], src["deaths"]),
        "severity": severity(src["crash"], src["fire"], src["injured"], src["deaths"]),
        "crash": src["crash"],
        "fire": src["fire"],
        "injured": src["injured"],
        "deaths": src["deaths"],
    }


train_records = [to_record(k) for k in train_keys]
eval_records = [to_record(k) for k in eval_keys]

with open("data/processed/train.jsonl", "w", encoding="utf-8") as f:
    for r in train_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open("data/processed/eval.jsonl", "w", encoding="utf-8") as f:
    for r in eval_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"train.jsonl: {len(train_records)} rows "
      f"({sum(r['safety_risk']=='yes' for r in train_records)} safety_risk=yes, "
      f"{sum(r['safety_risk']=='yes' for r in train_records)/len(train_records):.1%})")
print(f"eval.jsonl:  {len(eval_records)} rows "
      f"({sum(r['safety_risk']=='yes' for r in eval_records)} safety_risk=yes, "
      f"{sum(r['safety_risk']=='yes' for r in eval_records)/len(eval_records):.1%})")

print()
print("defect_type distribution (train + eval combined):")
from collections import Counter
dist = Counter(r["defect_type"] for r in train_records + eval_records)
total = len(train_records) + len(eval_records)
for label, n in dist.most_common():
    flag = "  <-- LARGEST BUCKET" if label == dist.most_common(1)[0][0] else ""
    print(f"  {n:>4} ({n/total:>5.1%})  {label}{flag}")
if dist.most_common(1)[0][0] == "OTHER":
    print("\nFLAG: OTHER is the largest defect_type bucket -- taxonomy needs another pass before locking.")

print()
print("severity distribution (train + eval combined):")
sev_dist = Counter(r["severity"] for r in train_records + eval_records)
for label, n in sev_dist.most_common():
    print(f"  {n:>4} ({n/total:>5.1%})  {label}")

print()
print("component distribution (train + eval combined):")
comp_dist = Counter(r["component"] for r in train_records + eval_records)
for label, n in comp_dist.most_common():
    print(f"  {n:>4} ({n/total:>5.1%})  {label}")
