"""Additive restore of the medium severity tier -- NOT a rebalance.

Context: the v2 rebuild (injury-only high fix) swapped 57 medium examples out to make
room for 57 new high-severity examples, cutting medium from 157 to 100. That fix worked
(severity=high accuracy 0% -> 88.9%), but 100 examples fell below the ~200-500-per-class
range that's the standard target for structured extraction QLoRA fine-tunes on a 7-8B
model, and the eval showed medium accuracy collapsed to 14.8% partly as a result.

This script ONLY ADDS new medium-tier examples on top of the existing 800 rows -- it
does not remove or touch anything, unlike the v2 rebuild which swapped. low (544) and
high (156) are left completely alone; only medium grows.

"Medium pattern" = crash=True or fire=True, injured=0, deaths=0 (exactly the severity
derivation rule for "medium"). Candidates are pulled from the full flat file, excluding
anything already in train.jsonl or eval.jsonl.
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
TARGET_MEDIUM_TOTAL = 200  # bottom of the 200-250 target band -- minimum sufficient restore

IDX_CMPLID, IDX_ODINO, IDX_MAKE, IDX_MODEL, IDX_YEAR = 0, 1, 3, 4, 5
IDX_CRASH, IDX_FIRE, IDX_INJURED, IDX_DEATHS, IDX_COMPDESC = 6, 8, 9, 10, 11
IDX_CDESCR, IDX_PROD_TYPE = 19, 45


def field(row, i):
    return row[i] if i < len(row) else ""


rng = random.Random(RANDOM_SEED)

train = [json.loads(l) for l in open("data/processed/train.jsonl", encoding="utf-8")]
eval_rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
used_odinos = {r["odino"] for r in train} | {r["odino"] for r in eval_rows}
print(f"already-used ODINOs (train+eval, excluded from candidates): {len(used_odinos)}")

current_medium = sum(1 for r in train if r["severity"] == "medium")
n_needed = TARGET_MEDIUM_TOTAL - current_medium
print(f"current medium count: {current_medium}  |  target: {TARGET_MEDIUM_TOTAL}  |  need to add: {n_needed}")
assert n_needed > 0, "medium is already at or above target -- nothing to add"

candidates = {}

total_rows = 0
vehicle_rows = 0

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
            if len(cdescr) < MIN_NARRATIVE_LEN or not odino or odino in used_odinos:
                continue

            crash = field(row, IDX_CRASH).strip().upper() == "Y"
            fire = field(row, IDX_FIRE).strip().upper() == "Y"
            if not (crash or fire):
                continue  # need crash or fire for "medium"

            injured_raw = field(row, IDX_INJURED).strip()
            deaths_raw = field(row, IDX_DEATHS).strip()
            injured = int(injured_raw) if injured_raw.isdigit() else 0
            deaths = int(deaths_raw) if deaths_raw.isdigit() else 0
            if injured != 0 or deaths != 0:
                continue  # that would be "high", not "medium"

            if odino in candidates:
                candidates[odino]["component_raw"].add(field(row, IDX_COMPDESC).strip())
                continue

            candidates[odino] = {
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

print(f"total rows scanned: {total_rows:,}")
print(f"vehicle rows: {vehicle_rows:,}")
print(f"TOTAL POOL of medium-pattern candidates (not already used): {len(candidates):,}")
assert len(candidates) >= n_needed, f"pool too small: need {n_needed}, found {len(candidates)}"

sampled = rng.sample(list(candidates.values()), n_needed)


def to_record(c):
    component_raw_list = sorted(c["component_raw"])
    primary_raw_top_level = component_raw_list[0].split(":")[0].strip()
    joined_raw = " ".join(component_raw_list)
    return {
        "odino": c["odino"],
        "cmplid": c["cmplid"],
        "make": c["make"],
        "model": c["model"],
        "year": c["year"],
        "narrative": c["narrative"],
        "component": bucket_component(primary_raw_top_level),
        "component_raw": ",".join(component_raw_list),
        "defect_type": defect_type(joined_raw, c["narrative"], c["fire"]),
        "safety_risk": safety_risk(c["crash"], c["fire"], c["injured"], c["deaths"]),
        "severity": severity(c["crash"], c["fire"], c["injured"], c["deaths"]),
        "crash": c["crash"],
        "fire": c["fire"],
        "injured": c["injured"],
        "deaths": c["deaths"],
    }


new_records = [to_record(c) for c in sampled]
assert all(r["severity"] == "medium" for r in new_records)

new_train = train + new_records
rng.shuffle(new_train)

assert len(set(r["odino"] for r in new_train)) == len(new_train), "duplicate ODINO after growth"
eval_odinos = {r["odino"] for r in eval_rows}
assert not (set(r["odino"] for r in new_train) & eval_odinos), "leaked eval ODINOs"

with open("data/processed/train.jsonl", "w", encoding="utf-8") as f:
    for r in new_train:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

from collections import Counter
sev_dist = Counter(r["severity"] for r in new_train)
n_yes = sum(1 for r in new_train if r["safety_risk"] == "yes")

print()
print("GROWN train.jsonl:")
print(f"  total: {len(new_train)}")
print(f"  low: {sev_dist['low']}  medium: {sev_dist['medium']}  high: {sev_dist['high']}")
print(f"  safety_risk=yes: {n_yes} ({n_yes/len(new_train):.1%})")
print(f"  added {len(new_records)} new medium examples, nothing removed")
