"""Targeted supplemental pull: severity=high complaints specifically where
injured>0 or deaths>0 but crash=False and fire=False -- the under-represented
sub-pattern identified in the Phase 3 eval diagnosis (the fine-tuned model learned
"crash/fire mentioned -> medium" as a shortcut and under-weights injury-only language).

Streams the full flat file once (same lightweight csv approach as build_dataset.py --
this machine has repeatedly run low on RAM with pandas' chunked reader). Excludes any
ODINO already used in the existing train.jsonl or eval.jsonl (eval must stay untouched,
and we don't want accidental duplicates in the rebuilt train set).

Reports the full population size before anything is decided about how many to add --
per the explicit instruction not to assume there are enough.
"""
import csv
import io
import json
import zipfile

from component_taxonomy import bucket_component
from label_rules import safety_risk, severity, defect_type

ZIP_PATH = "data/raw/FLAT_CMPL.zip"
INNER_NAME = "FLAT_CMPL.txt"
MIN_NARRATIVE_LEN = 40

IDX_CMPLID, IDX_ODINO, IDX_MAKE, IDX_MODEL, IDX_YEAR = 0, 1, 3, 4, 5
IDX_CRASH, IDX_FIRE, IDX_INJURED, IDX_DEATHS, IDX_COMPDESC = 6, 8, 9, 10, 11
IDX_CDESCR, IDX_PROD_TYPE = 19, 45


def field(row, i):
    return row[i] if i < len(row) else ""


train = [json.loads(l) for l in open("data/processed/train.jsonl", encoding="utf-8")]
ev = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
used_odinos = {r["odino"] for r in train} | {r["odino"] for r in ev}
print(f"already-used ODINOs (train+eval, excluded from candidates): {len(used_odinos)}")

candidates = {}  # odino -> record, merging multi-component rows like build_dataset.py

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
            if crash or fire:
                continue  # this is exactly the sub-pattern we're excluding

            injured_raw = field(row, IDX_INJURED).strip()
            deaths_raw = field(row, IDX_DEATHS).strip()
            injured = int(injured_raw) if injured_raw.isdigit() else 0
            deaths = int(deaths_raw) if deaths_raw.isdigit() else 0
            if injured == 0 and deaths == 0:
                continue  # not severity=high

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
print(f"TOTAL POOL: severity=high, crash=False, fire=False, not already used: {len(candidates):,}")

with open("data/raw/injury_only_high_candidates.json", "w", encoding="utf-8") as out:
    json.dump(
        [{**v, "component_raw": sorted(v["component_raw"])} for v in candidates.values()],
        out, ensure_ascii=False, indent=2,
    )
print("saved full candidate pool to data/raw/injury_only_high_candidates.json")
