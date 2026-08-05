"""Pass 1: tally top-level COMPDESC categories across the full flat file.
Streamed in chunks so memory stays bounded regardless of file size (~1.5GB uncompressed).
Output feeds the component taxonomy design -- top N by real frequency + OTHER.
"""
import zipfile
from collections import Counter

import pandas as pd

from nhtsa_schema import READ_CSV_KWARGS

ZIP_PATH = "data/raw/FLAT_CMPL.zip"
INNER_NAME = "FLAT_CMPL.txt"
CHUNKSIZE = 100_000

counts = Counter()
total_rows = 0
vehicle_rows = 0

with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(INNER_NAME) as f:
        reader = pd.read_csv(f, chunksize=CHUNKSIZE, **READ_CSV_KWARGS)
        for chunk in reader:
            total_rows += len(chunk)
            chunk = chunk[chunk["PROD_TYPE"] == "V"]
            vehicle_rows += len(chunk)
            top_level = chunk["COMPDESC"].str.split(":").str[0].str.strip()
            counts.update(top_level[top_level != ""].tolist())

print(f"total rows scanned: {total_rows:,}")
print(f"vehicle rows (PROD_TYPE=V): {vehicle_rows:,}")
print(f"unique top-level component categories: {len(counts)}")
print()
print("top 30 by frequency:")
for name, n in counts.most_common(30):
    print(f"{n:>8,}  {name}")
