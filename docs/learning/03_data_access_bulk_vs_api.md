# 3. Data Access: NHTSA Full Flat-File Bulk Download vs. Per-Make/Model API Pulls

## (1) The concept

**Selection bias** is what happens when *how* you choose your sample of data isn't
representative of the full population, so conclusions drawn from the sample end up
systematically skewed. Classic example: surveying only smartphone owners about internet
habits will overstate smartphone usage in the general population, because the sampling
method itself excluded non-owners.

An **API (Application Programming Interface)** lets you request a specific slice of
data on demand — like ordering à la carte at a restaurant. A **bulk/flat-file download**
gives you the entire dataset in one file — like buying the whole cookbook instead of
one recipe at a time. **Streaming / chunked processing** means reading a large file
piece by piece instead of loading the whole thing into memory at once — necessary when
the file is bigger than the computer's available RAM.

## (2) How this project uses it

NHTSA offers both: an API that returns complaints for a specific make/model/year you
name, and a single bulk file (`FLAT_CMPL.zip`) containing every complaint since 1995.
This project downloads the **whole bulk file** (`scripts/build_dataset.py`) and streams
through all 2,231,883 rows using Python's built-in `csv` module, one row at a time —
not pandas, because this machine's 8GB RAM (often under 1GB free) couldn't safely hold
pandas' more memory-hungry chunked reader (`docs/label-strategy.md` documents the actual
`MemoryError` hit and the fix).

Choosing the whole file over API-per-vehicle calls means the reservoir-sampling scripts
(`scripts/build_dataset.py`, `scripts/find_injury_only_high.py`,
`scripts/grow_medium_tier.py`) draw from the *entire* real population of NHTSA
complaints, across every make, model, and year — nobody manually chose "let's use Honda
and Toyota complaints." If specific vehicles had been hand-picked to query via the API
instead, the training data would reflect whatever quirks those particular
makes/models/years happen to have (different typical defect patterns, different
customer writing styles), not the general population of automotive safety complaints.

## (3) When the alternative would win

The API approach would be the right call if the goal were to **monitor a small, fixed
set of known vehicles on an ongoing basis** — e.g. a manufacturer's internal tool
watching new complaints about their own specific models as they arrive, where querying
"give me anything new about model X" is exactly the right shape of question, and the
full historical bulk file would be unnecessary overhead. It's also the better choice
when the full dataset is too large to download and process even in chunks — not the
case here, since 1.5GB uncompressed was manageable with careful streaming.
