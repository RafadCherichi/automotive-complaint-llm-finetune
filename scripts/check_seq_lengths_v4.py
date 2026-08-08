"""Round 4 pre-check: does the 7-field target still fit MAX_SEQ_LENGTH=768 with the
real Qwen3 tokenizer, now that crash_described/fire_described/injury_described add a
few extra tokens to every training example's output? Same method as the original
scripts/check_seq_lengths.py -- checked against the real tokenizer, not assumed.
"""
import json

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = (
    "You are an automotive safety complaint analyst. Given a raw consumer complaint "
    "about a vehicle, extract a structured JSON object with exactly these fields: "
    'component (string), defect_type (string), safety_risk ("yes" or "no"), '
    'severity ("low", "medium", or "high"), crash_described (true or false -- does '
    'the complaint text itself describe an actual collision/impact, not just '
    'mention a safety feature by name), fire_described (true or false -- does the '
    'text describe an actual fire/smoke/explosion event), injury_described (true '
    'or false -- does the text describe an actual injury to a person, not a '
    'hypothetical or averted one). Respond with only the JSON object.'
)


def format_example(r):
    user = f"Complaint:\n{r['narrative']}"
    output = json.dumps({
        "component": r["component"],
        "defect_type": r["defect_type"],
        "safety_risk": r["safety_risk"],
        "severity": r["severity"],
        "crash_described": r["crash_described"],
        "fire_described": r["fire_described"],
        "injury_described": r["injury_described"],
    })
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


tok = AutoTokenizer.from_pretrained(MODEL_ID)

rows = []
with open("data/processed/train_v4.jsonl", encoding="utf-8") as f:
    rows.extend(json.loads(l) for l in f)
with open("data/processed/eval.jsonl", encoding="utf-8") as f:
    eval_rows = [json.loads(l) for l in f]
# eval.jsonl doesn't have the 3 new fields (it's untouched) -- pad with False for the
# length check only, since the eval-time prompt is user+system only (no target JSON
# appended), so this only matters for train_v4.jsonl rows in practice. Included here
# anyway for a conservative combined check.
for r in eval_rows:
    r.setdefault("crash_described", False)
    r.setdefault("fire_described", False)
    r.setdefault("injury_described", False)
rows.extend(eval_rows)

lengths = []
for r in rows:
    text = format_example(r)
    n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
    lengths.append(n_tokens)

lengths.sort()
n = len(lengths)
print(f"n={n}  max={lengths[-1]}  p99.9={lengths[int(n*0.999)]}  p99={lengths[int(n*0.99)]}  p95={lengths[int(n*0.95)]}")
over_768 = sum(1 for l in lengths if l > 768)
print(f"rows exceeding 768: {over_768} ({100*over_768/n:.3f}%)")
