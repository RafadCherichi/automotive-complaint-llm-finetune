"""Phase 2 pre-check: token length distribution using Qwen3-8B-Instruct's real
tokenizer, over the actual formatted string (instruction + narrative + JSON output)
that gets fed to the model during training -- not raw narrative length. Tokenizer-only
load (no torch, no model weights) -- CPU text processing, not a training step.
"""
import json

from transformers import AutoTokenizer

MODEL_ID = "Qwen/Qwen3-8B"  # base instruct model's tokenizer; thinking mode irrelevant to tokenization

SYSTEM_PROMPT = (
    "You are an automotive safety complaint analyst. Given a raw consumer complaint "
    "about a vehicle, extract a structured JSON object with exactly these fields: "
    'component (string), defect_type (string), safety_risk ("yes" or "no"), '
    'severity ("low", "medium", or "high"). Respond with only the JSON object.'
)


def format_example(r):
    user = f"Complaint:\n{r['narrative']}"
    output = json.dumps({
        "component": r["component"],
        "defect_type": r["defect_type"],
        "safety_risk": r["safety_risk"],
        "severity": r["severity"],
    })
    # Matches Qwen's chat template structure closely enough for a length estimate;
    # the exact template gets applied for real in the Phase 2 training notebook.
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


tok = AutoTokenizer.from_pretrained(MODEL_ID)

rows = []
for fn in ["data/processed/train.jsonl", "data/processed/eval.jsonl"]:
    with open(fn, encoding="utf-8") as f:
        rows.extend(json.loads(l) for l in f)

lengths = []
for r in rows:
    text = format_example(r)
    n_tokens = len(tok(text, add_special_tokens=False)["input_ids"])
    lengths.append(n_tokens)

lengths.sort()
n = len(lengths)


def pct(p):
    return lengths[min(n - 1, int(n * p))]


print(f"examples: {n}")
print(f"min: {lengths[0]}  max: {lengths[-1]}")
print(f"p50: {pct(0.50)}  p90: {pct(0.90)}  p95: {pct(0.95)}  p99: {pct(0.99)}")
print()
for cap in (256, 384, 512, 768, 1024):
    over = sum(1 for x in lengths if x > cap)
    print(f"examples exceeding {cap} tokens: {over} ({over/n:.1%})")
