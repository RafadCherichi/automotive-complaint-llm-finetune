"""Phase 4 step 3: smoke-test the locally-loaded GGUF model via Ollama's REST API.

Not a full re-grading (that's what eval_baseline_vs_finetuned.ipynb already did on
Kaggle) -- just confirms the quantized model actually runs locally and produces
sensible, parseable JSON on a few real complaints, per the blueprint's local-inference
role. Run this after `ollama create qwen3-8b-automotive-complaint -f models/Modelfile`.

Usage: venv/Scripts/python.exe scripts/verify_local_gguf.py
"""
import json
import random
import re
import time

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3-8b-automotive-complaint"

SYSTEM_PROMPT = (
    "You are an automotive safety complaint analyst. Given a raw consumer complaint "
    "about a vehicle, extract a structured JSON object with exactly these fields: "
    'component (string), defect_type (string), safety_risk ("yes" or "no"), '
    'severity ("low", "medium", or "high"). Respond with only the JSON object.'
)

_JSON_OBJ_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)


def parse_json_output(raw_text):
    match = _JSON_OBJ_PATTERN.search(raw_text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def ask(narrative):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Complaint:\n{narrative}"},
        ],
        "stream": False,
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    elapsed = time.time() - t0
    raw = resp.json()["message"]["content"]
    return raw, elapsed


def main():
    print(f"checking Ollama has model '{MODEL_NAME}' loaded...")
    tags = requests.get("http://localhost:11434/api/tags").json()
    names = [m["name"] for m in tags.get("models", [])]
    if not any(MODEL_NAME in n for n in names):
        raise SystemExit(
            f"'{MODEL_NAME}' not found in Ollama (models present: {names}). "
            f"Run: ollama create {MODEL_NAME} -f models/Modelfile"
        )
    print("found. running a few real complaints through it...\n")

    eval_rows = [json.loads(l) for l in open("data/processed/eval.jsonl", encoding="utf-8")]
    rng = random.Random(7)
    # one of each severity tier, for a quick spread-check, not a full re-eval
    by_tier = {"low": [], "medium": [], "high": []}
    for r in eval_rows:
        by_tier[r["severity"]].append(r)
    sample = [rng.choice(by_tier[t]) for t in ("low", "medium", "high")]

    for row in sample:
        raw, elapsed = ask(row["narrative"])
        parsed = parse_json_output(raw)
        print(f"--- odino={row['odino']} (actual severity={row['severity']}) ---")
        print(f"narrative: {row['narrative'][:150]}")
        print(f"raw model output: {raw[:300]}")
        if "<think>" in raw.lower():
            print("FLAG: response contains a <think> tag -- thinking mode may not be fully suppressed.")
        if parsed is None:
            print("FLAG: output did not parse as JSON.")
        else:
            print(f"parsed: {parsed}")
            print(f"actual: component={row['component']} defect_type={row['defect_type']} "
                  f"safety_risk={row['safety_risk']} severity={row['severity']}")
        print(f"latency: {elapsed:.1f}s")
        print()


if __name__ == "__main__":
    main()
