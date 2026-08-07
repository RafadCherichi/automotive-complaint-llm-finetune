"""Phase 4 local demo -- paste a raw vehicle complaint, get structured JSON back.

Calls Ollama's REST API (localhost:11434) for inference -- no torch/GPU deps in this
process itself; Ollama runs the quantized GGUF model out-of-process. Run with:
    venv/Scripts/streamlit.exe run demo/streamlit_app.py
(after `ollama create qwen3-8b-automotive-complaint -f models/Modelfile`).
"""
import json
import re

import requests
import streamlit as st

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3-8b-automotive-complaint"

SYSTEM_PROMPT = (
    "You are an automotive safety complaint analyst. Given a raw consumer complaint "
    "about a vehicle, extract a structured JSON object with exactly these fields: "
    'component (string), defect_type (string), safety_risk ("yes" or "no"), '
    'severity ("low", "medium", or "high"). Respond with only the JSON object.'
)

_JSON_OBJ_PATTERN = re.compile(r"\{.*?\}", re.DOTALL)

EXAMPLE_COMPLAINT = (
    "While driving on the highway, I heard a loud pop and then the brake pedal went "
    "straight to the floor with no resistance. I had to use the emergency brake to "
    "stop the vehicle. No one was hurt but it was very close to a collision."
)


def parse_json_output(raw_text):
    match = _JSON_OBJ_PATTERN.search(raw_text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def call_model(narrative):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Complaint:\n{narrative}"},
        ],
        "stream": False,
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


st.set_page_config(page_title="Automotive Complaint Triage", page_icon="🚗")
st.title("Automotive Complaint Safety Triage")
st.caption(
    "QLoRA + DoRA fine-tuned Qwen3-8B (quantized to GGUF, Q4_K_M) — extracts "
    "component, defect type, safety risk, and severity from a raw complaint. "
    "See docs/eval-report.md for the full accuracy evidence and known limitations."
)

narrative = st.text_area(
    "Paste a vehicle complaint:",
    value=EXAMPLE_COMPLAINT,
    height=150,
)

if st.button("Analyze", type="primary"):
    try:
        with st.spinner("Running local model (partial GPU offload — may take a few seconds)..."):
            raw_output = call_model(narrative)
    except requests.exceptions.ConnectionError:
        st.error(
            "Couldn't reach Ollama at localhost:11434. Is Ollama running? "
            "(Check the system tray, or run `ollama serve`.)"
        )
        st.stop()
    except requests.exceptions.HTTPError as e:
        st.error(
            f"Ollama returned an error: {e}. Has the model been created? "
            f"Run: ollama create {MODEL_NAME} -f models/Modelfile"
        )
        st.stop()

    parsed = parse_json_output(raw_output)

    if parsed is None:
        st.warning("Model output did not parse as valid JSON — showing raw output:")
        st.code(raw_output)
    else:
        risk = str(parsed.get("safety_risk", "")).strip().lower()
        severity = str(parsed.get("severity", "")).strip().lower()

        cols = st.columns(4)
        cols[0].metric("Component", parsed.get("component", "?"))
        cols[1].metric("Defect Type", parsed.get("defect_type", "?"))
        cols[2].metric("Safety Risk", risk.upper() or "?")
        cols[3].metric("Severity", severity.upper() or "?")

        if risk == "yes":
            if severity == "high":
                st.error("Flagged: safety risk, HIGH severity — priority review recommended.")
            elif severity == "medium":
                st.warning(
                    "Flagged: safety risk, MEDIUM severity. Per docs/eval-report.md, "
                    "the medium/high severity boundary is where this model is weakest — "
                    "route to human review rather than trusting the tier automatically."
                )
            else:
                st.info("Flagged as a safety risk (severity: low).")
        else:
            st.success("Not flagged as a safety risk.")

        with st.expander("Raw JSON"):
            st.json(parsed)

st.divider()
st.caption(
    "Local-only demo. Model: shipped adapter (Phase 3 v2) merged + quantized to GGUF, "
    "served by Ollama on this machine. Not connected to any external service."
)
