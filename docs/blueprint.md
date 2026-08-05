# Master Execution Blueprint — Project 3: LLM Fine-Tuning (QLoRA) [LOCKED]

**Portfolio:** PM+ML, 5-project ranked roadmap
**Rank:** 3 of 5 — completes the "LLM competency triangle" (RAG → Agents → Fine-tuning)
**Predecessors:** Project 1 (RAG Feedback Analyzer, done) · Project 2 (Agentic Text-to-SQL, done)
**Worker:** Claude Code
**Status: all open decisions locked. Ready to start.**

---

## Section 1: Context & Strict Constraints (paste into Claude Code)

```
PROJECT: LLM Fine-Tuning via QLoRA — Automotive Safety-Complaint Structured
Extraction
PORTFOLIO: Project 3 of 5 in a PM+ML applied AI portfolio (Projects 1 and 2
are complete — same documentation rigor and hardware discipline applies).

TASK: Fine-tune an instruction-tuned open-weight LLM so that, given a raw
free-text vehicle complaint, it reliably outputs a structured JSON object:
  { "component": string, "defect_type": string,
    "safety_risk": "yes" | "no", "severity": "low" | "medium" | "high" }
Baseline (zero-shot prompted, unmodified base model) vs. fine-tuned model
must be compared on JSON validity rate and field-level accuracy — this
before/after evidence is the deliverable, not just "a fine-tuned model
exists."

DATASET: NHTSA (National Highway Traffic Safety Administration) public
Vehicle Complaints database — real consumer complaint text, existing and
free, no synthetic/custom data. Subsample a clean few-thousand-row slice
for training; do not attempt the full multi-million-row corpus.

FINE-TUNING METHOD: QLoRA (4-bit quantized base + LoRA adapters). This is
locked project-wide — not open for reconsideration mid-build.

BASE MODEL: Qwen3-8B-Instruct (Apache 2.0, open weights). Disable "thinking
mode" — this task wants fast, deterministic extraction, not exploratory
chain-of-thought reasoning bleeding into the JSON output.

HARDWARE PROFILE (STRICT):
- Local machine: 8GB total system RAM (often as little as ~2GB free),
  RTX 3050 with 4GB VRAM.
- Local machine NEVER trains the model. Its job: data cleaning/prep,
  writing the eval harness, and running the FINAL QUANTIZED model
  (GGUF, Q4) for the demo — same division of labor as Project 2.
- ALL TRAINING happens on Colab or Kaggle's free-tier GPU (T4/P100,
  16GB VRAM). If a step in Claude Code's plan implies loading the
  unquantized 8B model or running a training loop locally, STOP and
  flag it — that violates the hardware contract.

STANDING RULES (apply across the whole portfolio, not just this project):
1. Options-first: any nontrivial choice gets presented with real
   alternatives, pros/cons, and an explicit recommendation — not a
   single assumed path.
2. Learning docs must cover: alternatives considered, reasoning for the
   final pick, and a note on when the alternative would've been the
   better choice — exhaustive learning, not just documenting the pick.
3. Every project needs a deliberate PM-relevant angle, framed in
   product/decision-making terms, built in from the start.
4. Proactively check for free/trial external resources when relevant
   (already done for this project: confirmed Colab/Kaggle free GPU
   tiers are sufficient for an 8B QLoRA run).
5. Git commits: no Co-Authored-By Claude attribution.
6. Communication style while building: short bullet points, plain
   language, inline definitions for any ML jargon, real-world analogies
   where useful — Rafad is a beginner in linear algebra/probability/ML
   and wants concepts taught from scratch, not assumed.
```

---

## Section 2: Open Decisions (Locked)

### Decision 1 — Fine-tuning target task: **Automotive safety-complaint structured extraction**

| Option | Verdict |
|---|---|
| **A. Automotive complaint → structured JSON extraction** | **LOCKED.** On-brand for the Automotive/EV portfolio focus, well-precedented QLoRA use case, strong PM narrative (safety triage). |
| B. Dialogue summarization (SAMSum) | Rejected — safest tutorial coverage, but generic, no domain differentiation. |
| C. EV/car review aspect-sentiment tagging | Rejected — good EV tie-in, but weaker urgency/PM story than a safety-triage tool. |
| D. Text-to-SQL fine-tune | Rejected — too much overlap with Project 2's agent; weak differentiation. |

### Decision 2 — Dataset: **NHTSA Vehicle Complaints Database**

| Option | Verdict |
|---|---|
| **A. NHTSA's own public complaints database** | **LOCKED.** Authoritative source, free, real-world messy text, large enough to subsample cleanly. |
| B. Kaggle mirror of the same data | Rejected as primary — risk of staleness vs. the real source; noted as a fallback if the NHTSA API/download is ever unreachable. |

### Decision 3 — Base model: **Qwen3-8B-Instruct**

| Option | Verdict |
|---|---|
| **A. Qwen3-8B-Instruct** | **LOCKED.** Matches/beats Qwen2.5-14B on many benchmarks at 8B size — a real generational jump, not a marginal pick. Apache 2.0. Fits Colab/Kaggle free-tier QLoRA training and quantizes to GGUF for the RTX 3050 demo. Thinking mode disabled for this task. |
| B. Qwen2.5-7B-Instruct | Superseded — same family, but Qwen3-8B gives materially more capability at a comparable footprint. Kept as documented "previous generation" comparison point in the learning doc. |
| C. Llama-3.2-3B-Instruct | Rejected — weaker strict-JSON-schema reliability at this size; more retry/post-processing logic needed. |
| D. Mistral-7B-Instruct-v0.3 | Rejected — also well-precedented, but no continuity with Project 2's tooling/family. |
| E. Phi-3-mini (3.8B) | Rejected — was Project 1's OOM fallback model specifically, not a primary pick; same JSON-reliability weakness as Llama-3.2-3B. |

### Fine-tuning method: **QLoRA** (pre-locked, not reconsidered here)

For the learning doc, document these alternatives even though the method itself was fixed by the project's scope:

| Alternative | Why not used |
|---|---|
| Full fine-tuning | Needs to update all model weights in full precision — far beyond 4GB VRAM, not feasible on this hardware even on Colab's free tier for an 8B model. |
| Plain LoRA (fp16/bf16, no quantization) | Base model stays unquantized in memory during training — would blow past available VRAM even training adapters-only on an 8B model. |
| Prompt-tuning / soft prompts | Much lower capacity — unlikely to reliably learn a strict structured-output schema; better suited to simpler style/tone shifts than field-level JSON accuracy. |
| DoRA (QLoRA + DoRA combo) | A 2026-current refinement that often matches full fine-tune quality at the same rank — worth a footnote as a "next iteration" upgrade path, not required for this pass. |

---

## Section 3: PM Angle (built in, not bolted on)

**Framing:** *An OEM quality/safety team receives thousands of free-text customer complaints. Manually reading and tagging each one for component, defect type, and safety severity is slow and inconsistent across reviewers. This fine-tuned model auto-structures every incoming complaint — flagging safety-critical ones for priority review — cutting manual triage time and surfacing safety signals faster than a human-only queue.*

Deliverables that speak to this directly:
- Before/after accuracy table (zero-shot base model vs. fine-tuned) framed as "time saved / signal caught" — not just an ML metric table.
- A short "if this were a real product" section: what a false-negative on `safety_risk` costs (a missed safety issue) vs. a false-positive (extra analyst review time) — a precision/recall tradeoff framed as a business decision, not just a confusion matrix.

---

## Section 4: What "done" looks like

- A quantified before/after comparison: base Qwen3-8B (zero-shot, prompted) vs. QLoRA-fine-tuned Qwen3-8B, on JSON validity rate and field-level accuracy, over a hand-verified held-out set.
- A written error analysis: what kinds of complaints the fine-tuned model still gets wrong, and why (ambiguous component mentions, multi-defect complaints, etc.) — honest limitations as a portfolio strength, same standard as Projects 1 and 2.
- The trained LoRA adapter, quantized (GGUF, Q4) and running locally on the RTX 3050 for a live demo — paste in a raw complaint, get structured JSON back.
- Learning docs covering every alternative-considered table above, in full (not just this summary version).
- A PM-perspective doc (matching Project 2's `pm-perspective.md` pattern) laying out the safety-triage business case, the precision/recall tradeoff, and what a v2 (e.g. multi-label defects, severity calibrated against real recall data) would look like.

---

Ready to start Phase 1 (NHTSA data pull + subsample + label-schema design) whenever you are.
