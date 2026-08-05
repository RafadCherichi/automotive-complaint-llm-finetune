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

FINE-TUNING METHOD: QLoRA + DoRA (4-bit quantized base + LoRA adapters,
with use_dora=True). QLoRA is locked project-wide — not open for
reconsideration mid-build. DoRA is a same-cost refinement on top of it
(decomposes the weight update into magnitude and direction, converges
faster, often matches full fine-tuning quality at the same rank) — enable
it via the training framework's use_dora flag, no extra VRAM cost.

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

### Fine-tuning method: **QLoRA + DoRA** (pre-locked, not reconsidered here)

**Update (verified via research):** QLoRA alone is confirmed correct for this hardware — it's the current standard default specifically for single-GPU fine-tuning when the base model doesn't fit unquantized. On structured-output tasks like this one, QLoRA/LoRA already land within a few percent of full fine-tuning, so the "full FT is secretly better" concern doesn't really apply here. **DoRA is added on top as a same-cost upgrade** — it's a training-config flag (`use_dora=True`), not a different method or extra VRAM cost, and it converges faster / often matches full fine-tuning quality at the same rank. It has effectively no downside for this project, so it's promoted from "footnote" to "locked."

For the learning doc, document these alternatives even though the method itself was fixed by the project's scope:

| Alternative | Why not used |
|---|---|
| Full fine-tuning | Needs to update all model weights in full precision — far beyond 4GB VRAM, not feasible on this hardware even on Colab's free tier for an 8B model. Also, the quality gap vs. QLoRA is smallest on exactly this kind of structured-output task, so the cost wouldn't be well spent even with unlimited hardware. |
| Plain LoRA (fp16/bf16, no quantization) | Base model stays unquantized in memory during training — would blow past available VRAM even training adapters-only on an 8B model. |
| Prompt-tuning / soft prompts | Much lower capacity — unlikely to reliably learn a strict structured-output schema; better suited to simpler style/tone shifts than field-level JSON accuracy. |
| QLoRA without DoRA | Still a valid, widely-used baseline — but DoRA is a free quality upgrade at the same memory cost, so there's no real reason to skip it here. |

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

## Section 5: Label Strategy (how ground-truth labels get created — zero manual labeling)

**Key insight to give Claude Code:** NHTSA complaint records aren't just free text — each complaint already ships with structured metadata columns alongside the narrative: a component description field, and boolean-ish flags for crash/fire/injury/death involvement, plus recall linkage. This means the target schema can largely be **derived from existing columns, not hand-labeled from scratch**:

| Target field | Derivation source |
|---|---|
| `component` | NHTSA's own component field (may need light normalization/bucketing — e.g. collapsing near-duplicate component strings into a clean taxonomy) |
| `defect_type` | Short categorical summary — likely needs a small taxonomy designed from the free-text narrative + component field (this is the one field most likely to need Claude Code's judgment — flag it as an options-first decision in Phase 1, not something to invent silently) |
| `safety_risk` | Derived from crash/fire/injury/death flags — if any are present, `"yes"`, else `"no"` |
| `severity` | Derived tiering from the same flags (e.g. death/injury → high, crash/fire without injury → medium, none → low) — exact thresholds are a Phase 1 decision, not pre-locked |

This keeps the project honest against "existing data only, zero budget": labels are **derived programmatically from real NHTSA metadata**, not invented, not LLM-generated synthetic labels, and not manually annotated by Rafad. Claude Code should confirm this derivation logic explicitly (with worked examples) before generating the training set — this is what the existing Phase 1 instruction to "propose the exact JSON label schema... and show me before writing bulk code" is for.

**Data sizing target:** aim for roughly 500–1,000 labeled training pairs (well within documented QLoRA norms for classification/extraction-style tasks on 7-8B models) plus a separate held-out set of 100–150 hand-spot-checked examples for the before/after eval — spot-checked by Rafad, not the full training set, to keep this from becoming a manual-labeling bottleneck.

---

## Section 6: Phase Breakdown

| Phase | Deliverable |
|---|---|
| **Phase 1** | Local dev env, NHTSA data pull, label derivation logic proposed + confirmed, training set (~500-1,000 pairs) + held-out eval set (~100-150 pairs) built |
| **Phase 2** | Colab/Kaggle QLoRA + DoRA training notebook, base model loaded in 4-bit, LoRA config (rank/alpha — options-first if nontrivial), training run, adapter saved |
| **Phase 3** | Evaluation: baseline (zero-shot Qwen3-8B) vs. fine-tuned, on JSON validity rate + field-level accuracy over the held-out set; error analysis writeup |
| **Phase 4** | Merge/export adapter, quantize to GGUF (Q4), verify it runs locally on the RTX 3050; build the demo (Streamlit, matching Project 2's pattern) |
| **Phase 5** | Docs: learning docs (alternatives-considered tables in full), `pm-perspective.md`, README, final repo cleanup |

---

## Section 7: Repo Structure & Dependencies

```
automotive-complaint-llm-finetune/
├── README.md
├── .gitignore
├── requirements.txt          # local: data prep + eval only, no training libs
├── docs/
│   ├── blueprint.md           (this file)
│   ├── label-strategy.md      (Phase 1 output)
│   ├── model-choice.md        (learning doc: alternatives tables in full)
│   ├── eval-report.md         (Phase 3 output)
│   └── pm-perspective.md      (Phase 5 output)
├── data/                      # gitignored — raw NHTSA pulls + processed sets
├── notebooks/
│   └── train_qlora_dora.ipynb # runs on Colab/Kaggle, not local
├── eval/
│   └── eval_harness.py        # runs locally against the quantized model
└── demo/
    └── streamlit_app.py
```

**Local requirements.txt (data prep + eval only):** `pandas`, `requests`, `jsonschema` (for JSON validity checks), `llama-cpp-python` or `ollama` (for running the quantized GGUF locally in eval/demo).

**Colab/Kaggle notebook dependencies (training only, never installed locally):** `unsloth`, `transformers`, `peft`, `bitsandbytes`, `trl`.

---

Ready to start Phase 1 (NHTSA data pull + label-derivation proposal + training/eval set construction) whenever you are.
