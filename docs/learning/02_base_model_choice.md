# 2. Base Model Choice: Qwen3-8B vs. Qwen2.5-7B vs. Llama-3.2-3B vs. Mistral-7B vs. Phi-3-mini

## (1) The concept

An **open-weight model** is one whose trained numbers are published and downloadable —
anyone can run it, inspect it, or fine-tune it, unlike closed models (e.g. GPT-4) that
are only reachable through a paid API. **Parameter count** (the "8B" in "Qwen3-8B") is
the number of trainable numbers in the model — roughly, more parameters means more
capacity to learn complex patterns, but also more memory and compute needed to run it.
**Benchmarks** are standardized tests researchers run across many models so capabilities
can be compared on a level playing field. An **Apache 2.0 license** is a permissive
open-source license — free to use, modify, and redistribute, including commercially, no
royalties or copyleft restrictions.

## (2) How this project uses it

`docs/blueprint.md` Section 2, Decision 3 has the full locked comparison table. Summary
of the real reasoning: Qwen3-8B matches or beats the *previous-generation* Qwen2.5-14B
(a model with nearly twice the parameters) on many benchmarks — a genuine generational
jump, not a marginal pick. It's Apache 2.0, fits a free-tier Colab/Kaggle GPU under
QLoRA, and quantizes down small enough to run on the local RTX 3050 for the demo. In
code: `model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit"` in
`notebooks/train_qlora_dora.ipynb` cell 4 (and the same string is the base model
identity checked in every adapter's `adapter_config.json`, per
`docs/training-hyperparameters.md`'s completeness checks).

One project-specific detail not in the original blueprint table: **thinking mode is
disabled**. Qwen3 models can optionally generate a visible "thinking" section before
their real answer (useful for step-by-step reasoning tasks). This task wants a fast,
deterministic JSON object, not reasoning text mixed into the output, so thinking is
turned off via `enable_thinking=False` wherever the chat template is applied — e.g.
`notebooks/train_qlora_dora.ipynb` cell 10's `tokenizer.apply_chat_template(...,
enable_thinking=False)`, and the identical setting in the eval notebook (cell 5) so
inference matches training exactly.

## (3) When the alternative would win

- **Qwen2.5-7B-Instruct** — if Qwen3 weren't available/mature yet, or the project needed
  strict compatibility with an existing Qwen2.5-specific toolchain from prior work.
- **Llama-3.2-3B-Instruct** — if hardware were *even more* constrained than this
  project's (e.g. genuinely needed to fit on a phone or a much smaller card) and could
  tolerate weaker JSON-schema reliability with extra retry/validation logic to catch
  malformed output.
- **Mistral-7B-Instruct** — if continuity with a different existing project's tooling
  mattered more than raw capability, or for architectural diversity in an ensemble.
- **Phi-3-mini (3.8B)** — as a last-resort fallback specifically if even the 8B-in-4-bit
  footprint somehow didn't fit available memory — smaller and safer, but with the same
  JSON-reliability weakness as the Llama-3.2-3B option.
