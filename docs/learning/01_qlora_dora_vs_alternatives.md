# 1. QLoRA + DoRA vs. Full Fine-Tuning vs. Plain LoRA vs. Prompt-Tuning

Every real technical decision in this project gets its own concept card like this one,
explained from scratch — no assumed background in linear algebra, probability, or ML,
every term defined inline the first time it's used.

Each card has 3 parts: **(1) the concept in plain language**, **(2) exactly how this
project uses it, with a real file/function to look at**, **(3) when the alternative
would have actually been the better choice** — not just "why we picked this," but
"here's when we'd have picked something else."

## (1) The concept

**Fine-tuning** means taking a model that's already been trained on a huge amount of
general text (it already "knows" grammar, facts, reasoning patterns) and training it a
little more on a smaller, specific dataset so it gets good at *your* task. Think of it
like hiring an experienced generalist and giving them a few weeks of on-the-job training
for your specific role, instead of training someone from scratch.

**Full fine-tuning** updates *every single number* in the model. Qwen3-8B has 8 billion
of these numbers ("parameters"). To train them, you need to store: the numbers
themselves, how much each one should change ("gradients"), and a running memory of past
changes to make training stable ("optimizer state"). That's roughly 3-4x the model's own
size in memory, on top of the model itself — for an 8B model at normal precision (16-bit
numbers), full fine-tuning needs 100GB+ of GPU memory. A consumer GPU or a free
Colab/Kaggle GPU doesn't have that.

**Quantization** shrinks each number from a high-precision format (like 16-bit) down to
a lower-precision one (like 4-bit) — the same idea as compressing a photo: you lose a
little detail, but the file gets much smaller. A 4-bit version of an 8B model needs
roughly 4-5GB instead of 16GB+.

**LoRA (Low-Rank Adaptation)**: instead of touching the model's original numbers at all,
LoRA freezes them completely and adds small extra "adapter" matrices next to specific
layers. Only the adapters get trained. Why does this save memory? A weight update matrix
that would normally be huge (say, 4096×4096 numbers) gets approximated by multiplying
two much smaller matrices together (like 4096×16 and 16×4096) — the "16" here is called
the **rank**. Lower rank = fewer numbers to train = less memory, at some cost to how
much the adapter can learn. This is "low-rank" because it deliberately uses a small
rank instead of the full size.

**QLoRA** = quantization + LoRA together: load the frozen base model in 4-bit (small),
and only train the small LoRA adapters (kept at higher precision for training
stability). This is what makes fine-tuning an 8B model possible on a free-tier GPU at
all.

**DoRA (Weight-Decomposed LoRA)**: a refinement of LoRA. Any weight update can be
thought of as having a **magnitude** (how big the change is) and a **direction** (which
way it points) — like a vector in physics. Plain LoRA learns both mixed together. DoRA
learns them separately, which empirically converges faster and gets closer to
full-fine-tuning quality at the same rank — for the same memory cost as plain LoRA, it's
a training-config flag, not a different pipeline.

**Prompt-tuning / soft prompts**: don't touch the model's weights at all. Instead, learn
a small set of extra "virtual words" that get silently added to the front of every
input, nudging the model's behavior. Much cheaper than any of the above, but much lower
capacity — good for shifting tone/style, not for teaching a model a whole new structured
output format.

## (2) How this project uses it

`notebooks/train_qlora_dora.ipynb`:
- **The "Q"**: cell 4 loads the base model with `load_in_4bit=True` via
  `FastLanguageModel.from_pretrained(model_name="unsloth/Qwen3-8B-unsloth-bnb-4bit", ...)`
  — the base model never exists at full precision on the training GPU.
- **The "LoRA" + "DoRA"**: cell 6, `FastLanguageModel.get_peft_model(model, r=16,
  lora_alpha=32, lora_dropout=0, bias="none", target_modules=["q_proj", "k_proj",
  "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], use_dora=True, ...)`.
  `r=16` is the rank described above. `use_dora=True` is the entire difference between
  plain LoRA and DoRA — one flag, verified against Unsloth's own documentation before
  use (see `docs/training-hyperparameters.md`).
- **Proof DoRA actually trained, not just got configured**: `docs/training-hyperparameters.md`
  documents checking the saved adapter file for a `lora_magnitude_vector` tensor — that
  tensor only exists if DoRA (the magnitude/direction split) genuinely ran.

## (3) When the alternative would win

- **Full fine-tuning** — if there were access to a data-center GPU (e.g. an 80GB A100)
  with no memory constraint, and the task needed the absolute maximum achievable
  quality. The quality gap between QLoRA and full fine-tuning is smallest on exactly
  this kind of structured-extraction task, though, so even with unlimited hardware the
  extra cost wouldn't buy much here.
- **Plain LoRA (no quantization)** — if the GPU had enough memory to hold the base model
  at full precision anyway (e.g. a 24GB+ card), plain LoRA is slightly simpler (no
  quantization step) with a marginal speed edge, at the cost of needing far more memory
  than QLoRA to get there in the first place.
- **Prompt-tuning** — if the task were something like "always answer a bit more
  formally" rather than "output a strict 4-field JSON schema with a specific label
  taxonomy." Prompt-tuning doesn't have the capacity to reliably learn a structured
  output format like this one.
