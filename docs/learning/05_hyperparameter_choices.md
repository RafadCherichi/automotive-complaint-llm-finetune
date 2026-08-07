# 5. Hyperparameter Choices: rank=16, alpha=32, learning rate=2e-4, 3 epochs, max_seq_length=768

## (1) The concept

**Hyperparameters** are settings chosen *before* training starts (as opposed to the
model's internal weights, which get learned *during* training).

- **Rank (r)**, from `01_qlora_dora_vs_alternatives.md`: how much capacity the LoRA/DoRA
  adapter has. Higher rank = more capacity to learn complex patterns, but also more risk
  of **overfitting** — memorizing the specific training examples instead of learning the
  general pattern, like a student who memorizes practice-test answers instead of
  understanding the material, and then fails on the real test with different questions.
- **Alpha**: a scaling factor controlling how strongly the adapter's learned changes
  affect the frozen base model's behavior.
- **Learning rate**: how big a step the model takes when correcting itself after each
  batch of examples. Too high and training becomes unstable (like overcorrecting a car's
  steering and swerving); too low and training barely progresses in the time available.
- **Epoch**: one full pass through the entire training dataset. More epochs generally
  help the model learn better, up to a point — beyond that point it starts overfitting
  instead of generalizing.
- **max_seq_length / tokens**: models process text broken into "tokens" (roughly
  word-pieces, not whole words). This setting caps how many tokens of input+output each
  training example can use. Too short and real content gets cut off mid-complaint
  (truncation); too long wastes memory and compute on padding that isn't needed.

## (2) How this project uses it

All five values, and the reasoning for each, are locked in
`docs/training-hyperparameters.md`'s table — this card adds the "why," grounded in real
evidence gathered *before* locking each number, not textbook defaults assumed blind:

- **r=16, alpha=32** (`notebooks/train_qlora_dora.ipynb` cell 6): the standard
  alpha = 2×rank convention. Rank 16 was chosen specifically *because* the training set
  is small (800-900 examples) — a higher rank (32/64) adds more parameters to fit
  without more data to fit them on, the wrong direction for a dataset this size.
- **learning rate = 2e-4** (cell 12's `SFTConfig`): the standard, well-documented QLoRA
  default for 7-8B models.
- **3 epochs, but checked, not assumed**: cell 13's markdown flags overfitting as a real
  risk on a small dataset; cell 14 explicitly logs eval loss after every epoch to check
  for it rather than trusting epoch count blindly. The result justified the caution —
  **both v1 and v2 showed eval loss best at epoch 2 and worse at epoch 3**
  (`docs/training-hyperparameters.md`'s three-round table: v1 1.0595 vs 1.0683, v2
  1.0600 vs 1.0684 — almost identical regression size both times, not a fluke). v3's
  epoch 3 wasn't separately confirmed, but the same checkpoint-selection logic was
  applied on the strength of the pattern holding twice already. This is a repeatable
  signal that this dataset size sits right at the edge of what 3 full passes can support
  before the model starts memorizing instead of generalizing. Epoch 2 was selected as
  the final checkpoint every round, for this reason.
- **max_seq_length = 768**: not assumed from a common default of 512. `scripts/check_seq_lengths.py`
  measured the *actual* token length of every formatted training example against
  Qwen3's real tokenizer (not estimated from character count) before locking this
  number: p95 = 528 tokens, and 512 would have silently truncated 5.2% of real examples
  (49 of 940). 768 covers 99.9% of examples, with only one exceeding it by about 14
  tokens.

## (3) When the alternative would win

- **Higher rank (32/64)** — with a much larger training set (thousands to tens of
  thousands of examples), more adapter capacity stops being a memorization risk and
  starts being genuinely useful headroom.
- **Lower learning rate** — if training showed instability (loss spiking or diverging)
  rather than the smooth, repeatable epoch-2-best pattern actually observed here.
- **More than 3 epochs** — with a much larger dataset, where the overfitting risk from
  repeated passes is lower relative to how much new information each pass still
  provides.
- **Longer max_seq_length** — if the source text itself were longer. NHTSA caps its
  complaint narrative field at 2048 characters by its own database schema, so 768 tokens
  was empirically enough without needing to go to 1024+ "just in case."
