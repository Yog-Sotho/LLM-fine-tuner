# 05 — Alignment (RLHF) Pipeline

Alignment training teaches a model which of its possible answers are *better* — the technique behind helpful chat assistants. The **🤖 RLHF Pipeline** tab offers four methods, all built on Hugging Face TRL.

> **This is an advanced feature.** If you're new to fine-tuning, complete standard SFT training first ([04 — Training](04_training.md)).

---

## The Big Picture

Standard fine-tuning (SFT) teaches a model to imitate your examples. Alignment goes further: it teaches the model which outputs are *good*.

```
Reward-based:     SFT  →  A. Reward model  →  B. GRPO
Verifiable tasks: SFT  →  B. GRPO with a `reference` column (no reward model needed)
One-step:         SFT  →  C. ORPO   (ranked pairs)
                  SFT  →  D. KTO    (thumbs-up / thumbs-down feedback)
```

---

## Checking Dependencies

At the top of the tab you'll see a status line:

```
Reward model ✅ | GRPO ✅ | ORPO ✅ | KTO ✅
```

If you see ❌, install TRL:

```bash
pip install "trl>=0.29.1,<2"
```

---

## Tab A — Reward Model

### What is a Reward Model?

A reward model is a second model that scores a response **for a given prompt**: high score = good response. It is a sequence classifier with a single score output, trained so that the `chosen` answer scores higher than the `rejected` one.

### Data Format

Columns `prompt`, `chosen` (a good response) and `rejected` (a worse response to the same prompt):

```csv
prompt,chosen,rejected
"How long is a year?","The Earth orbits the Sun once every 365.25 days.","The Earth doesn't move."
"How do I cook pasta?","Boil salted water, add the pasta and cook for the time on the package.","Just put pasta in water."
```

### Settings

| Setting | Default | Notes |
|---|---|---|
| Base Model ID | Auto | A model of the same family as your policy works best |
| Output Directory | `./reward_model` | Where the reward model is saved |
| Epochs | 3 | 1–3 is usually enough |
| Learning Rate | 1×10⁻⁴ | Trained with LoRA, so higher than full fine-tuning |
| Batch Size | 4 | Reduce if you run out of memory |
| Eval Steps | 100 | How often to evaluate on the 10% holdout |
| Max Length | 1024 | Maximum tokens for prompt + response |

Training uses LoRA; when it finishes the adapter is **merged** and a complete model is saved, so its folder can be used directly as the GRPO reward in Tab B.

---

## Tab B — GRPO

### What is GRPO?

GRPO (Group Relative Policy Optimization) is the reinforcement-learning method behind recent reasoning models. For every prompt the model writes several answers, each answer is scored, and the model is pushed towards the answers that beat their group's average. Unlike PPO it needs no separate value model, which makes it simpler and lighter.

### Where the reward comes from

You can use either or both:

1. **A reward model** from Tab A — enter its folder in *Reward Model Path*.
2. **Reference answers** — add a `reference` column. An answer earns reward 1 when it contains the reference (case-insensitive). Ideal for maths, extraction and other tasks with a checkable answer.

```csv
prompt,reference
"What is 12 × 12?","144"
"What is the capital of France?","Paris"
```

A prompts-only file (`prompt` column) works when you give a reward model path.

### Settings

| Setting | Default | Notes |
|---|---|---|
| Policy Model ID | Auto | Your SFT model or a Hub model ID |
| Reward Model Path | — | Folder from Tab A (optional if you have `reference`) |
| Output Directory | `./grpo_model` | The trained LoRA adapter is saved here |
| Learning Rate | 1×10⁻⁵ | Keep low — RL is sensitive |
| Epochs | 1 | Passes over the prompts |
| KL Beta | 0 | Penalty for drifting from the original model (0 = off, the current recommended default) |
| Completions per Prompt | 4 | More = better baseline, slower |
| Prompts per Step | 1 | Batch size is prompts × completions |
| Max Completion Tokens | 128 | Length of each generated answer |

GRPO generates text during training, so it is slower per step than SFT.

---

## Tab C — ORPO

### What is ORPO?

ORPO (Odds Ratio Preference Optimization) aligns the model in a single step by comparing good and bad responses — no reward model and no reference model.

### Data Format

Same as DPO: `prompt`, `chosen` and `rejected` columns.

```csv
prompt,chosen,rejected
"How do I apologise to a friend?","Be sincere, acknowledge what you did wrong, and listen to their response.","Just say sorry and move on."
"What is a healthy snack?","Fruit, nuts, yogurt, or vegetable sticks with hummus are all great choices.","Chips and soda."
```

### Settings

| Setting | Default | Notes |
|---|---|---|
| Base Model ID | Auto | Starting model |
| Output Directory | `./orpo_model` | Where to save |
| Learning Rate | 1×10⁻⁴ | |
| Beta | 0.1 | ORPO loss weight. Higher = stronger preference signal. |
| Alpha | 0.1 | Balances SFT and preference loss components (older TRL only). |
| Epochs | 3 | |
| Batch Size | 2 | |

---

## Tab D — KTO

### What is KTO?

KTO (Kahneman-Tversky Optimization) learns from **individual** responses labelled good or bad — the kind of data you get from thumbs-up / thumbs-down buttons. You don't need two ranked answers per prompt.

### Data Format

Either unpaired feedback:

```csv
prompt,completion,label
"Summarise this email","Meeting moved to Friday at 3pm.",true
"Summarise this email","lol idk",false
```

`label` accepts `true/false`, `1/0`, `yes/no`, `good/bad`.

Or ranked pairs (`prompt`, `chosen`, `rejected`) — each pair becomes one good and one bad example. You need at least one good and one bad example.

### Settings

| Setting | Default | Notes |
|---|---|---|
| Base Model ID | Auto | Starting model |
| Output Directory | `./kto_model` | The trained LoRA adapter is saved here |
| Learning Rate | 5×10⁻⁵ | |
| Beta | 0.1 | How far the model may move from the original |
| Epochs | 1 | |
| Batch Size | 4 | At least 2 (KTO estimates a baseline per batch) |
| Max Length | 512 | Maximum tokens for prompt + response |

---

## Which Should I Use?

| Scenario | Recommendation |
|---|---|
| I have ranked pairs (good vs bad answer) | **ORPO**, or **DPO** in the Training tab |
| I only have thumbs-up / thumbs-down feedback | **KTO** |
| My task has a checkable answer (maths, extraction) | **GRPO** with a `reference` column |
| I want a reusable judge to optimise against | **Reward model → GRPO** |
| I'm just starting out | **SFT only** is fine for most use cases |

---

## Next Step

→ [06 — Inference](06_inference.md): Test and serve your model.
