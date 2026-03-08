# 05 — RLHF Pipeline

RLHF stands for Reinforcement Learning from Human Feedback. It's the technique used to make AI models like ChatGPT helpful, harmless, and honest — and now you can use the same approach on your own models.

> **This is an advanced feature.** If you're new to fine-tuning, complete the standard SFT training first ([04 — Training](04_training.md)) before coming here.

---

## The Big Picture

Standard fine-tuning (SFT) teaches a model to imitate your examples. RLHF goes further: it teaches the model which of its possible outputs are *good* and uses that feedback to improve.

A full RLHF pipeline has three steps:

```
Step 1: SFT       — teach the model the basic task
Step 2: Reward    — train a "judge" that scores responses
Step 3: PPO       — use the judge to reinforce good behaviour
```

As an alternative to the full 3-step pipeline, **ORPO** combines alignment into a single training step.

---

## Checking Dependencies

At the top of the **🤖 RLHF Pipeline** tab you'll see a status line:

```
Dependencies: RewardTrainer ✅ | PPO ✅ (ValueHead Rewards Fixed) | ORPO ✅
```

If you see ❌, install the missing package:

```bash
pip install trl>=0.8.0
```

---

## Tab A — Reward Model Training

### What is a Reward Model?

A reward model is a second AI that you train to judge the quality of your main model's outputs. It takes a response and outputs a score: high score = good response, low score = bad response.

### Data Format

Your dataset needs two columns: `chosen` (a good response) and `rejected` (a bad response) for the same context.

```csv
chosen,rejected
"The Earth orbits the Sun once every 365.25 days.","The Earth doesn't move."
"To make pasta, boil salted water first, then add pasta and cook for the time on the package.","Just put pasta in water."
```

### Settings

| Setting | Default | Notes |
|---|---|---|
| Base Model ID | Auto | The same model you SFT-trained, or a compatible base |
| Output Directory | `./reward_model` | Where the reward model is saved |
| Epochs | 3 | Usually 1–5 epochs is sufficient |
| Learning Rate | 1.4×10⁻⁵ | Lower than SFT — reward models are more sensitive |
| Batch Size | 4 | Keep at 4 unless you have VRAM issues |
| Eval Steps | 100 | How often to evaluate on the holdout set |
| Max Length | 1024 | Maximum token length for each example |

### Starting Training

1. Fill in **Base Model ID** (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
2. Upload your preference dataset
3. Set an output directory
4. Click **🎖️ Train Reward Model**

The status box shows training progress. When done, note the output path — you'll need it for PPO.

---

## Tab B — PPO Fine-Tuning

### What is PPO?

PPO (Proximal Policy Optimization) is a reinforcement learning algorithm. After reward model training, PPO uses the reward model as a "critic" to further improve the main model's responses.

In plain terms: the model generates responses, the reward model scores them, and PPO nudges the model to generate higher-scoring responses more often.

### Steps

You need:
1. A trained **policy model** (your SFT-trained model)
2. A trained **reward model** (from Tab A)
3. A dataset of **prompts** (just questions, no answers)

```csv
prompt
"What is the best way to learn a new language?"
"Explain how vaccines work."
"How do I write a professional email?"
```

### Settings

| Setting | Default | Notes |
|---|---|---|
| Policy Model ID | Auto | Your SFT model path or HF model ID |
| Reward Model Path | — | Path from Tab A (e.g. `./reward_model`) |
| Output Directory | `./ppo_model` | Where the PPO-trained model is saved |
| Learning Rate | 1.4×10⁻⁵ | Keep low |
| Batch Size | 1 | PPO is memory-intensive; keep at 1–2 |
| Mini Batch Size | 1 | Must be ≤ Batch Size |
| PPO Epochs | 1 | 1–3 is typical |
| Max New Tokens | 128 | How long each generated response can be |

### Starting PPO

1. Enter your policy model ID or path
2. Enter the reward model path (from Tab A)
3. Upload your prompts dataset
4. Click **🔁 Run PPO Fine-Tuning**

PPO training is slower than SFT because the model generates responses and scores them for each batch. Be patient.

---

## Tab C — ORPO Training

### What is ORPO?

ORPO (Odds Ratio Preference Optimization) is a modern alternative to the full SFT → Reward → PPO pipeline. It trains alignment directly in a single step by comparing good and bad responses using a mathematical "odds ratio".

**Why use ORPO instead of PPO?**
- Simpler: only one training step, no reward model needed
- Faster: no generation loop
- Competitive quality to full RLHF for most use cases

### Data Format

Same as DPO: you need `prompt`, `chosen`, and `rejected` columns.

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
| Alpha | 0.1 | Balances SFT and preference loss components. |
| Epochs | 3 | |
| Batch Size | 2 | |

### Starting ORPO

1. Enter your base model ID
2. Upload your preference dataset
3. Adjust settings if needed
4. Click **🌀 Run ORPO Training**

---

## Which Should I Use?

| Scenario | Recommendation |
|---|---|
| I just want better, more helpful outputs | **ORPO** — one step, great results |
| I want full control and the best possible alignment | **Full RLHF** (SFT → Reward → PPO) |
| I have no preference data but want better outputs | **DPO** (see [04 — Training](04_training.md)) |
| I'm just starting out | **SFT only** is fine for most use cases |

---

## Next Step

→ [06 — Inference](06_inference.md): Test and serve your model.
