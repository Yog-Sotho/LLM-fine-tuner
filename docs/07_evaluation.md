# 07 — Evaluation

Evaluation tells you how good your trained model actually is by comparing its responses to known correct answers using automatic scoring metrics.

---

## The 📊 Evaluation Tab

The evaluation suite supports four metrics out of the box. You don't need to understand the maths — just know what each one is good for.

---

## The Four Metrics Explained (Simply)

### BLEU — "Does it use the right words?"

BLEU (Bilingual Evaluation Understudy) checks how many words and short phrases in the model's response match the reference answer. Originally designed for translation, it works for any task where exact wording matters.

- Score range: **0.0 to 1.0** (higher is better)
- A score of **0.3** is considered decent; **0.5+** is strong
- Best for: translation, structured output, factual Q&A

**Example:**
```
Reference:  "The capital of France is Paris."
Model output: "Paris is the capital of France."
BLEU: ~0.4   (same words, different order — BLEU penalises this slightly)
```

### ROUGE — "Does it cover the key points?"

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures how much of the reference answer is captured in the model's response. It comes in three flavours:

| Metric | Measures |
|---|---|
| ROUGE-1 | Single word overlap |
| ROUGE-2 | Two-word phrase overlap |
| ROUGE-L | Longest matching sequence |

- Score range: **0.0 to 1.0** (higher is better)
- Best for: summarisation, text generation, chatbots

### BERTScore — "Does it mean the same thing?"

BERTScore uses a language model to compare meaning, not just exact words. It can tell that "automobile" and "car" mean the same thing, even though BLEU can't.

- Score range: **0.0 to 1.0** (higher is better, usually 0.8–0.95 for good outputs)
- Slower to compute (uses a neural model internally)
- Best for: tasks where phrasing can vary but meaning should be consistent
- Requires: `pip install bert-score`

### LLM-as-Judge — "What does an AI think?"

This uses a separate language model (acting as a judge) to score your model's responses on multiple criteria. It's the most flexible metric but requires access to a capable judge model.

Criteria: **helpfulness, accuracy, coherence, safety, relevance**

---

## Running an Evaluation

### Step 1 — Prepare your evaluation dataset

You need a CSV with a `prompt` column and a `reference` column (the correct answers):

```csv
prompt,reference
"What year did World War II end?","World War II ended in 1945."
"What is the boiling point of water?","Water boils at 100°C (212°F) at standard atmospheric pressure."
"Who wrote Romeo and Juliet?","Romeo and Juliet was written by William Shakespeare."
"What is the square root of 144?","The square root of 144 is 12."
"Name the largest planet in our solar system.","Jupiter is the largest planet in our solar system."
```

> **Tip:** Use examples your model hasn't been trained on. Evaluation on training data is misleading — the model has already memorised those answers.

### Step 2 — Load your model

In the evaluation settings:
- **Model ID / Path** — your trained model or adapter path
- **PEFT Adapter** (optional) — if using a LoRA adapter, enter its path here

### Step 3 — Select metrics

Check the boxes for the metrics you want to compute:

- ✅ **BLEU** — always a good baseline, fast
- ✅ **ROUGE** — good for most generation tasks, fast
- ☐ **BERTScore** — tick this for a deeper semantic comparison (slower, ~2–5 min)
- ☐ **LLM Judge** — tick this if you have a capable judge model available

### Step 4 — Run

1. Upload your evaluation CSV
2. Click **🧪 Run Evaluation**

Results appear in a table:

```
Metric         Score
─────────────────────
BLEU           0.41
ROUGE-1        0.68
ROUGE-2        0.45
ROUGE-L        0.62
BERTScore F1   0.87
```

A downloadable CSV of all predictions and scores is saved automatically.

---

## Interpreting Results

| Metric | Poor | Acceptable | Good | Excellent |
|---|---|---|---|---|
| BLEU | < 0.1 | 0.1–0.3 | 0.3–0.5 | > 0.5 |
| ROUGE-1 | < 0.3 | 0.3–0.5 | 0.5–0.7 | > 0.7 |
| ROUGE-L | < 0.2 | 0.2–0.4 | 0.4–0.6 | > 0.6 |
| BERTScore | < 0.7 | 0.7–0.8 | 0.8–0.9 | > 0.9 |

> **Important:** These numbers are guidelines, not rules. A BLEU of 0.2 on a creative writing task can still be excellent if the model is producing fluent, relevant text. Always read some actual outputs alongside the numbers.

---

## How to Improve Low Scores

| Problem | Likely cause | Fix |
|---|---|---|
| Low BLEU and ROUGE | Model isn't learning the content | More training data, more epochs |
| Low BERTScore | Model responses are off-topic | Check your training data quality |
| High train scores, low eval scores | Overfitting | Reduce epochs, get more data |
| All scores low | Wrong model or format | Check column mapping and training mode |

---

## Evaluation via CLI

You can also run evaluation from the command line without the UI. See [09 — CLI Reference](09_cli_reference.md) for the `evaluate` command.

---

## Next Step

→ [08 — Export & Deploy](08_export_and_deploy.md): Share and deploy your model.
