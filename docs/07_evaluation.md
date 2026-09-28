# 07 — Evaluation

Evaluation tells you how good your trained model actually is by comparing its responses to known correct answers using automatic scoring metrics.

---

## The 🧪 Evaluation Tab

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

This uses a separate language model (acting as a judge) to rate each response from 1 to 10 on one criterion. It's the most flexible metric but needs a capable judge model — ideally an instruction-tuned one with a chat template.

Criteria: **helpfulness, accuracy, coherence, safety, relevance**

The judge is asked to answer `Score: N`. The metrics show the **mean score** and, if some replies had no readable score, how many; the results table has each `judge_score` and the full `judgment` text. Small or base models often fail to follow the format — use a stronger judge if many replies have no score.

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
- **Model to Evaluate / custom model** — the base model ID or a local model path
- **PEFT Adapter Path** (optional) — if using a LoRA adapter, enter its path here
- **Compare with the base model** (optional, LoRA only) — also generates every answer with the adapter switched off, so you see whether fine-tuning helped

### Step 3 — Select metrics

**BLEU** and **ROUGE** are always computed when the dataset has a `reference` column. Optionally:

- ☐ **Compute BERTScore** — a deeper semantic comparison (slower, ~2–5 min)
- ☐ **Run LLM-as-Judge** — enter a **Judge Model ID** and pick a **Judge Criterion**

### Step 4 — Run

1. Upload your evaluation CSV
2. Click **🧪 Run Evaluation**

Answers are generated greedily (no sampling), so the same model gives the same answers every run and comparisons are fair. With **Compare with the base model** ticked, results appear side by side:

```
| Metric             | Fine-tuned | Base  | Δ       |
|--------------------|------------|-------|---------|
| BLEU-1             | 0.41       | 0.30  | +0.1100 |
| ROUGE-L            | 0.62       | 0.48  | +0.1400 |
| Judge score (1-10) | 7.2        | 5.9   | +1.3000 |
```

The table below lists every prompt with its `prediction`, `base_prediction`, `reference` and judge scores.

---

## Standard Benchmarks

The **📏 Standard benchmarks** section of the same tab runs well-known multiple-choice and maths benchmarks through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), using the model and adapter chosen above:

| Benchmark | Tests |
|---|---|
| ARC-Easy / ARC-Challenge | Grade-school science questions |
| HellaSwag | Commonsense sentence completion |
| PIQA | Physical commonsense |
| WinoGrande | Pronoun resolution |
| BoolQ | Yes/no reading comprehension |
| TruthfulQA (MC2) | Avoiding common misconceptions |
| GSM8K | Grade-school maths (generates answers; slow) |

1. Tick the benchmarks
2. Set **Examples per benchmark** — 100 gives a quick, rough estimate; more is slower but more reliable
3. Optionally tick **Compare with the base model** (needs an adapter path)
4. Click **📏 Run Benchmarks**

Scores are fractions (1.0 = 100%); `acc_norm` is accuracy normalised for answer length. Fine-tuning for a narrow task often lowers general benchmark scores slightly — compare with the base model to see by how much. Requires `pip install "lm-eval>=0.4.13,<0.5"` (part of the `eval` extra); benchmark datasets are downloaded from the Hub on first use.

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

You can also run evaluation from the command line without the UI. See [09 — CLI Reference](09_cli_reference.md) for the `evaluate` and `benchmark` commands.

---

## Next Step

→ [08 — Export & Deploy](08_export_and_deploy.md): Share and deploy your model.
