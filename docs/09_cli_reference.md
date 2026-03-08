# 09 — CLI Reference

The CLI (Command Line Interface) lets you run every feature of LLM Fine-Tuner from your terminal — no browser or GUI needed. This is perfect for servers, automation, scripts, and scheduled jobs.

---

## How It Works

Any time you launch `main.py` with arguments, it runs in CLI mode. With no arguments, it launches the Gradio UI.

```bash
python main.py            # → opens the Gradio UI in your browser
python main.py --help     # → shows CLI help
python main.py train ...  # → runs headless training
```

---

## Global Help

```bash
python main.py --help
```

Output:
```
Usage: main.py [OPTIONS] COMMAND [ARGS]...

  🧠 LLM Fine-Tuner v3.2 — headless CLI for every training mode.

Commands:
  train     Headless SFT training
  reward    Train a Reward Model from preference data
  orpo      ORPO alignment training
  ppo       PPO fine-tuning with a trained reward model
  evaluate  Batched BLEU / ROUGE / BERTScore evaluation
```

Each command also has its own `--help`:
```bash
python main.py train --help
```

---

## Commands

---

### `train` — Supervised Fine-Tuning

The main training command. Reuses exactly the same pipeline as the UI.

```bash
python main.py train \
    --model mistralai/Mistral-7B-v0.1 \
    --data train.csv \
    --output ./my_model \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4 \
    --peft LoRA \
    --lora-rank 8
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Base model ID or local path |
| `--data` | *(required)* | Dataset file (`.csv` or `.jsonl`) |
| `--output` | `./output` | Where to save the trained model |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `2` | Per-device batch size |
| `--max-length` | `256` | Maximum sequence length in tokens |
| `--lr` | `2e-4` | Learning rate |
| `--peft` | `LoRA` | PEFT method: `LoRA`, `QLoRA Enhanced`, `Full Fine-tuning`, `Auto` |
| `--lora-rank` | `8` | LoRA rank |
| `--qlora-enhanced` | off | Enable QLoRA Enhanced (overrides `--peft`) |
| `--flash-attn` | off | Enable Flash Attention 2 |

**Example — fine-tune TinyLlama on a JSONL dataset:**
```bash
python main.py train \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data my_qa_data.jsonl \
    --output ./tinyllama_finetuned \
    --epochs 5 \
    --lr 1e-4
```

**Example — QLoRA Enhanced for low-VRAM machines:**
```bash
python main.py train \
    --model mistralai/Mistral-7B-v0.1 \
    --data data.csv \
    --output ./mistral_qlora \
    --qlora-enhanced \
    --lora-rank 32
```

---

### `reward` — Train a Reward Model

Trains a reward model from preference (chosen/rejected) data.

```bash
python main.py reward \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data reward_pairs.csv \
    --output ./reward_model \
    --epochs 3 \
    --lr 1.4e-5 \
    --max-length 1024 \
    --batch-size 4
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Base model ID |
| `--data` | *(required)* | CSV/JSONL with `chosen` and `rejected` columns |
| `--output` | `./reward_model` | Where to save the reward model |
| `--epochs` | `3` | Training epochs |
| `--lr` | `1.4e-5` | Learning rate |
| `--max-length` | `1024` | Max sequence length |
| `--batch-size` | `4` | Batch size |

**Data format** (`reward_pairs.csv`):
```csv
chosen,rejected
"A detailed and accurate answer.","A vague or wrong answer."
```

---

### `orpo` — ORPO Alignment

Single-step preference alignment — no reward model needed.

```bash
python main.py orpo \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data preference_pairs.csv \
    --output ./orpo_model \
    --epochs 3 \
    --lr 1e-4 \
    --beta 0.1 \
    --alpha 0.1 \
    --batch-size 2
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Base model ID |
| `--data` | *(required)* | CSV/JSONL with `prompt`, `chosen`, `rejected` columns |
| `--output` | `./orpo_model` | Output directory |
| `--epochs` | `3` | |
| `--lr` | `1e-4` | Learning rate |
| `--beta` | `0.1` | ORPO loss weight |
| `--alpha` | `0.1` | SFT vs preference balance |
| `--batch-size` | `2` | |

---

### `ppo` — PPO Fine-Tuning

Reinforcement learning step using a trained reward model.

```bash
python main.py ppo \
    --policy-model ./my_sft_model \
    --reward-model ./reward_model \
    --data prompts.csv \
    --output ./ppo_model \
    --epochs 1 \
    --lr 1.4e-5 \
    --batch-size 1 \
    --max-new-tokens 128
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--policy-model` | *(required)* | SFT model path or HF ID |
| `--reward-model` | *(required)* | Path to trained reward model |
| `--data` | *(required)* | CSV/JSONL with `prompt` column |
| `--output` | `./ppo_model` | Output directory |
| `--epochs` | `1` | PPO epochs |
| `--lr` | `1.4e-5` | Learning rate |
| `--batch-size` | `1` | Keep at 1–2 (PPO is memory-intensive) |
| `--mini-batch-size` | `1` | Must be ≤ batch size |
| `--max-new-tokens` | `128` | Max tokens to generate per prompt |

**Data format** (`prompts.csv`):
```csv
prompt
"What is a healthy diet?"
"Explain machine learning simply."
"How do I manage stress?"
```

---

### `evaluate` — Batch Evaluation

Runs BLEU, ROUGE, and optionally BERTScore on your model.

```bash
python main.py evaluate \
    --model ./my_model \
    --data eval.csv \
    --lora ./my_model \
    --bertscore \
    --batch-size 4 \
    --max-new-tokens 150
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model ID or path |
| `--data` | *(required)* | CSV with `prompt` and `reference` columns |
| `--lora` | *(optional)* | PEFT adapter path (if separate from model) |
| `--bertscore` | off | Compute BERTScore (slower) |
| `--batch-size` | `4` | Generation batch size |
| `--max-new-tokens` | `150` | Max tokens per response |

**Output:**
```
📊 EVALUATION RESULTS
══════════════════════════════════════════════════
BLEU           : 0.412
ROUGE-1        : 0.674
ROUGE-2        : 0.441
ROUGE-L        : 0.618

✅ Evaluation complete — 50 examples
💾 Predictions saved to: eval_results_20260308_143012.csv
```

---

## Practical Automation Example

Run a full pipeline with a shell script:

```bash
#!/bin/bash
set -e

echo "Step 1: Fine-tune"
python main.py train \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data data/train.csv \
    --output ./models/sft \
    --epochs 3

echo "Step 2: Train reward model"
python main.py reward \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data data/reward.csv \
    --output ./models/reward \
    --epochs 2

echo "Step 3: PPO alignment"
python main.py ppo \
    --policy-model ./models/sft \
    --reward-model ./models/reward \
    --data data/prompts.csv \
    --output ./models/final \
    --epochs 1

echo "Step 4: Evaluate"
python main.py evaluate \
    --model ./models/final \
    --data data/eval.csv \
    --bertscore

echo "All done!"
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Error (message printed to stderr) |

---

## Next Step

→ [10 — Advanced Usage](10_advanced.md): Heretic Mode, hardware tips, and expert tuning.
