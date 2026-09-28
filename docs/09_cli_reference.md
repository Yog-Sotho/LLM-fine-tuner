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
  grpo      GRPO fine-tuning with a reward model and/or reference answers
  kto       KTO alignment from desirable / undesirable examples
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
| `--packing` | off | Pack short samples into full-length sequences (needs `--flash-attn` on a CUDA GPU; ignored otherwise) |
| `--hf-dataset` | — | Hugging Face Hub dataset (`owner/name`) instead of `--data`; streamed |
| `--hf-config` / `--hf-split` | — / `train` | Hub dataset config and split |
| `--hf-max-rows` | `20000` | Rows to stream from the Hub dataset |
| `--eval-split` | `0.1` | Share of rows held out for evaluation (`0` = none) |
| `--seed` | `42` | Random seed for the data split and training (same seed + data + settings = same weights) |
| `--report-to` | `none` | Experiment tracker: `trackio`, `wandb`, `mlflow` or `tensorboard` (must be installed) |
| `--config` | — | Replay a saved `run_config.yaml`: model, mode (SFT/DPO) and all settings come from the file; only `--data`/`--output` are used. Warns if the data differs from the recorded fingerprint |

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

Trains a prompt-aware reward model (a sequence classifier with one score output) from preference pairs. LoRA is merged at the end, so the output folder can be passed straight to `grpo --reward-model`.

```bash
python main.py reward \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data reward_pairs.csv \
    --output ./reward_model \
    --epochs 3 \
    --lr 1e-4 \
    --max-length 1024 \
    --batch-size 4
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Base model ID |
| `--data` | *(required)* | CSV/JSONL with `prompt`, `chosen` and `rejected` columns |
| `--output` | `./reward_model` | Where to save the reward model |
| `--epochs` | `3` | Training epochs |
| `--lr` | `1e-4` | Learning rate (LoRA) |
| `--max-length` | `1024` | Max tokens for prompt + response |
| `--batch-size` | `4` | Batch size |

**Data format** (`reward_pairs.csv`):
```csv
prompt,chosen,rejected
"How long is a year?","About 365.25 days.","No idea."
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

### `grpo` — GRPO Fine-Tuning

Online reinforcement learning: several answers are generated per prompt and the ones that beat their group's average reward are reinforced. At least one reward source is required: `--reward-model` and/or a `reference` column in the data.

```bash
python main.py grpo \
    --policy-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --reward-model ./reward_model \
    --data prompts.csv \
    --output ./grpo_model \
    --num-generations 4 \
    --max-completion-length 128
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--policy-model` | *(required)* | Model to train (HF ID or full model folder) |
| `--data` | *(required)* | CSV/JSONL with `prompt`; optional `reference` (expected answer) |
| `--reward-model` | — | Folder from the `reward` command |
| `--output` | `./grpo_model` | Where the LoRA adapter is saved |
| `--epochs` | `1` | Passes over the prompts |
| `--lr` | `1e-5` | Learning rate |
| `--num-generations` | `4` | Completions per prompt (≥ 2) |
| `--prompts-per-step` | `1` | Prompts per optimisation step |
| `--max-completion-length` | `128` | Tokens generated per completion |
| `--beta` | `0.0` | KL penalty towards the original model (0 = off) |

**Data format** (`prompts.csv`):
```csv
prompt,reference
"What is 12 × 12?","144"
"What is the capital of France?","Paris"
```

A reference-match reward gives 1 when the completion contains the `reference` text (case-insensitive).

---

### `kto` — KTO Alignment

Aligns from single responses labelled good or bad — no ranked pairs needed.

```bash
python main.py kto \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data feedback.csv \
    --output ./kto_model \
    --batch-size 4
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Base model ID |
| `--data` | *(required)* | `prompt`,`completion`,`label` or `prompt`,`chosen`,`rejected` |
| `--output` | `./kto_model` | Where the LoRA adapter is saved |
| `--epochs` | `1` | |
| `--lr` | `5e-5` | Learning rate |
| `--beta` | `0.1` | How far the model may move from the original |
| `--batch-size` | `4` | At least 2 |
| `--max-length` | `512` | Max tokens for prompt + response |

**Data format** (`feedback.csv`):
```csv
prompt,completion,label
"Summarise this email","Meeting moved to Friday at 3pm.",true
"Summarise this email","lol idk",false
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

echo "Step 3: GRPO alignment"
# The policy must be a full model: use the base model, or merge the SFT adapter
# first (Inference tab → Merge Adapter) and pass the merged folder.
python main.py grpo \
    --policy-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --reward-model ./models/reward \
    --data data/prompts.csv \
    --output ./models/final

echo "Step 4: Evaluate"
python main.py evaluate \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --lora ./models/final \
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
