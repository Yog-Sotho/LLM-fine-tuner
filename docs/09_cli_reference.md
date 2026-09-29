# 09 — CLI Reference

The CLI (Command Line Interface) lets you run every feature of LLM Fine-Tuner from your terminal — no browser or GUI needed. This is perfect for servers, automation, scripts, and scheduled jobs.

---

## How It Works

Any time you launch `main.py` with arguments, it runs in CLI mode. With no arguments, it launches the Gradio UI.

```bash
python main.py            # → opens the Gradio UI in your browser
python main.py --help     # → shows CLI help
python main.py --version  # → prints the version
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

  🧠 LLM Fine-Tuner v3.2.0 — headless CLI for every training mode.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  train     Headless SFT training
  reward    Train a Reward Model from preference data
  orpo      ORPO alignment training
  grpo      GRPO fine-tuning with a reward model and/or reference answers
  kto       KTO alignment from desirable / undesirable examples
  evaluate  Batched BLEU / ROUGE / BERTScore evaluation (greedy decoding)
  benchmark Standard benchmarks with lm-evaluation-harness (ARC, HellaSwag, GSM8K, …)
  merge     Merge a LoRA adapter into its base model
  export    Export for deployment: GGUF or FP8/W4A16 safetensors (vLLM)
  push      Upload a model folder to the Hugging Face Hub
  serve     Serve a model behind an OpenAI-compatible API
```

Each command also has its own `--help`:
```bash
python main.py train --help
```

The app logs to the console. `LFT_LOG_LEVEL=DEBUG` (or `WARNING`, `ERROR`; default `INFO`) sets how
much its own modules report; other libraries only show warnings.

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
| `--lora-variant` | `LoRA` | `LoRA`, `rsLoRA` or `DoRA` (LoRA goes on every linear layer) |
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

Online reinforcement learning: several answers are generated per prompt and the ones that beat their group's average reward are reinforced. At least one reward source is required: `--reward-model` and/or built-in rewards (`--reward`). Without `--reward`, the reference-answer reward is used when the data has a `reference` column.

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
| `--resume` | off | Continue from the newest checkpoint in `--output` (saved every 50 steps) |
| `--reward` | see above | Built-in reward, repeatable: `reference`, `math`, `think_format`, `json`, `regex` |
| `--regex` | — | Pattern for `--reward regex` (the whole output must match) |
| `--loss-type` | `dapo` | `dapo`, `dr_grpo`, `grpo` or `bnpo` |
| `--lora-rank` / `--lora-alpha` | `16` / `32` | LoRA size |
| `--lora-variant` | `LoRA` | `LoRA`, `rsLoRA` or `DoRA` |
| `--use-vllm` | off | Generate with vLLM on the training GPU (CUDA; `pip install "trl[vllm]"`) |

Example — reward JSON output that also matches a pattern:
```bash
python main.py grpo --policy-model ./sft_merged --data prompts.csv \
    --reward json --reward regex --regex '\{"answer": \d+\}' --loss-type dr_grpo
```

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
| `--resume` | off | Continue from the newest checkpoint in `--output` (saved every 50 steps) |

**Data format** (`feedback.csv`):
```csv
prompt,completion,label
"Summarise this email","Meeting moved to Friday at 3pm.",true
"Summarise this email","lol idk",false
```

---

### `evaluate` — Batch Evaluation

Runs BLEU, ROUGE, and optionally BERTScore on your model. Generation is greedy
(deterministic), so two runs — or two models — can be compared fairly.

```bash
python main.py evaluate \
    --model mistralai/Mistral-7B-v0.1 \
    --lora ./my_model \
    --data eval.csv \
    --compare-base \
    --max-new-tokens 150
```

**All options:**

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model ID or path |
| `--data` | *(required)* | CSV/JSONL with a `prompt` column and optionally `reference` |
| `--lora` | *(optional)* | PEFT adapter path |
| `--compare-base` | off | Also score the base model with the LoRA adapter switched off (needs `--lora`) |
| `--bertscore` | off | Compute BERTScore (slower) |
| `--batch-size` | `8` | Generation batch size |
| `--max-new-tokens` | `150` | Max tokens per response |

**Output** (with `--compare-base`):
```
📊 EVALUATION RESULTS
==================================================
metric           fine-tuned        base         Δ
BLEU-1                0.412       0.305     0.107
ROUGE-1               0.674       0.551     0.123
...

✅ Evaluation complete — 50 examples
💾 Saved to: eval_results_20260308_143012.csv
```

The CSV holds `prompt`, `prediction`, `base_prediction` (with `--compare-base`) and `reference`.

---

### `benchmark` — Standard Benchmarks

Scores a model on standard benchmarks through
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
(`pip install "lm-eval>=0.4.13,<0.5"`, included in the `eval` extra).

```bash
python main.py benchmark \
    --model mistralai/Mistral-7B-v0.1 \
    --lora ./my_model \
    --tasks arc_easy,hellaswag \
    --limit 200 \
    --compare-base \
    --output scores.csv
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model ID or path |
| `--tasks` | `arc_easy` | Comma-separated: `arc_easy`, `arc_challenge`, `hellaswag`, `piqa`, `winogrande`, `boolq`, `truthfulqa_mc2`, `gsm8k` |
| `--lora` | *(optional)* | PEFT adapter path (safetensors) |
| `--limit` | `100` | Examples per benchmark (1–10 000). Small limits give rough estimates |
| `--compare-base` | off | Also score the base model (needs `--lora`) |
| `--batch-size` | `8` | Evaluation batch size |
| `--output` | *(optional)* | Save the scores table as CSV |

Scores are fractions (1.0 = 100%). Benchmark datasets are downloaded from the
Hugging Face Hub on first use.

---

### `merge` — Merge a LoRA Adapter

```bash
python main.py merge --adapter ./runs/my-run --output ./runs/my-run-merged
```

`--base` overrides the base model named in `adapter_config.json`.

---

### `export` — Export for Deployment

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Trained model or LoRA adapter folder |
| `--output` | *(required)* | Output folder |
| `--format` | `gguf` | `gguf`, `fp8` or `w4a16` |
| `--quant` | `q6_k` | GGUF quantisation (`q8_0`, `q6_k`, `q5_k_m`, `q4_k_m`) |
| `--calibration-data` | — | `w4a16`: CSV/JSONL of training-like examples |
| `--calibration-samples` | `128` | `w4a16`: how many examples to use |

GGUF needs Unsloth or llama.cpp; `fp8`/`w4a16` need `pip install "llm-fine-tuner[compress]"`.
See [08 — Export & Deploy](08_export_and_deploy.md).

---

### `push` — Upload to the Hub

```bash
HF_TOKEN=hf_... python main.py push --model ./runs/my-run --repo my-user/my-model
```

Creates the repo if needed and completes the model card (canonical base model, license).
`--token` works too, but the environment keeps the token out of your shell history.

---

### `serve` — OpenAI-compatible Server

```bash
LFT_SERVE_API_KEY=secret python main.py serve --model ./model.gguf --host 0.0.0.0 --port 8000
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | `.gguf` file (llama-server), model/adapter folder or Hub id (vLLM) |
| `--host` | `127.0.0.1` | Bind address |
| `--port` | `8000` | Port |
| `--name` | `model` | Model name clients send |

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
