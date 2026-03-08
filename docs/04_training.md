# 04 — Training

This guide explains every option in the **🚀 Training** tab. You don't need to understand all of them — the defaults are carefully chosen to work well for most people.

---

## The Basic Workflow

1. Upload your data (📂 Data tab)
2. Choose a base model
3. Choose a training mode
4. Choose a preset (or adjust manually)
5. Click **▶ Start Training**
6. Watch the loss curve go down
7. Test in the 💬 Inference tab

---

## Choosing a Base Model

The base model is the pre-trained AI that you'll be teaching. Think of it as hiring a smart employee — you're not creating intelligence from scratch, you're specialising existing intelligence.

### Model List

| Model | VRAM needed | Speed | Quality | Good for |
|---|---|---|---|---|
| `distilgpt2` | < 1 GB | Very fast | Low | Testing only |
| `gpt2` | 1 GB | Fast | Low | Quick experiments |
| `facebook/opt-125m` | 1 GB | Fast | Low | Quick experiments |
| `facebook/opt-350m` | 2 GB | Medium | Medium | Small tasks |
| `EleutherAI/pythia-70m` | < 1 GB | Very fast | Low | Testing |
| `EleutherAI/pythia-160m` | 1 GB | Fast | Low | Quick experiments |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 4 GB | Medium | Good | Most tasks |
| `mistralai/Mistral-7B-v0.1` | 16 GB | Slow | Excellent | Production use |

> **Auto-recommendation:** The tool automatically suggests the best model for your hardware based on available VRAM. You'll see this recommendation pre-selected when you open the Training tab.

You can also type any model ID from [HuggingFace Hub](https://huggingface.co/models) into the **"Or enter any HuggingFace model ID"** box. For example: `meta-llama/Llama-2-7b-hf`.

---

## Training Mode

### SFT — Supervised Fine-Tuning (Default)

The standard mode. You show the model many examples of the behaviour you want, and it learns to replicate it.

Use this when:
- You want the model to answer questions in a specific style
- You're teaching domain-specific knowledge
- You're building a chatbot with a particular personality

**Required data columns:** `instruction` + `output`, or `text`

### DPO — Direct Preference Optimization

A more advanced technique. Instead of just showing correct answers, you show pairs of answers where one is better than the other. The model learns to prefer the good style.

Use this when:
- You want to align the model with human preferences
- You have existing SFT output and want to improve quality
- You want to make the model safer or more helpful

**Required data columns:** `prompt`, `chosen`, `rejected`

**DPO Beta:** Controls how strongly the model penalises bad responses. Default (0.1) works well for most cases. Higher values make the model more conservative.

---

## Training Presets

Presets are shortcuts that set the number of epochs and learning rate for you.

| Preset | Epochs | Learning Rate | Time | Use when |
|---|---|---|---|---|
| **Quick (1 epoch)** | 1 | 5×10⁻⁴ | Minutes | Testing, first runs |
| **Balanced (3 epochs)** | 3 | 2×10⁻⁴ | 30–60 min | Most real projects |
| **Accurate (5 epochs)** | 5 | 1×10⁻⁴ | 1–2 hours | Production models |
| **Advanced** | Custom | Custom | Depends | Expert tuning |

> **What is an epoch?** One epoch means the model has seen every example in your dataset once. More epochs = more learning, but too many causes overfitting (the model memorises instead of generalising).

---

## PEFT Methods (Parameter-Efficient Fine-Tuning)

Fine-tuning a large model means updating billions of parameters — which requires enormous VRAM. PEFT methods work around this by only updating a small fraction of the parameters.

| Method | VRAM usage | Speed | Quality | Use when |
|---|---|---|---|---|
| **Auto** | Varies | — | — | Let the tool decide (recommended) |
| **LoRA** | Low | Fast | Excellent | Default choice for most users |
| **QLoRA Enhanced** | Very low | Medium | Very good | Limited VRAM (< 8 GB) |
| **Full Fine-tuning** | Very high | Slow | Best possible | You have a lot of VRAM (> 40 GB) |
| **Prefix Tuning** | Very low | Fast | Good | Extremely limited hardware |
| **Prompt Tuning** | Minimal | Very fast | Moderate | Fastest possible training |
| **Adapters** | Low | Fast | Good | Alternative to LoRA |

### LoRA — The Recommended Choice

LoRA (Low-Rank Adaptation) is the most popular PEFT method. It inserts small trainable matrices into the model and only updates those. The result is:
- 90%+ less VRAM than full fine-tuning
- Only slightly worse quality
- Very fast training

**LoRA Settings (in Advanced Hyperparameters):**
- **LoRA Rank** (default: 8) — Higher = more capacity to learn, more VRAM. Try 16 or 32 for better quality.
- **LoRA Alpha** (default: 16) — Usually set to 2× the rank. Controls the strength of the adaptation.

### QLoRA Enhanced — For Low-VRAM Machines

QLoRA Enhanced uses 4-bit quantisation (NF4 format) to load the model in a fraction of the VRAM, then applies LoRA on top. This lets you fine-tune a 7B model on a GPU with just 8 GB of VRAM.

> **How to enable:** Select `QLoRA Enhanced` from the PEFT Method radio buttons. The NF4 settings are applied automatically.

---

## Additional Training Options

### Unsloth (🚀 2–5× faster)

Unsloth is a library that dramatically speeds up training and reduces VRAM usage. If you installed it during setup, check this box for every training run. There's no downside.

> Unsloth is only compatible with LoRA training on NVIDIA GPUs.

### Smart Chat Template (💬 recommended ON)

Automatically formats your data using the correct chat format for the model you chose (e.g. Llama-3 uses `<|im_start|>` tags, Mistral uses `[INST]` blocks). This improves output quality significantly for chat models.

### Flash Attention 2 (⚡ for modern GPUs)

A highly optimised attention algorithm that's 2–3× faster and uses less VRAM. Requires:
- An NVIDIA GPU with Ampere architecture or newer (RTX 3000 series or later)
- bfloat16 precision support

### System Prompt

The text the model will use as its "identity" during inference. For example:
```
You are a friendly customer support agent for Acme Corp. Always be polite and helpful.
```

### Heretic Mode 🔓

See [10 — Advanced Usage](10_advanced.md) for a full explanation. In short: removes built-in safety restrictions from the model after training.

---

## Advanced Hyperparameters

Click **⚙️ Advanced hyperparameters** to expand these options. You only need these if you want precise control.

| Parameter | Default | What it does |
|---|---|---|
| **Learning Rate** | 2×10⁻⁴ | How fast the model updates its weights. Too high → unstable training. Too low → slow learning. |
| **Epochs** | 3 | How many times the model sees the full dataset |
| **Batch Size** | 2 | How many examples processed at once. Higher = faster but uses more VRAM. |
| **Gradient Accumulation** | 4 | Simulates a larger batch size without using extra VRAM. |
| **Max Sequence Length** | 256 | Maximum number of tokens per example. Longer = more VRAM. |
| **Warmup Steps** | 100 | Slowly ramps up the learning rate at the start to avoid unstable early training. |
| **Early Stopping Patience** | 3 | Stops training early if the model stops improving. 0 = disabled. |
| **LR Scheduler** | cosine | How the learning rate changes over time. `cosine` is best for most cases. |
| **Gradient Checkpointing** | OFF | Saves VRAM at the cost of ~20% slower training. Enable if you're running out of VRAM. |
| **Resume from checkpoint** | OFF | Continue a previous training run that was interrupted. |

---

## Reading the Training Log

The log on the right side shows:

```
🚀 Loading model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
✅ Model loaded (PEFT: LoRA, rank=8)
📊 Dataset: 450 examples → 360 train / 90 eval
🏃 Epoch 1/3 — Step 50/113 — Train loss: 1.847 — Eval loss: 1.923
🏃 Epoch 2/3 — Step 100/113 — Train loss: 1.432 — Eval loss: 1.541
🏃 Epoch 3/3 — Step 113/113 — Train loss: 1.201 — Eval loss: 1.318
✅ Training complete! Time: 0:18:42
📁 Model saved to: /tmp/abc123
```

**What to watch for:**
- **Train loss decreasing** — the model is learning ✅
- **Eval loss decreasing too** — it's generalising, not just memorising ✅
- **Eval loss rising while train loss falls** — overfitting. Reduce epochs or get more data. ⚠️

The **📉 Loss Curve** below the log plots these values visually.

---

## Stopping Training Early

Click **⏹ Stop** at any time. The model will finish the current step, save what it has, and stop. Your partially-trained model is still usable.

---

## Clearing GPU Memory

If you get out-of-memory errors or want to free up VRAM between runs, click **🧹 Clear GPU Cache**. This releases any cached tensors without restarting the app.

---

## Next Step

→ [05 — RLHF Pipeline](05_rlhf_pipeline.md): Advanced alignment with Reward Models, PPO, and ORPO.  
→ [06 — Inference](06_inference.md): Test your trained model.
