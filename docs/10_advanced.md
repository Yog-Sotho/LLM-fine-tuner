# 10 — Advanced Usage

This guide covers the more powerful and niche features of LLM Fine-Tuner for users who want maximum control or are pushing the limits of their hardware.

---

## Heretic Mode 🔓

### What is it?

Many language models have built-in restrictions that make them refuse certain questions or respond in heavily filtered ways. These restrictions are added during a phase called "alignment" (usually RLHF or RLAIF) by the model's creators.

Heretic Mode uses a technique called **abliteration** to remove these restrictions after fine-tuning. The result is a model that responds more directly and is less likely to refuse questions.

> **Use responsibly.** Removing safety restrictions means the model can produce harmful content. Only use this for legitimate research, testing, or private deployments where you control access.

### How to Enable

In the **🚀 Training** tab, scroll to the Advanced section and check **🔓 Heretic Mode**.

That's it. The abliteration is applied automatically at the end of your training run.

### Technical Details

Heretic Mode uses the `heretic-llm` library, which identifies and removes the "refusal direction" from the model's residual stream. This is different from just removing RLHF — it directly modifies the model's internal weights to reduce the influence of the trained refusal behaviour.

- Requires: `pip install heretic-llm>=1.2.0` (installed by default)
- Works with: LoRA, QLoRA, Full Fine-tuning
- Source: [github.com/Yog-Sotho/LLM-fine-tuner](https://github.com/Yog-Sotho/LLM-fine-tuner)

---

## Flash Attention 2 ⚡

Flash Attention 2 is a mathematically equivalent but much more efficient implementation of the attention mechanism used in transformers. It processes attention in tiles rather than loading the full matrix into VRAM, resulting in:

- **2–3× faster training**
- **3–5× less VRAM used during attention computation**

### Requirements

| Requirement | Details |
|---|---|
| GPU | NVIDIA Ampere or newer (RTX 3000 series or later) |
| VRAM | At least 8 GB |
| Precision | bfloat16 (automatically enforced when Flash Attention is on) |
| Package | `pip install flash-attn --no-build-isolation` |

### How to Enable

In the Training tab, check **⚡ Flash Attention 2**. The tool will automatically:
1. Check if your GPU supports bfloat16
2. Set the dtype accordingly
3. Enable Flash Attention in the model config

If your GPU doesn't support it, you'll see a warning and training will fall back to standard attention.

---

## QLoRA Enhanced — Deep Dive

QLoRA Enhanced combines two techniques:

**4-bit NF4 Quantisation** — loads the base model weights in 4-bit NF4 (Normal Float 4) format instead of the default 16-bit. This cuts VRAM usage by ~75% for the base model.

**Double Quantisation** — quantises the quantisation constants themselves, saving another ~0.4 bits per parameter. Sounds small, but on a 7B model that's ~4 GB saved.

On top of this compressed base, LoRA adapters are still trained in full 16-bit precision, so the trainable parameters retain their quality.

**The result:** fine-tune a 7B model on a GPU with 8 GB VRAM.

### Settings Used Automatically

When you select `QLoRA Enhanced`:

```python
# BitsAndBytes config (applied automatically)
load_in_4bit = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = bfloat16
bnb_4bit_use_double_quant = True

# LoRA config (applied automatically)
r = 64  # Higher rank than standard LoRA
lora_alpha = 128
target_modules = "all-linear"  # every linear layer of the transformer blocks
lora_dropout = 0.05
```

You don't need to enter these manually — just select `QLoRA Enhanced` in the PEFT method radio.

### Limitation

QLoRA Enhanced requires a CUDA GPU. If you're on CPU or a non-NVIDIA GPU, the tool will automatically fall back to standard LoRA with a clear warning message.

---

## Unsloth — Maximum Speed

Unsloth rewrites the core training kernels in Triton (a GPU programming language) to be 2–5× faster than the standard HuggingFace implementation. It also reduces VRAM usage by 60–80%.

| Feature | HuggingFace | Unsloth |
|---|---|---|
| Training speed (7B, LoRA) | Baseline | 2–5× faster |
| VRAM usage | Baseline | 60–80% less |
| GGUF export | Via llama.cpp | Native (faster) |
| Maximum context | Limited by VRAM | Extended |

### How to Enable

Check **🚀 Enable Unsloth** in the Training tab. That's it — the rest is automatic.

Unsloth is only available for LoRA and QLoRA training on NVIDIA GPUs. If you try to enable it with Full Fine-Tuning or without a NVIDIA GPU, you'll see a warning.

---

## Smart Chat Templates 💬

Modern instruct models (Llama-3, Mistral, Qwen, Gemma-2, Phi, etc.) expect prompts formatted in a specific way. For example:

**Llama-3 format:**
```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is photosynthesis?<|im_end|>
<|im_start|>assistant
```

**Mistral format:**
```
[INST] What is photosynthesis? [/INST]
```

When **Smart Chat Template** is enabled, the tool automatically applies the correct format for the model you chose. This usually improves output quality significantly for chat and instruction-following tasks.

When it's disabled, your data is used as-is.

---

## Multi-GPU Training

Training runs **data-parallel** across GPUs when you start the CLI with `accelerate launch`
(or `torchrun`): each GPU trains its own copy of the model on its share of every batch, and
the gradients are averaged, so the copies stay identical.

```bash
# All GPUs of this machine (after a one-time `accelerate config`, or pass flags):
accelerate launch --multi_gpu --num_processes 4 main.py train \
    --model Qwen/Qwen2.5-7B-Instruct --data train.jsonl --output ./runs/multi

# Same with torchrun:
torchrun --nproc_per_node 4 main.py train --model ... --data ... --output ...
```

Works for `train` (SFT/DPO, including vision-language data), `reward`, `orpo`, `kto` and `grpo`.

- **Batch size** is per GPU: the effective batch is `--batch-size` × GPUs × gradient
  accumulation. With 4 GPUs, divide `--batch-size` (or accumulation) by 4 to keep the same
  training recipe.
- **QLoRA / 4-bit:** each process loads its own 4-bit copy onto *its* GPU.
- **Outputs** (adapter, `run_config.yaml`, model card) are written once, by the main process.
- **The web UI is single-process.** Launching `main.py` without a command under
  `accelerate launch`/`torchrun` stops with a message instead of starting one UI per GPU.
- **Not supported yet:** model sharding (FSDP, DeepSpeed ZeRO-3) for models that don't fit on
  one GPU — every GPU needs a full copy (4-bit helps).

Verified with two processes on CPU (gloo), where both copies end with identical weights; the
multi-GPU (NCCL) path uses the same code but hasn't been run in this project's CI.

### GPU job queue (web UI)

In the web UI, training, evaluation, benchmarks, GGUF export, adapter merging and vLLM
generation share **one queue**: if a job is running, the next one waits (Gradio shows its
place in line) instead of loading a second model onto the same GPU. Stop still works while a
job runs, and the quick Generate buttons stay outside the queue. Set `LFT_GPU_JOBS=2` (1–8) to
let more jobs run at once, e.g. on a multi-GPU machine.

---

## Gradient Checkpointing

Gradient checkpointing trades speed for memory. Instead of storing all intermediate activations during the forward pass (needed for backpropagation), it recomputes them during the backward pass.

- **Effect:** ~30–40% less VRAM used during training
- **Cost:** ~20% slower training
- **When to use:** If you're getting CUDA out-of-memory errors

Enable with the **Gradient Checkpointing** checkbox in the Advanced Hyperparameters section.

---

## Custom System Prompts

The system prompt is injected at the start of every training example when **Smart Chat Template** is on. It shapes how the model introduces itself and behaves.

**Examples:**

```
# Customer support bot
You are a helpful support agent for Acme Corp. Always be polite, concise, and solution-focused.

# Medical information assistant  
You are a medical information assistant. Provide accurate, evidence-based information. Always recommend consulting a qualified doctor for personal medical advice.

# Code reviewer
You are an expert code reviewer. Identify bugs, suggest improvements, and explain your reasoning clearly.
```

---

## Learning Rate Schedulers

The learning rate scheduler controls how the learning rate changes over the course of training.

| Scheduler | Behaviour | When to use |
|---|---|---|
| `cosine` | Smoothly decreases following a cosine curve | Best for most cases — default |
| `linear` | Decreases linearly to zero | Simple, predictable |
| `constant` | Never changes | Rarely useful |
| `cosine_with_restarts` | Cosine with periodic increases | Long training runs |

For most users: **leave it on `cosine`**.

---

## Resuming an Interrupted Training Run

If training was interrupted (power cut, crash, timeout), you can resume from the last checkpoint:

1. In the Training tab, enable **Resume from checkpoint**
2. Set the **Output directory** to the same path as the interrupted run
3. Start training

The tool will find the latest checkpoint automatically and continue from there.

---

## Next Step

→ [11 — Troubleshooting](11_troubleshooting.md): Fix the most common errors.
