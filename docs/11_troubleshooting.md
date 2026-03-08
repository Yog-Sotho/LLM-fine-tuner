# 11 — Troubleshooting

This guide covers the most common errors, what they mean, and how to fix them.

---

## Installation Issues

---

### ❌ `Python 3.10+ required`

```
✘ Python 3.10+ required (found 3.9.x)
```

**Fix:** Install Python 3.11 from [python.org](https://www.python.org/downloads/). On Linux:
```bash
sudo apt install python3.11 python3.11-venv python3.11-pip
```

---

### ❌ `pip install` fails with network errors

```
ERROR: Could not find a version that satisfies the requirement ...
```

**Fixes:**
1. Check your internet connection
2. Try with a mirror: `pip install -r requirements.txt -i https://pypi.org/simple/`
3. If behind a corporate proxy: `pip install --proxy http://proxy:port -r requirements.txt`

---

### ❌ Flash Attention install fails

```
Failed to build flash-attn
```

Flash Attention requires a C++ compiler and CUDA toolkit to build from source. This is normal on some systems.

**Fix:** Flash Attention is optional. The installer handles this:
```
⚠ Flash Attention 2 optional – continuing
```
Training works fine without it — just slightly slower.

---

## Training Issues

---

### ❌ CUDA out of memory

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

This is the most common error for GPU users. Your model is too large for your VRAM.

**Fixes (try in this order):**

1. **Enable QLoRA Enhanced** — select it in the PEFT Method radio. Cuts VRAM by ~75%.
2. **Reduce Batch Size** — set it to 1 in Advanced Hyperparameters.
3. **Enable Gradient Checkpointing** — saves ~30% VRAM at the cost of ~20% speed.
4. **Reduce Max Sequence Length** — try 128 or 64 instead of 256.
5. **Use a smaller model** — switch from Mistral-7B to TinyLlama-1.1B.
6. **Clear GPU cache** — click **🧹 Clear GPU Cache** in the UI, then try again.

---

### ❌ Dataset is empty after validation

```
❌ Dataset empty after validation
```

All rows in your dataset were removed during cleaning.

**Fixes:**
1. Open your CSV in a spreadsheet editor and check for empty rows
2. Make sure the column names match what the tool expects (`instruction`, `output`, or `text`)
3. If your column names are different, use the Column Mapping dropdowns

---

### ❌ `KeyError: 'instruction'`

The tool expected a column called `instruction` but didn't find it.

**Fix:** Use the Column Mapping dropdowns in the Data tab to map your actual column names to the expected ones. Then click **🔄 Apply Mapping & Refresh Preview**.

---

### ❌ Training loss not decreasing

If the loss stays flat or barely changes after the first epoch:

**Likely causes and fixes:**

| Cause | Fix |
|---|---|
| Learning rate too low | Try `5e-4` or `1e-3` |
| Learning rate too high | Try `1e-5` |
| Too few examples | Add more training data |
| Data quality issues | Review your examples — are the outputs consistent? |
| Wrong training mode | Make sure you're using SFT for instruction data |

---

### ❌ Training loss goes to 0 immediately

If loss hits 0.0 after just a few steps, the model is memorising your data, not learning from it.

**Likely cause:** Very small dataset (< 10 examples).

**Fix:** Add more training data. At minimum, aim for 50 examples. Also reduce epochs to 1 for tiny datasets.

---

### ❌ `ValueError: DPO requires columns: prompt, chosen, rejected`

You selected DPO training mode but your dataset doesn't have the right columns.

**Fix:** Either:
1. Switch to SFT training mode if your data has `instruction`/`output` columns
2. Add the missing columns to your dataset

---

### ❌ `Install: pip install trl>=0.7.0`

The reward trainer or PPO/ORPO isn't available because `trl` is too old or not installed.

**Fix:**
```bash
pip install trl>=0.8.0 --upgrade
```

---

### ❌ Unsloth warning: `Unsloth + non-LoRA emits warning`

```
⚠️ Unsloth requires LoRA. Falling back to standard training.
```

**This is not an error.** Unsloth only supports LoRA. If you selected Full Fine-tuning or Adapters, it falls back gracefully. Change the PEFT method to LoRA to use Unsloth.

---

## Inference Issues

---

### ❌ Model generates blank or repetitive output

```
Output: ,,,,,,,,,,,,,,,,,,,,
```

**Fixes:**
1. **Increase temperature** — set it to 0.7–0.9 (if it was very low)
2. **Increase max new tokens** — the default (150) might be cutting off the response
3. **Check the adapter path** — make sure the adapter was loaded correctly
4. **Try a fresh prompt** — some prompts are out of distribution

---

### ❌ Model ignores instructions and talks about the wrong thing

The model isn't following the format of your prompts.

**Likely cause:** During training, **Smart Chat Template** was enabled/disabled but you're now prompting in a different format.

**Fix:** Use the same prompt format in inference as you used in training. If you trained with Smart Chat Template on, wrap your prompt the same way:
```
[INST] Your question here [/INST]   # Mistral format
```

---

### ❌ vLLM fails to start

```
❌ vLLM requires CUDA and a merged model
```

**Fixes:**
1. Make sure you have a CUDA GPU: `python -c "import torch; print(torch.cuda.is_available())"`
2. Make sure you're passing a **merged** model path (see [06 — Inference](06_inference.md))
3. Check vLLM is installed: `pip install vllm`

---

## Export Issues

---

### ❌ GGUF export fails

```
❌ GGUF export failed: llama.cpp not found
```

**Fix:** Install llama.cpp:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j$(nproc)
```
Or re-run the installer and say yes to the llama.cpp step.

---

### ❌ HuggingFace Hub push fails — 401 Unauthorized

```
huggingface_hub.utils._errors.RepositoryNotFoundError: 401
```

**Fix:** Your API token is missing or expired.
1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with **Write** access
3. Paste it into the **HF API Token** field in the Share tab

---

## Getting More Help

If your problem isn't listed here:

1. **Check the full error message** — the last line is usually the most informative
2. **Search the GitHub Issues** — [github.com/Yog-Sotho/LLM-fine-tuner/issues](https://github.com/Yog-Sotho/LLM-fine-tuner/issues)
3. **Open a new issue** — include your OS, Python version, GPU model + VRAM, and the full error traceback

---

## Next Step

→ [12 — FAQ](12_faq.md): Quick answers to common questions.
