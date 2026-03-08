# 08 — Export & Deploy

Once your model is trained, you have several ways to share it and put it to use. This guide covers all four export methods.

---

## Overview

| Method | Best for | Format |
|---|---|---|
| **ZIP Download** | Saving a backup, sharing privately | `.zip` folder |
| **HuggingFace Hub** | Sharing publicly, version control | HF model repository |
| **GGUF Export** | Running locally with Ollama or LM Studio | `.gguf` file |
| **Model Registry** | Tracking multiple models within the tool | Internal catalogue |

---

## Method 1 — ZIP Download

The simplest option. Packages your entire model (weights + tokenizer + config) into a single downloadable ZIP.

1. Go to the **📤 Share** tab
2. Under **Download Model as ZIP**, click **📦 Create ZIP**
3. Wait for the ZIP to be created (a few seconds to a minute depending on model size)
4. Click the download link that appears

**What's inside the ZIP:**
```
my_model.zip
├── adapter_config.json       ← LoRA adapter configuration
├── adapter_model.safetensors ← The trained weights
├── tokenizer.json            ← Tokenizer files
├── tokenizer_config.json
└── README.md                 ← Auto-generated model card
```

> **Note:** If you used LoRA (recommended), the ZIP contains only the small adapter (~50–200 MB), not the full base model (~3–14 GB). To use it, you still need the base model separately. For a standalone model, use the Merge Adapter feature first (see [06 — Inference](06_inference.md)).

---

## Method 2 — HuggingFace Hub

Share your model publicly (or privately) on HuggingFace — the world's largest AI model repository. Other people can download and use your model directly from there.

### Setup (one-time)

You need a free HuggingFace account and an API token:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Go to **Settings → Access Tokens → New token**
3. Create a token with **Write** permission
4. Copy the token (starts with `hf_...`)

### Uploading

1. In the **📤 Share** tab, find **Push to HuggingFace Hub**
2. Fill in:
   - **HF API Token** — paste your `hf_...` token
   - **Repository Name** — e.g. `my-username/my-cool-model`
   - **Model Path** — your trained model folder
   - **Private repository** — tick this if you don't want the model to be public
3. Click **🚀 Push to Hub**

Your model will appear at `https://huggingface.co/my-username/my-cool-model`.

### Auto-generated Model Card

The tool automatically creates a `README.md` (called a model card on HuggingFace) that describes:
- Base model used
- Training method and parameters
- Dataset information
- Usage instructions

You can edit this before uploading.

---

## Method 3 — GGUF Export

GGUF is a file format designed for running AI models efficiently on consumer hardware — even without a GPU. It's used by:

- **[Ollama](https://ollama.ai)** — run models with a single terminal command
- **[LM Studio](https://lmstudio.ai)** — a user-friendly desktop app for running models
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — the underlying engine

### Quantisation Levels

GGUF files are quantised (compressed) to reduce file size. Choose based on your target hardware:

| Preset | Quality | File size (7B model) | Good for |
|---|---|---|---|
| `q8_0` | Near-lossless (99%) | ~7 GB | High quality, plenty of VRAM/RAM |
| `q6_k` | Excellent | ~5.5 GB | Best balance — recommended |
| `q5_k_m` | Very good | ~4.7 GB | Slightly less RAM |
| `q4_k_m` | Good | ~4 GB | Minimum RAM, most compression |

> **Which to pick?** If unsure, use `q6_k`. The quality difference from `q8_0` is barely noticeable in practice, but you save 1.5 GB.

### Exporting

1. Go to the **🗜️ GGUF Export** tab
2. Fill in:
   - **Model path** — your trained model (merged adapter, not raw adapter)
   - **Quantisation** — select from the dropdown
   - **Output path** — where to save the `.gguf` file
3. Click **🗜️ Export to GGUF**

The tool will:
1. First try Unsloth (fastest, if available)
2. Fall back to llama.cpp (if Unsloth fails or isn't installed)

> **Prerequisite:** llama.cpp must be installed for the fallback to work. The installer sets this up automatically. If you installed manually, run `git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make`.

### Using the GGUF with Ollama

After exporting:

```bash
# Create a Modelfile
echo 'FROM ./my_model_q6_k.gguf' > Modelfile

# Register the model with Ollama
ollama create my-model -f Modelfile

# Run it
ollama run my-model
```

### Using the GGUF with LM Studio

1. Open LM Studio
2. Go to **My Models** → **Import**
3. Select your `.gguf` file
4. Click **Load** and start chatting

---

## Method 4 — Model Registry

The Model Registry is an internal catalogue inside the tool for keeping track of your models. Useful if you're training many versions and want to compare them.

1. In the **📤 Share** tab, find **Model Registry**
2. Click **➕ Register Model**
3. Fill in the model path — the registry will automatically read the config files and fill in the base model name
4. Add any notes you want

You can then:
- Browse all registered models
- Filter by base model or training type
- See training metadata at a glance

---

## Next Step

→ [09 — CLI Reference](09_cli_reference.md): Run the entire pipeline without the UI.
