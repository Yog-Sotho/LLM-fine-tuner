# 08 — Export & Deploy

Once your model is trained, you have several ways to share it and put it to use. This guide covers all four export methods.

---

## Overview

| Method | Best for | Format |
|---|---|---|
| **ZIP Download** | Saving a backup, sharing privately | `.zip` folder |
| **HuggingFace Hub** | Sharing publicly, version control | HF model repository |
| **GGUF Export** | Running locally with Ollama or LM Studio | `.gguf` file |
| **Quantized safetensors** | Serving on NVIDIA GPUs with vLLM | FP8 / W4A16 model folder |
| **Serve** | An OpenAI-compatible API for apps and clients | `/v1/chat/completions` |
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
3. Click **🚀 Push to Hub** — it uploads the model you trained in this session

The repository is created if it doesn't exist yet, with your account's default visibility
(public unless you changed that default — you can make it private in the repo settings).
Your model will appear at `https://huggingface.co/my-username/my-cool-model`.

### Auto-generated Model Card

Every training run (UI or CLI, any method) writes a `README.md` next to the model — called a
model card on Hugging Face — built from the run's `run_config.yaml`:
- **Metadata the Hub reads:** `base_model` (so the Hub links your model to its base),
  `library_name` (`peft` for LoRA adapters), `pipeline_tag`, tags, and `datasets` when you
  trained on a Hub dataset
- Training method, all settings and the seed
- Dataset size and SHA-256 fingerprint, library versions
- How to load the model

When you push, the base model is checked on the Hub: its canonical name is used (e.g.
`gpt2` → `openai-community/gpt2`) and its license is copied into the card. A base model
that is a local folder or isn't on the Hub is left out, because the Hub rejects cards with
an invalid `base_model`.

You can edit the card before uploading.

---

## Method 3 — GGUF Export

GGUF is a file format designed for running AI models efficiently on consumer hardware — even without a GPU. It's used by:

- **[Ollama](https://ollama.ai)** — run models with a single terminal command
- **[LM Studio](https://lmstudio.ai)** — a user-friendly desktop app for running models
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** — the underlying engine

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

If the path is a LoRA adapter (the usual training output), it is first merged into its base model — named in `adapter_config.json` — in a temporary folder, because llama.cpp converts full models only. Prefix/Prompt-tuning adapters cannot be merged and are rejected.

> **Prerequisite:** llama.cpp must be installed for the fallback to work. The installer can set it up. If you installed manually:
> ```bash
> git clone --depth 1 https://github.com/ggml-org/llama.cpp
> cmake -S llama.cpp -B llama.cpp/build
> cmake --build llama.cpp/build --target llama-quantize
> pip install "sentencepiece>=0.1.98,<0.3.0"   # used by the converter for some tokenizers
> export PATH="$PATH:$PWD/llama.cpp:$PWD/llama.cpp/build/bin"
> ```
> Without `llama-quantize` the export stops at an FP16 GGUF.

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

## Method 4 — Quantized safetensors for vLLM

For serving on NVIDIA GPUs with [vLLM](https://docs.vllm.ai), export a compressed model with
[llm-compressor](https://github.com/vllm-project/llm-compressor) (`pip install
"llm-fine-tuner[compress]"` — it pins recent torch/Transformers, so a separate environment can
be simpler):

| Format | What it does | Needs |
|---|---|---|
| **FP8** | 8-bit float weights + dynamic 8-bit activations — about half the size, near-lossless; native speed on Hopper/Ada GPUs | nothing (data-free) |
| **W4A16** | 4-bit GPTQ weights, 16-bit activations — about ¼ of the size | calibration text: your training data (≈128 examples); layer widths divisible by 128 (true for real models) |

In the **📦 Export** tab choose the format and click **🗜️ Export quantized** (W4A16 uses the data
loaded in 📂 Data). From the CLI:

```bash
python main.py export --model ./runs/my-run --format fp8 --output ./runs/my-run-fp8
python main.py export --model ./runs/my-run --format w4a16 --calibration-data train.jsonl \
    --output ./runs/my-run-w4a16
```

A LoRA adapter is merged into its base model first. Both formats run on CPU too (slower), and
the model card is tagged with the format.

---

## Method 5 — Serve behind an OpenAI-compatible API

```bash
python main.py serve --model ./gguf/model_q4_k_m.gguf --port 8000   # llama.cpp (CPU or GPU)
python main.py serve --model ./runs/my-run-fp8 --port 8000          # vLLM (CUDA)
python main.py serve --model ./runs/my-run --port 8000 --name my-bot # LoRA adapter on its base (vLLM)
```

- `.gguf` files are served by llama.cpp's `llama-server` (build it as in the GGUF prerequisite,
  target `llama-server`); anything else by `vllm serve` (`pip install "trl[vllm]"`, CUDA).
- Clients use `http://<host>:8000/v1` — `/v1/chat/completions`, `/v1/models` — with the model
  name from `--name` (default `model`).
- `--host` defaults to `127.0.0.1`; use `0.0.0.0` to accept network connections, and then set
  `LFT_SERVE_API_KEY` so clients must send `Authorization: Bearer <key>`. The key is passed to the
  server through its environment, never on the command line.

Test it from the **💬 Inference** tab (**🌐 Remote endpoint**) or with any OpenAI client.

---

## Method 6 — Model Registry

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
