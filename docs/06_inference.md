# 06 — Inference

Inference means using your trained model to generate text. This guide covers the Inference tab, batch testing, and the high-performance vLLM engine.

---

## The 💬 Inference Tab

After training finishes, the adapter path is filled in automatically. You can start generating right away.

### Loading a Model

| Field | What to enter |
|---|---|
| **Base Model ID** | The same model you trained on (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`) |
| **PEFT Adapter Path** | The folder where your fine-tuned adapter was saved (filled in automatically after training) |

The model is loaded once and cached — repeated generations don't reload it, so they're fast.

> **Tip:** To switch to a different model, change the model ID and click Generate. The cache clears automatically when you load a new model.

---

## Single Generation

1. Type your prompt in the **Prompt** box:
   ```
   Explain the water cycle to a 10-year-old.
   ```
2. Adjust the generation settings (optional — defaults work well)
3. Click **✨ Generate**

The response appears in the **Output** box within a few seconds.

### Generation Settings

| Setting | Default | What it does |
|---|---|---|
| **Max New Tokens** | 150 | Maximum length of the response. Increase for longer answers. |
| **Temperature** | 0.7 | Creativity vs. accuracy. Lower (0.1–0.3) = more focused and predictable. Higher (0.8–1.0) = more creative and varied. |
| **Top-p** | 0.9 | Filters out very unlikely words. 0.9 is a good balance. Leave this alone unless you know what it does. |

**Temperature examples:**

| Temperature | Behaviour | Good for |
|---|---|---|
| 0.1 | Very predictable, almost deterministic | Facts, structured data, Q&A |
| 0.7 | Balanced — natural but consistent | Most use cases |
| 1.0 | Creative and unpredictable | Creative writing, brainstorming |

---

## Batch Testing

Batch testing sends many prompts at once and gives you all the responses as a CSV. Perfect for checking your model's consistency.

1. Create a CSV with a `prompt` column (other columns are ignored), or a `.txt` file with one prompt per line:
   ```csv
   prompt
   "What is photosynthesis?"
   "Who invented the telephone?"
   "Name three primary colours."
   ```

2. In the **Batch inference** section, upload this file
3. Click **Run Batch** — prompts are generated in batches of up to 8

The **Status** box shows how many responses were written (or what went wrong, e.g. a missing
`prompt` column), and **Download responses CSV** gives you a `prompt,response` file. To score
responses against reference answers (BLEU / ROUGE / BERTScore / LLM judge), use the
[Evaluation tab](07_evaluation.md).

---

## Merge Adapter for Deployment

By default your fine-tuned model is stored as a small "adapter" file that sits on top of the base model. This is efficient for training, but some deployment tools (like Ollama and LM Studio) need the adapter merged into the base model first.

1. In the **vLLM / Merge** section, find the **🔗 Merge Adapter** tool
2. Enter:
   - **Base Model ID** — same as used in training
   - **Adapter Path** — the folder with your fine-tuned adapter
   - **Output Path** — where to save the merged model (e.g. `./my_model_merged`)
3. Click **🔗 Merge Adapter into Base**

This creates a new folder containing a standalone model that doesn't need the base model separately. Use this merged folder for GGUF export or direct deployment.

---

## vLLM High-Throughput Inference

vLLM is a specialised inference engine that can handle many users simultaneously with much higher speed than the standard HuggingFace pipeline. It's designed for production or multi-user scenarios.

### Requirements

- NVIDIA GPU with CUDA
- vLLM installed (CUDA): `pip install "trl[vllm]"`
- A **merged** model (see above — vLLM doesn't support unmerged adapters)

### How to Use

1. In the **vLLM Generation** section:
   - Enter your merged model path
   - Select the quantisation type (leave on `none` unless you know you need gptq/awq)
2. Click **⚡ Start vLLM Engine** — this loads the model once into the engine
3. Type a prompt and click **⚡ vLLM Generate**

The engine stays loaded between requests, so each generation after the first is very fast.

> **Tip:** The vLLM engine is cached — if you call it multiple times with the same model, it reuses the loaded engine instead of reloading. This makes repeated inference much faster.

### When to Use vLLM vs Standard Inference

| Scenario | Use |
|---|---|
| Testing your model | Standard inference |
| Running in production with multiple users | vLLM |
| Generating hundreds of responses in a row | vLLM |
| Simple one-off testing | Standard inference |

---

## Remote Endpoint (OpenAI-compatible)

The **🌐 Remote endpoint** section chats with a deployed model: `python main.py serve` (see
[08 — Export & Deploy](08_export_and_deploy.md)), vLLM, llama.cpp, TGI, or a hosted
OpenAI-compatible API.

- **Endpoint URL** — e.g. `http://127.0.0.1:8000` (`/v1` is added if missing); http(s) only.
- **Model name** — leave blank to use the server's first model.
- **API key** — sent as `Authorization: Bearer …` when set.

The request is sent from the machine running this app, not from your browser.

---

## Next Step

→ [07 — Evaluation](07_evaluation.md): Measure your model's quality with automatic metrics.
