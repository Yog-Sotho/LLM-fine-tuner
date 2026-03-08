# 02 — Quick Start

This guide walks you through fine-tuning your first model in about 5 minutes using a small example dataset. No technical knowledge required.

---

## What We'll Do

1. Launch the app
2. Upload a small example dataset
3. Pick a model and a training preset
4. Start training
5. Test the result

---

## Step 1 — Launch the App

```bash
# Activate your environment first
source llm_finetuner_env/bin/activate   # Linux / macOS
# or: llm_finetuner_env\Scripts\activate  (Windows)

# Launch
python main.py
```

Your browser opens to `http://localhost:7860`. You'll see the LLM Fine-Tuner interface with a dark purple theme.

At the top you'll see your hardware summary — this tells you what model sizes you can realistically train.

---

## Step 2 — Prepare a Sample Dataset

Create a file called `my_data.csv` with this content (or download it from the repo's `examples/` folder):

```csv
instruction,output
"What is the capital of France?","The capital of France is Paris."
"Explain photosynthesis in simple terms.","Plants use sunlight, water, and air to make their own food. The sunlight gives them energy to turn water and carbon dioxide into sugar."
"Write a short poem about the ocean.","The waves roll in and roll away, / The sea sings softly every day, / Salt and spray and endless blue, / The ocean's always something new."
"What should I eat for breakfast?","A healthy breakfast includes protein (like eggs or yogurt), complex carbs (like oats or whole grain toast), and some fruit."
"How do I stay motivated?","Break big goals into small steps, celebrate small wins, and remind yourself why you started. Rest when you need to — motivation follows action."
```

Save this file somewhere easy to find, like your Desktop.

> **What are these columns?**
> - `instruction` — the question or prompt you want the model to respond to
> - `output` — the ideal answer you want the model to learn

---

## Step 3 — Upload Your Data

1. Click the **📂 Data** tab at the top
2. Under **Upload File**, click the upload box and select your `my_data.csv`
3. The tool will automatically detect the columns and show you a preview of the first 10 rows
4. You should see a status message like: `✅ Loaded 5 examples`

That's it — no column mapping needed when your columns are already called `instruction` and `output`.

---

## Step 4 — Configure Training

1. Click the **🚀 Training** tab

2. **Pick a model** — for a quick test, leave it on the auto-recommended model (likely `gpt2` or `distilgpt2` if you have limited RAM/VRAM). These are small and fast.

3. **Training Mode** — leave it on `SFT (Supervised Fine-Tuning)`. This is the standard mode for teaching a model new knowledge from question-answer pairs.

4. **Training Preset** — select `Quick (1 epoch)`. This trains fast (under 2 minutes on most machines) and is perfect for testing.

5. **PEFT Method** — leave it on `Auto`. The tool will pick the best method for your hardware automatically.

Everything else can stay at its default value.

---

## Step 5 — Start Training

Click the big **▶ Start Training** button.

You'll see the training log fill up in real time on the right side:

```
🚀 Loading model: gpt2
✅ Model loaded
📊 Dataset: 5 examples
⚙️ PEFT method: LoRA
🏃 Epoch 1/1 — step 1/3 — loss: 3.42
🏃 Epoch 1/1 — step 2/3 — loss: 2.87
🏃 Epoch 1/1 — step 3/3 — loss: 2.31
✅ Training complete! Time: 0:01:12
📁 Model saved to: /tmp/xxxxx
```

The **loss number dropping** is good — it means the model is learning. When it says `✅ Training complete`, you're done.

---

## Step 6 — Test Your Model

1. Click the **💬 Inference** tab
2. The **PEFT adapter path** box should already be filled in automatically
3. Type a prompt in the **Prompt** box — try one from your dataset:
   ```
   What is the capital of France?
   ```
4. Click **Generate ✨**

The model will respond. After training on only 5 examples with 1 epoch, don't expect perfection — but you should see the model producing relevant output. With more data and more epochs, quality improves dramatically.

---

## Step 7 — Download Your Model

1. Go to the **📤 Share** tab
2. Click the **Model ZIP** download button
3. Save the ZIP file — this contains your fine-tuned model

---

## What's Next?

| Goal | Guide |
|---|---|
| Better results | Use more data and `Balanced (3 epochs)` preset — see [04 — Training](04_training.md) |
| Format your own data | [03 — Data Preparation](03_data_preparation.md) |
| Export for Ollama / LM Studio | [08 — Export & Deploy](08_export_and_deploy.md) |
| Run without the UI | [09 — CLI Reference](09_cli_reference.md) |

---

## Tips for Better Results

- **More data = better results.** Even 100–500 high-quality examples makes a big difference.
- **Consistent formatting matters.** All your `output` answers should follow a similar style.
- **Use `Balanced (3 epochs)`** for real training runs, not just quick tests.
- **Use a bigger model** if your hardware allows — TinyLlama or Mistral-7B will produce much better outputs than gpt2.
