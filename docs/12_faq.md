# 12 — FAQ

Quick answers to the most frequently asked questions.

---

## General

**Q: Do I need a GPU?**

A: No. You can fine-tune small models (gpt2, distilgpt2) on CPU. It will take longer — expect 10–30 minutes instead of 1–2 minutes for a small dataset — but it works. For 7B+ models, a GPU is practically required.

---

**Q: How much does it cost to run?**

A: Nothing, if you run it on your own hardware. The software is free and open source. If you use Google Colab, the free tier gives you a T4 GPU which is enough for most small models.

---

**Q: Will this work on a Mac?**

A: Yes. Macs with Apple Silicon (M1/M2/M3/M4) can run fine-tuning using the MPS (Metal Performance Shaders) backend. Performance is between CPU and a dedicated NVIDIA GPU. QLoRA Enhanced and Flash Attention 2 are not supported on MPS, so use standard LoRA.

---

**Q: Can I fine-tune any model from HuggingFace?**

A: Most decoder-only models (GPT-style) work out of the box. This covers the vast majority of popular models: Llama, Mistral, Qwen, Gemma, Phi, TinyLlama, GPT-2, OPT, Falcon, and more. Encoder-only models (BERT, RoBERTa) are not currently supported.

---

**Q: How do I know my model is actually learning?**

A: Watch the training loss in the log. It should decrease over time — from maybe 3.0 down toward 1.0 or below. A loss that doesn't decrease means something is wrong (usually a data or learning rate issue). A loss that drops to zero immediately means the model is memorising, not learning (too little data or too many epochs).

---

**Q: My model sounds good during training but bad at inference. Why?**

A: The most common causes are:
- Wrong temperature setting at inference (try 0.7)
- Prompt format mismatch (training used chat templates but inference doesn't)
- Too few epochs — the model hasn't learned enough
- Overfitting — trained for too many epochs on too little data

---

**Q: How long does training take?**

A: It depends heavily on model size, dataset size, and hardware. As a rough guide:

| Model | Dataset | GPU | Time |
|---|---|---|---|
| gpt2 (124M) | 100 examples | CPU | ~5 min |
| TinyLlama (1.1B) | 500 examples | RTX 3060 (12 GB) | ~15 min |
| Mistral-7B | 1,000 examples | RTX 4090 (24 GB) | ~45 min |
| Mistral-7B | 1,000 examples | RTX 3060 (QLoRA) | ~2 hours |

---

## Data

**Q: How many examples do I need?**

A: For a basic proof of concept, 20–50 is enough. For noticeable quality improvements, aim for 200–500. For strong domain specialisation, 1,000–5,000 is ideal. Quality matters more than quantity.

---

**Q: My CSV has different column names. What do I do?**

A: Use the Column Mapping dropdowns in the Data tab to map your columns. For example, if your columns are `question` and `answer`, map them to `instruction` and `output` respectively. See [03 — Data Preparation](03_data_preparation.md).

---

**Q: Can I use data in languages other than English?**

A: Yes, as long as the base model you're using supports that language. Multilingual models like mBERT or Qwen support many languages natively. For best results, use a base model that was pre-trained on text in your target language.

---

**Q: Can I mix CSV and JSONL data?**

A: Not directly in a single upload. However, you can put multiple files into a ZIP and upload the ZIP — the tool will combine them automatically.

---

## Training

**Q: What is LoRA and why should I use it?**

A: LoRA (Low-Rank Adaptation) is a technique that fine-tunes only a small set of added parameters instead of all the model's weights. This reduces VRAM usage by 90%+ and speeds up training significantly, with only a tiny drop in quality. It's the recommended method for almost everyone.

---

**Q: What's the difference between LoRA and QLoRA Enhanced?**

A: QLoRA Enhanced adds 4-bit quantisation on top of LoRA. It uses even less VRAM — allowing you to fine-tune a 7B model on 8 GB — but training is slightly slower. Use LoRA if you have enough VRAM; use QLoRA Enhanced if you're tight on VRAM.

---

**Q: Should I use Unsloth?**

A: Yes, always — if you have an NVIDIA GPU and you're using LoRA. It's free, it works, and it makes training 2–5× faster with 60–80% less VRAM. There's no meaningful downside.

---

**Q: Training stopped early. Is my model ruined?**

A: No. When training stops early (either manually or via early stopping), the model is saved at that checkpoint. You can use it as-is or resume training from the checkpoint by enabling **Resume from checkpoint**.

---

**Q: What does "overfitting" mean and how do I avoid it?**

A: Overfitting is when the model memorises your training examples instead of learning general patterns. Signs: training loss is very low but the model gives wrong or weird answers to new questions.

Fixes:
- Use more training data
- Reduce the number of epochs
- Enable early stopping (patience = 3)
- Enable gradient checkpointing (adds some regularisation)

---

## Export & Deployment

**Q: What's the difference between an adapter and a full model?**

A: When you train with LoRA, only a small adapter is saved (typically 50–200 MB), not the full model (3–14 GB). To use it, you need both the adapter and the original base model. If you want a single standalone file, use **Merge Adapter** first and then export the merged result.

---

**Q: What is GGUF and do I need it?**

A: GGUF is a file format for running models efficiently on consumer hardware. You need it if you want to use your model with Ollama, LM Studio, or llama.cpp. If you're using the model through the LLM Fine-Tuner interface or Python code, you don't need GGUF.

---

**Q: Can I sell a model I trained with this tool?**

A: That depends on the licence of the base model you used. Many popular models (Llama, Mistral) have licences that allow commercial use if you agree to their terms. Always check the base model's licence on HuggingFace before selling or distributing a derived model.

---

**Q: Can I make the HuggingFace repo private?**

A: Yes. Tick the **Private repository** checkbox in the Share tab before pushing. Private repos require a HuggingFace Pro account for more than a small number of private models.

---

## Heretic Mode

**Q: Is Heretic Mode legal?**

A: Modifying a model you've fine-tuned is generally legal in most jurisdictions, especially for research and private use. However, if you publish or deploy the model publicly, you're responsible for any content it produces. Check your local laws and the base model's licence terms.

---

**Q: Does Heretic Mode affect training quality?**

A: No. It's applied at the end of the training process and doesn't change the training itself. Think of it as a post-processing step.

---

## Still Have Questions?

Open an issue on [GitHub](https://github.com/Yog-Sotho/LLM-fine-tuner/issues) and the community will help.
