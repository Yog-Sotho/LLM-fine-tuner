<div align="center">
  <img src="logo.jpg" alt="LLM Fine-Tuner Logo" width="200"/>
  <h1>🧠 LLM Fine-Tuner v2.4</h1>
  <p><strong>The easiest way to fine-tune LLMs — no coding required.</strong><br>
  Upload your data → click Train → get a ready-to-use model in minutes.<br>
  Now powered by <strong>Unsloth</strong> (2-5× faster, 60-80% less VRAM) + Smart Chat Templates + GGUF Export + DPO Alignment + <strong>Heretic Mode</strong>.</p>

  <a href="https://github.com/Yog-Sotho/LLM-fine-tuner/stargazers">
    <img src="https://img.shields.io/github/stars/Yog-Sotho/LLM-fine-tuner?style=for-the-badge&logo=github&color=7c3aed" alt="Stars">
  </a>
  <a href="https://github.com/Yog-Sotho/LLM-fine-tuner/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Yog-Sotho/LLM-fine-tuner?style=for-the-badge&color=10b981" alt="License">
  </a>
  <a href="https://github.com/Yog-Sotho/LLM-fine-tuner/releases">
    <img src="https://img.shields.io/github/v/release/Yog-Sotho/LLM-fine-tuner?style=for-the-badge&color=3b82f6" alt="Release">
  </a>
  <a href="https://huggingface.co/spaces?sort=trending">
    <img src="https://img.shields.io/badge/🤗-Try_on_HF_Spaces-8b5cf6?style=for-the-badge" alt="HF Spaces">
  </a>
</div>

---

## ✨ Why LLM Fine-Tuner?

- **Zero coding** — Drag & drop CSV, JSONL, TXT, Excel, PDF, ZIP
- **Smart defaults** — Auto-detects your hardware and recommends the best model
- **Unsloth powered** — Train 7B–14B models on a single RTX 4090/5090
- **Perfect chat formatting** — Automatic `apply_chat_template` for Llama-3, Mistral, Qwen, Gemma-2, Phi, etc.
- **Multiple PEFT methods** — LoRA, Prefix Tuning, Prompt Tuning, Adapters + Full Fine-Tuning
- **Live loss chart + one-click stop** — Real-time monitoring
- **Export ready** — ZIP download, HF Hub push, GGUF coming soon
- **DPO Alignment** — Direct Preference Optimization for professional-grade helpfulness and alignment
- **Heretic Mode** — One-click automatic uncensoring to unlock the full potential of your model (use responsibly)

Perfect for creators, small teams, researchers, and anyone who wants their own custom AI without the headache.

### 🔍 Features Deep Dive

- **Supported formats:** CSV, JSONL, JSON, TXT, Excel, PDF, ZIP (with path-traversal protection)
- **Hardware auto-detect + model recommendation**
- **Live training with stop button and Plotly loss curve**
- **One-click HF Hub push with beautiful auto-generated model card**
- **Batch inference** (CSV or TXT prompts → downloadable results)
- **PEFT methods:** LoRA (with Unsloth), Prefix Tuning, Prompt Tuning, Adapters, Full Fine-Tuning
- **GGUF Export:** One-click quantized models (q8_0, q6_k, q5_k_m, q4_k_m) ready for Ollama, LM Studio, llama.cpp
- **DPO Alignment:** Learn from chosen vs rejected responses for better alignment
- **Heretic Mode:** Automatic restriction removal after training (use responsibly)

### 🗺️ Roadmap (v2.4 → v3.0)

- [x] GGUF Export (one click)
- [x] DPO Alignment tab
- [x] Heretic Mode integration
- [ ] Synthetic data generator
- [ ] Docker + CLI support
- [ ] Multi-GPU via Accelerate

## 🤝 Contributing

Pull requests welcome!

Fork → create feature branch → open PR with clear description.

## 📜 License

GPL-3.0 — feel free to use, modify, and share. Attribution appreciated ❤️

---

**Made with ❤️ for the open-source community**

Star the repo if it helps you build something cool!

## 🚀 Quick Start (2 minutes)

```bash
git clone https://github.com/Yog-Sotho/LLM-fine-tuner.git
cd LLM-fine-tuner

# Install dependencies
pip install -r requirements.txt

# (Optional but highly recommended) Unsloth + Heretic + DPO
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps
pip install heretic-llm trl

python LLM_Fine_Tuner_v2.4.py
