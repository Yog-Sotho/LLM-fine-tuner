# 🧠 LLM Fine-Tuner v3.2 — Documentation

Welcome! This documentation covers everything you need to go from zero to a fine-tuned, deployed AI model — no coding required.

---

## What is LLM Fine-Tuner?

LLM Fine-Tuner is a graphical tool that lets you take a pre-trained AI language model and teach it new skills using your own data. Think of it like taking a general-purpose assistant and training it to become an expert in your specific domain — whether that's customer support, writing in a particular style, answering medical questions, or anything else.

You upload your data, click a button, and get back a custom model ready to use.

---

## Who is this for?

- **Non-technical users** — if you can use a spreadsheet, you can use this tool
- **Researchers** — full control over every training parameter
- **Small teams** — no GPU cluster needed; runs on a single gaming GPU
- **Developers** — full CLI for automation and scripting

---

## Documentation Map

| Guide | What you'll learn |
|---|---|
| [01 — Installation](01_installation.md) | How to install on Windows, Mac, Linux, or Google Colab |
| [02 — Quick Start](02_quick_start.md) | Train your first model in 5 minutes |
| [03 — Data Preparation](03_data_preparation.md) | How to format your data, upload it, and fix column issues |
| [04 — Training](04_training.md) | All training modes, presets, and settings explained |
| [05 — RLHF Pipeline](05_rlhf_pipeline.md) | Reward models, PPO, and ORPO alignment |
| [06 — Inference](06_inference.md) | Testing your model, batch runs, and vLLM serving |
| [07 — Evaluation](07_evaluation.md) | Measuring quality with BLEU, ROUGE, BERTScore |
| [08 — Export & Deploy](08_export_and_deploy.md) | Download, push to HuggingFace Hub, export to GGUF |
| [09 — CLI Reference](09_cli_reference.md) | Full command-line interface with examples |
| [10 — Advanced Usage](10_advanced.md) | Heretic Mode, Flash Attention, QLoRA, hardware tips |
| [11 — Troubleshooting](11_troubleshooting.md) | Common errors and how to fix them |
| [12 — FAQ](12_faq.md) | Frequently asked questions |

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 or 3.12 |
| RAM | 8 GB | 16 GB+ |
| GPU VRAM | None (CPU only) | 8 GB+ (NVIDIA) |
| Disk space | 10 GB | 50 GB+ |
| OS | Windows 10, macOS 12, Ubuntu 20.04 | Any modern OS |

> **No GPU?** You can still fine-tune small models (gpt2, distilgpt2) on CPU. It will be slow but it works.

---

## Quick Links

- [GitHub Repository](https://github.com/Yog-Sotho/LLM-fine-tuner)
- [HuggingFace Hub](https://huggingface.co)
- [Report a Bug](https://github.com/Yog-Sotho/LLM-fine-tuner/issues)
