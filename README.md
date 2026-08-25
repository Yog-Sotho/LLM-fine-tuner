<div align="center">
  <img src="Images/logo.jpg" alt="LLM Fine-Tuner" width="800"/>

  <h1>🧠 LLM Fine-Tuner v3.2</h1>

  <p><strong>Your own custom AI — no coding, no PhD, no drama.</strong><br>
  Upload your data → click Train → get a ready-to-use model in minutes.</p>

  <a href="https://github.com/Yog-Sotho/LLM-fine-tuner/stargazers">
    <img src="https://img.shields.io/github/stars/Yog-Sotho/LLM-fine-tuner?style=for-the-badge&logo=github&color=7c3aed" alt="Stars">
  </a>
  <a href="https://github.com/Yog-Sotho/LLM-fine-tuner/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/Yog-Sotho/LLM-fine-tuner?style=for-the-badge&color=10b981" alt="License">
  </a>
  <a href="https://github.com/Yog-Sotho/LLM-fine-tuner/releases">
    <img src="https://img.shields.io/badge/version-v3.2-3b82f6?style=for-the-badge" alt="Release v3.2">
  </a>
  <a href="https://huggingface.co/spaces?sort=trending">
    <img src="https://img.shields.io/badge/🤗-Try_on_HF_Spaces-8b5cf6?style=for-the-badge" alt="HF Spaces">
  </a>
  <a href="https://github.com/sponsors/Yog-Sotho" target="_blank" rel="noopener">
    <img src="https://img.shields.io/badge/Sponsor❤️-30363D.svg?logo=githubsponsors&logoColor=EA4AAA" alt="Sponsor on GitHub">
  </a>
</div>

---

## 🤔 What is this?

Imagine you could take a smart AI assistant and teach it to be an expert in *your* specific topic — your business, your writing style, your data. That's exactly what fine-tuning is.

LLM Fine-Tuner lets you do it through a simple visual interface. No programming knowledge needed. If you can use a spreadsheet and a web browser, you can fine-tune an AI model.

**What you can build:**
- 💼 A customer support bot that knows your products inside and out
- ✍️ A writing assistant that matches your exact tone and style
- 🏥 A domain expert trained on your specialised knowledge
- 🎮 A character AI with a specific personality
- 📚 A Q&A tool trained on your own documents

---

## ✨ Why people love it

- **No coding** — Everything happens through a point-and-click interface
- **Works on regular hardware** — Even a gaming laptop with 8 GB of GPU memory is enough
- **Fast** — Train a model in minutes, not days
- **Your data stays yours** — Everything runs on your own machine
- **Export anywhere** — Use your model in Ollama, LM Studio, or share it online

---

## 🖼️ Gallery

<div align="center">

### ⚡ What fine-tuning actually looks like

<img src="Images/LLM1.png" alt="Fine-tuning visualised as precision forging" width="750"/>

*Think of fine-tuning like a master craftsman shaping raw material into a precision tool. The AI already knows a lot — you're just focusing that knowledge toward exactly what you need.*

---

### 🚀 Supercharged with Unsloth

<img src="Images/unsloth.png" alt="Unsloth acceleration" width="750"/>

*LLM Fine-Tuner uses Unsloth under the hood — a turbo engine that makes training 2–5× faster and uses dramatically less memory. Even a regular gaming GPU becomes a fine-tuning powerhouse.*

---

### 🔓 Heretic Mode

<img src="Images/heretic.png" alt="Heretic Mode" width="750"/>

*Some AI models have built-in restrictions that get in the way. Heretic Mode lets you remove those restrictions with one click. Use it responsibly — with great power comes great responsibility.*

</div>

---

## 🚀 Get Started in 2 Minutes

### The Easy Way (recommended)

```bash
# Download the project
git clone https://github.com/Yog-Sotho/LLM-fine-tuner.git
cd LLM-fine-tuner

# Run the installer — it handles everything for you
chmod +x install.sh && ./install.sh
```

The installer will ask you a few yes/no questions. When it's done, launch with:

```bash
source llm_finetuner_env/bin/activate
llm-finetune
```

Your browser opens automatically. You're ready to train.

### Want to skip all the questions?

```bash
./install.sh --yes
```

### Just want to try it quickly?

```bash
pip install -r requirements.txt
python main.py
```

> **Using Google Colab?** You can run this entirely in your browser with a free GPU. See the [full installation guide](docs/01_installation.md).

---

## 📋 How it works — step by step

### Step 1 — Prepare your data

Create a simple spreadsheet (CSV file) with two columns:

| instruction | output |
|---|---|
| What are your opening hours? | We're open Monday to Saturday, 9am to 6pm. |
| Do you offer refunds? | Yes! We offer full refunds within 30 days of purchase. |
| Where are you located? | We're at 123 Main Street, downtown. |

That's it. No special formatting, no coding. Just questions and the answers you want the model to give.

> Don't have a CSV? The tool also accepts Word docs, PDFs, Excel files, plain text files, and more.

### Step 2 — Upload and train

1. Open the app (it runs in your browser)
2. Click **📂 Data** and upload your file
3. Click **🚀 Training**, pick a preset, and hit **▶ Start Training**
4. Watch the progress bar — training takes minutes to an hour depending on your hardware

### Step 3 — Test and export

1. Go to **💬 Inference** and type a question
2. Your model responds using everything it learned from your data
3. When happy, go to **📤 Share** to download or publish your model

---

## 📂 What data formats does it accept?

| Format | How to create it |
|---|---|
| CSV | Save any spreadsheet as CSV from Excel or Google Sheets |
| Excel (.xlsx) | Drag in your Excel file directly |
| JSON / JSONL | Export from your database or app |
| Plain text (.txt) | One example per line |
| PDF | Any PDF document — text is extracted automatically |
| ZIP | Put multiple files in a ZIP and upload them all at once |

---

## 🎛️ Training Options (Plain English)

You don't need to understand all of these to get started — the defaults work great. But here's what they mean if you're curious:

| Option | What it does | Beginner recommendation |
|---|---|---|
| **Training Preset** | Quick = fast test, Balanced = good results, Accurate = best quality | Start with **Balanced** |
| **Base Model** | The starting AI brain you're teaching | Leave on **Auto** — the tool picks the right one for your hardware |
| **PEFT Method** | How the training is done internally | Leave on **Auto** |
| **Unsloth** | Turbo mode — makes training much faster | Always turn **ON** if available |
| **Heretic Mode** | Removes built-in restrictions from the model | Optional — use responsibly |

---

## 📊 Supported Training Modes

| Mode | What it does | When to use it |
|---|---|---|
| **Standard Training (SFT)** | Teaches the model using your question-answer examples | Starting point for almost everyone |
| **Preference Training (DPO)** | Teaches the model which answers are better vs worse | After standard training, to improve quality |
| **Reward + PPO** | Advanced alignment with a scoring system | When you want the highest quality alignment |
| **ORPO** | Modern single-step alignment | Faster alternative to full Reward + PPO |

---

## 📤 Export Options

Once trained, your model can be:

- **Downloaded as a ZIP** — keep a backup on your computer
- **Published to HuggingFace Hub** — share it with the world (or keep it private)
- **Exported as GGUF** — run it offline with [Ollama](https://ollama.ai) or [LM Studio](https://lmstudio.ai) on any computer, even without internet
- **Served with vLLM** — high-speed serving for multiple users at once

---

## 🗺️ What's been built, what's coming

**Already available in v3.2:**
- ✅ Visual interface — no coding needed
- ✅ All major training modes (SFT, DPO, RLHF, ORPO)
- ✅ GGUF export for Ollama & LM Studio
- ✅ Heretic Mode
- ✅ Command-line mode for power users
- ✅ Batch evaluation tools
- ✅ Data augmentation

**Coming soon:**
- 🔲 Synthetic data generator — create training data with AI
- 🔲 Docker image — one-command setup with no dependencies
- 🔲 Multi-GPU training
- 🔲 Vision + language models

---

## 📖 Documentation

New to fine-tuning? The docs are written for non-technical users:

| Guide | What you'll learn |
|---|---|
| [Installation](docs/01_installation.md) | 4 ways to install, including Google Colab |
| [Quick Start](docs/02_quick_start.md) | Train your first model in 5 minutes |
| [Preparing Your Data](docs/03_data_preparation.md) | How to format your CSV, fix column names |
| [Training Guide](docs/04_training.md) | All settings explained in plain English |
| [Exporting Your Model](docs/08_export_and_deploy.md) | Download, publish, or deploy |
| [Troubleshooting](docs/11_troubleshooting.md) | Fix the most common errors |
| [FAQ](docs/12_faq.md) | Quick answers to common questions |

---

## 🤝 Contributing

Contributions welcome! Fork the repo, make your changes, and open a pull request with a clear description of what you did and why.

Found a bug? [Open an issue](https://github.com/Yog-Sotho/LLM-fine-tuner/issues) with your OS, GPU model, and the full error message.

---

## 📜 License

GPL-3.0 — free to use, modify, and share. Attribution appreciated ❤️

---

<div align="center">

**Made with ❤️ for the open-source community**

*If this tool helped you build something cool, a ⭐ on GitHub means the world.*
Yog-Sotho
</div>
