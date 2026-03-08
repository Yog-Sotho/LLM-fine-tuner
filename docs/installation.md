# 01 — Installation

This guide walks you through installing LLM Fine-Tuner on your computer. There are four options depending on your setup.

---

## Option A — Automatic Install (Recommended for Linux / macOS)

This is the easiest method. One command does everything: creates a virtual environment, detects your GPU, installs the right version of PyTorch, and sets up a launcher shortcut.

```bash
# 1. Download the project
git clone https://github.com/Yog-Sotho/LLM-fine-tuner.git
cd LLM-fine-tuner

# 2. Run the installer
chmod +x install.sh
./install.sh
```

The installer will ask a few yes/no questions about optional components. If you want to skip all prompts and accept everything automatically:

```bash
./install.sh --yes
# or
AUTO_INSTALL=true ./install.sh
```

**What it installs:**
- Python virtual environment (isolated, won't affect your system)
- PyTorch (GPU version if NVIDIA detected, CPU version otherwise)
- All core dependencies (transformers, gradio, peft, datasets, trl, accelerate…)
- Flash Attention 2 (if your GPU supports it — big speed boost)
- Optional: Unsloth, vLLM, evaluation tools, data tools

After installation, launch the app with:

```bash
source llm_finetuner_env/bin/activate
llm-finetune
```

---

## Option B — Manual Install (Windows or any OS)

Use this if the automatic installer doesn't work on your system.

### Step 1 — Install Python

Download Python 3.11 from [python.org](https://www.python.org/downloads/). During installation on Windows, **check "Add Python to PATH"**.

Verify it worked:
```bash
python --version
# Should print: Python 3.11.x
```

### Step 2 — Download the project

```bash
git clone https://github.com/Yog-Sotho/LLM-fine-tuner.git
cd LLM-fine-tuner
```

If you don't have `git`, download the ZIP from GitHub (green "Code" button → "Download ZIP") and unzip it.

### Step 3 — Create a virtual environment

A virtual environment keeps the project's packages separate from the rest of your computer.

```bash
# Create the environment
python -m venv llm_finetuner_env

# Activate it (Linux / macOS)
source llm_finetuner_env/bin/activate

# Activate it (Windows)
llm_finetuner_env\Scripts\activate
```

You'll see `(llm_finetuner_env)` at the start of your terminal prompt. That means it's active.

### Step 4 — Install PyTorch

**If you have an NVIDIA GPU (check with `nvidia-smi` in terminal):**

```bash
# CUDA 12.x (most modern NVIDIA cards)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# CUDA 11.x (older cards like RTX 2000 series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**If you have no GPU or an AMD/Apple GPU:**

```bash
pip install torch torchvision torchaudio
```

### Step 5 — Install all other dependencies

```bash
pip install -r requirements.txt
```

### Step 6 — Install Unsloth (highly recommended)

Unsloth makes training 2–5× faster and uses 60–80% less VRAM. It's optional but strongly recommended if you have an NVIDIA GPU.

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps
```

### Step 7 — Launch

```bash
python main.py
```

Your browser should open automatically at `http://localhost:7860`. If it doesn't, open that address manually.

---

## Option C — Google Colab (No GPU required on your machine)

Google Colab gives you free GPU access in your browser. Perfect if your computer is too slow.

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook
3. In the first cell, paste and run:

```python
# Clone the repo
!git clone https://github.com/Yog-Sotho/LLM-fine-tuner.git
%cd LLM-fine-tuner

# Install dependencies
!pip install -r requirements.txt -q
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" --no-deps -q

# Launch (share=True gives you a public URL)
import subprocess
proc = subprocess.Popen(["python", "main.py", "--share"])
```

4. A public URL will appear (e.g. `https://xxxxx.gradio.live`) — open it in your browser.

> **Tip:** In Colab, go to **Runtime → Change runtime type → T4 GPU** before running for much faster training.

---

## Option D — Docker

```bash
# Build the image
docker build -t llm-fine-tuner .

# Run it
docker run -p 7860:7860 --gpus all llm-fine-tuner
```

Then open `http://localhost:7860` in your browser.

> Note: `--gpus all` requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). Remove it for CPU-only mode.

---

## Verifying the Installation

After launching, the top of the UI shows a hardware summary like:

```
🖥️ GPU: NVIDIA RTX 4090 — 24.0 GB VRAM
💾 RAM: 64.0 GB
🔥 PyTorch: 2.5.1+cu126
✅ Unsloth: available
✅ Flash Attention 2: available
```

If you see your GPU listed, everything is working correctly.

---

## Updating

To get the latest version:

```bash
cd LLM-fine-tuner
git pull
pip install -r requirements.txt --upgrade
```

---

## Next Step

→ [02 — Quick Start](02_quick_start.md): Train your first model in 5 minutes.
