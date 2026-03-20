# 13 — Docker

Docker packages the entire LLM Fine-Tuner — Python, PyTorch, llama.cpp, all dependencies — into a single container. You don't install anything on your machine except Docker itself.

---

## Who Should Use Docker?

- You want to avoid installing Python packages globally
- You're running on a server or remote machine
- You want to share an identical environment with your team
- The regular installer gave you dependency errors

---

## Prerequisites

### Everyone

Install Docker Engine: [docs.docker.com/get-docker](https://docs.docker.com/get-docker/)

Verify it works:
```bash
docker --version
# Docker version 24.x.x
```

### GPU Users (additional step)

Install the NVIDIA Container Toolkit so Docker can access your GPU:

```bash
# Ubuntu / Debian
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU access works:
```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
# Should print your GPU name and driver version
```

---

## Option A — Docker Compose (Recommended)

Docker Compose manages everything — building, volumes, GPU access, environment variables — in one command.

### GPU

```bash
# Clone the repo
git clone https://github.com/Yog-Sotho/LLM-fine-tuner.git
cd LLM-fine-tuner

# Build and start (first run downloads ~8 GB of layers — be patient)
docker compose up llm-fine-tuner-gpu
```

Open your browser at **http://localhost:7860** — the UI is ready.

### CPU (no NVIDIA GPU)

```bash
docker compose up llm-fine-tuner-cpu
```

Same URL. Training will be slow but fully functional for small models.

### With your HuggingFace token

```bash
HF_TOKEN=hf_your_token_here docker compose up llm-fine-tuner-gpu
```

Or create a `.env` file in the project root:
```bash
# .env
HF_TOKEN=hf_your_token_here
```

Then just run `docker compose up llm-fine-tuner-gpu` — Compose picks up `.env` automatically.

> **Why do I need a token?** Gated models like Llama-3 and Mistral require you to accept their licence on HuggingFace first, then authenticate with a token to download them.

### Stop the container

```bash
# Ctrl+C to stop, or from another terminal:
docker compose down
```

---

## Option B — Plain Docker (without Compose)

### Build

```bash
# GPU image
docker build -t llm-fine-tuner:gpu .

# CPU image
docker build -f Dockerfile.cpu -t llm-fine-tuner:cpu .
```

### Run the UI

```bash
# GPU
docker run --gpus all \
    -p 7860:7860 \
    -v $(pwd)/cache:/app/cache/huggingface \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/outputs:/app/outputs \
    -e HF_TOKEN=hf_your_token \
    llm-fine-tuner:gpu

# CPU
docker run \
    -p 7860:7860 \
    -v $(pwd)/cache:/app/cache/huggingface \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/outputs:/app/outputs \
    llm-fine-tuner:cpu
```

---

## Using the CLI Inside Docker

Any argument after the image name goes straight to `main.py` as a CLI command.

### Train a model

```bash
# First, put your dataset in the ./data/ folder on your HOST machine
cp my_training_data.csv ./data/

# Then run training — it reads from /app/data/ inside the container
docker compose run --rm llm-fine-tuner-gpu \
    train \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data /app/data/my_training_data.csv \
    --output /app/models/my_model \
    --epochs 3
```

When training finishes, your model appears in `./models/my_model/` on your host machine.

### Full pipeline example

```bash
# Step 1 — SFT
docker compose run --rm llm-fine-tuner-gpu \
    train --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data /app/data/sft.csv --output /app/models/sft --epochs 3

# Step 2 — Reward model
docker compose run --rm llm-fine-tuner-gpu \
    reward --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --data /app/data/reward.csv --output /app/models/reward

# Step 3 — Evaluate
docker compose run --rm llm-fine-tuner-gpu \
    evaluate --model /app/models/sft \
    --data /app/data/eval.csv --bertscore
```

### Show help

```bash
docker compose run --rm llm-fine-tuner-gpu --help
docker compose run --rm llm-fine-tuner-gpu train --help
```

---

## Volumes Explained

The container uses four persistent volumes. Everything inside them survives container restarts and rebuilds.

| Host path | Container path | What goes here |
|---|---|---|
| `./cache/` | `/app/cache/huggingface` | Downloaded model weights (saves re-downloading) |
| `./data/` | `/app/data` | Your training datasets |
| `./models/` | `/app/models` | Trained model outputs |
| `./outputs/` | `/app/outputs` | Evaluation CSVs, GGUF exports, ZIPs |

> **First run tip:** The HuggingFace cache can grow large (10–50 GB for 7B models). Mount it to a drive with plenty of space.

---

## Environment Variables

Set these with `-e` (plain Docker) or in a `.env` file (Compose).

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | HuggingFace API token — needed for Hub push and gated models |
| `SHARE` | `false` | Set to `true` for a public Gradio URL (useful on remote servers) |
| `EXTRA_ARGS` | *(empty)* | Extra args appended to the Gradio launch command |
| `HF_HOME` | `/app/cache/huggingface` | HuggingFace cache location inside the container |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Faster HuggingFace downloads (recommended ON) |
| `TOKENIZERS_PARALLELISM` | `false` | Suppresses tokeniser warning in Docker |

---

## Getting a Public URL (Remote Server)

If you're running on a remote machine and want to access the UI from your local browser:

```bash
SHARE=true docker compose up llm-fine-tuner-gpu
```

A public URL like `https://abc123.gradio.live` will be printed. Open it anywhere.

---

## Rebuilding After Code Changes

```bash
# Rebuild and restart
docker compose up --build llm-fine-tuner-gpu

# Or force a clean rebuild (no layer cache)
docker compose build --no-cache llm-fine-tuner-gpu
docker compose up llm-fine-tuner-gpu
```

---

## Troubleshooting

### "docker: Error response from daemon: could not select device driver"

The NVIDIA Container Toolkit is not installed or Docker wasn't restarted after installing it.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Build fails at `flash-attn`

Flash Attention requires a CUDA compiler matching your driver. If the build fails:
- This is non-fatal — the entrypoint will warn you and continue
- Training still works, just without Flash Attention's speed boost

### "Port 7860 is already in use"

Something else is running on port 7860. Either stop it or change the port:
```bash
# Use port 8080 instead
docker run --gpus all -p 8080:7860 llm-fine-tuner:gpu
# Open: http://localhost:8080
```

### Container starts but UI never loads

The first start can take 60–90 seconds while PyTorch initialises. Wait for this line in the logs:
```
Running on local URL:  http://0.0.0.0:7860
```

### Models aren't being saved

Check that your `./models/` directory is writable and the volume is mounted:
```bash
docker inspect llm-fine-tuner-gpu | grep -A 5 Mounts
```

---

## Next Step

→ [09 — CLI Reference](09_cli_reference.md): Full list of CLI commands for headless use inside Docker.
