
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

# Set environment variable to avoid downloading models if possible,
# but here we probably need to download.
os.environ["TRANSFORMERS_OFFLINE"] = "0"

def benchmark_model():
    model_id = "sshleifer/tiny-gpt2" # Much smaller for faster test
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    prompt = "The quick brown fox jumps over the lazy dog"
    inputs = tokenizer(prompt, return_tensors="pt")

    iters = 50

    # Warmup
    for _ in range(5):
        _ = model.generate(**inputs, max_new_tokens=20)

    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            _ = model.generate(**inputs, max_new_tokens=20)
    no_grad_time = time.time() - start

    start = time.time()
    with torch.inference_mode():
        for _ in range(iters):
            _ = model.generate(**inputs, max_new_tokens=20)
    inf_mode_time = time.time() - start

    print(f"Tiny-GPT2 no_grad: {no_grad_time:.4f}s")
    print(f"Tiny-GPT2 inference_mode: {inf_mode_time:.4f}s")
    if no_grad_time > 0:
        print(f"Improvement: {(no_grad_time - inf_mode_time) / no_grad_time * 100:.2f}%")

if __name__ == "__main__":
    benchmark_model()
