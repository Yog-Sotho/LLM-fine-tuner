import torch
import time
import numpy as np

def benchmark_inference():
    # Setup a dummy model and input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 1024)
    ).to(device)
    x = torch.randn(128, 1024).to(device)

    # Warmup
    for _ in range(10):
        model(x)

    # Benchmark torch.no_grad()
    no_grad_times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(10):
                model(x)
        no_grad_times.append(time.perf_counter() - start)

    # Benchmark torch.inference_mode()
    inference_mode_times = []
    for _ in range(100):
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(10):
                model(x)
        inference_mode_times.append(time.perf_counter() - start)

    avg_no_grad = np.mean(no_grad_times)
    avg_inference = np.mean(inference_mode_times)

    print(f"Average time with torch.no_grad():      {avg_no_grad:.6f}s")
    print(f"Average time with torch.inference_mode(): {avg_inference:.6f}s")
    print(f"Speedup: {avg_no_grad / avg_inference:.2f}x")

if __name__ == "__main__":
    benchmark_inference()
