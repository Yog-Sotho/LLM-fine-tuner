import torch
import time

def benchmark_inference_mode_view():
    # Operations that involve views
    device = "cpu"
    x = torch.randn(1024, 1024, device=device)

    num_iterations = 100000

    print(f"Running {num_iterations} view operations...")

    # Benchmark torch.no_grad()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            y = x.view(512, 2048)
            z = y + 1
    no_grad_time = time.time() - start_time
    print(f"torch.no_grad(): {no_grad_time:.4f}s")

    # Benchmark torch.inference_mode()
    start_time = time.time()
    with torch.inference_mode():
        for _ in range(num_iterations):
            y = x.view(512, 2048)
            z = y + 1
    inference_mode_time = time.time() - start_time
    print(f"torch.inference_mode(): {inference_mode_time:.4f}s")

    speedup = (no_grad_time / inference_mode_time - 1) * 100
    print(f"Speedup: {speedup:.2f}%")

if __name__ == "__main__":
    benchmark_inference_mode_view()
