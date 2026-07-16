
import torch
import time
import numpy as np

def benchmark():
    # Simple operation that might benefit from inference_mode (views etc)
    size = 10000
    x = torch.randn(size, size)

    # Warmup
    for _ in range(10):
        y = x.view(-1)[:size] * 2

    iters = 1000

    # no_grad
    start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            y = x.view(-1)[:size] * 2
    no_grad_time = time.time() - start

    # inference_mode
    start = time.time()
    with torch.inference_mode():
        for _ in range(iters):
            y = x.view(-1)[:size] * 2
    inference_mode_time = time.time() - start

    print(f"no_grad time: {no_grad_time:.4f}s")
    print(f"inference_mode time: {inference_mode_time:.4f}s")
    print(f"Improvement: {(no_grad_time - inference_mode_time) / no_grad_time * 100:.2f}%")

if __name__ == "__main__":
    benchmark()
