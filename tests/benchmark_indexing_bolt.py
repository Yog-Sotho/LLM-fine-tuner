
import time
from datasets import Dataset
import pandas as pd
import numpy as np

def benchmark():
    print("Creating a large dataset (1,000,000 rows)...")
    data = {
        "prompt": ["This is a prompt " + str(i) for i in range(1000000)],
        "chosen": ["This is chosen " + str(i) for i in range(1000000)],
        "rejected": ["This is rejected " + str(i) for i in range(1000000)],
    }
    ds = Dataset.from_dict(data)

    COL = "prompt"
    N = 5

    print(f"Benchmarking dataset[COL][:N] (Slow)...")
    start = time.time()
    for _ in range(100):
        _ = ds[COL][:N]
    end = time.time()
    slow_time = (end - start) / 100
    print(f"Slow time: {slow_time:.6f}s")

    print(f"Benchmarking ds[:N][COL] (Fast)...")
    start = time.time()
    for _ in range(100):
        _ = ds[:N][COL]
    end = time.time()
    fast_time = (end - start) / 100
    print(f"Fast time: {fast_time:.6f}s")

    print(f"Speedup: {slow_time / fast_time:.2f}x")

if __name__ == "__main__":
    benchmark()
