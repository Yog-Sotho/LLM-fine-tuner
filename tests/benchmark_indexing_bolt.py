
import time
import os
import pandas as pd
from datasets import Dataset

# Create a large-ish dataset
N = 1_000_000
print(f"Creating dataset with {N} rows...")
data = {
    "instruction": ["Instruction " + str(i) for i in range(N)],
    "output": ["Output " + str(i) for i in range(N)]
}
dataset = Dataset.from_dict(data)

def benchmark_indexing():
    print("\n--- Benchmarking Indexing ---")

    # Pattern 1: dataset[COL][:N] (Current)
    start = time.time()
    for _ in range(10):
        _ = dataset["instruction"][:5]
        _ = dataset["output"][:5]
    end = time.time()
    print(f"Current pattern (dataset[COL][:5]): {(end - start)/10:.6f}s per call")

    # Pattern 2: dataset[:N][COL] (Optimized)
    start = time.time()
    for _ in range(10):
        _ = dataset[:5]["instruction"]
        _ = dataset[:5]["output"]
    end = time.time()
    print(f"Optimized pattern (dataset[:5][COL]): {(end - start)/10:.6f}s per call")

if __name__ == "__main__":
    benchmark_indexing()
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
