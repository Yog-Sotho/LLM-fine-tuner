
import time
import pandas as pd
from datasets import Dataset
import numpy as np

def benchmark_slicing():
    # Create a large dummy dataset (1M rows)
    num_rows = 1_000_000
    data = {
        "text": ["This is a test sentence for benchmarking purposes." for _ in range(num_rows)],
        "instruction": ["Instruction " + str(i) for i in range(num_rows)],
        "output": ["Output " + str(i) for i in range(num_rows)]
    }
    ds = Dataset.from_dict(data)

    N = 10

    # Method 1: dataset[COL][:N]
    start_time = time.time()
    for _ in range(100):
        _ = ds["text"][:N]
    duration_1 = time.time() - start_time
    print(f"dataset[COL][:N] took: {duration_1:.6f}s")

    # Method 2: dataset[:N][COL]
    start_time = time.time()
    for _ in range(100):
        _ = ds[:N]["text"]
    duration_2 = time.time() - start_time
    print(f"dataset[:N][COL] took: {duration_2:.6f}s")

    improvement = duration_1 / duration_2
    print(f"Improvement: {improvement:.2f}x")

if __name__ == "__main__":
    benchmark_slicing()
