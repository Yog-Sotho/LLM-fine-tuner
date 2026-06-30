
import time
from datasets import Dataset
import os
import pandas as pd

def benchmark_indexing():
    # Create a large dummy dataset (100k rows)
    print("Creating 100k row dataset...")
    data = {"text": ["Row " + str(i) for i in range(100000)]}
    dataset = Dataset.from_dict(data)

    N = 5
    iters = 1000

    print(f"Benchmarking dataset[COL][:N] vs dataset[:N][COL] for N={N}, iters={iters}")

    # Method 1: dataset[COL][:N]
    start = time.time()
    for _ in range(iters):
        _ = dataset["text"][:N]
    end = time.time()
    time_1 = (end - start) / iters
    print(f"dataset['text'][:N]: {time_1:.8f} seconds")

    # Method 2: dataset[:N][COL]
    start = time.time()
    for _ in range(iters):
        _ = dataset[:N]["text"]
    end = time.time()
    time_2 = (end - start) / iters
    print(f"dataset[:N]['text']: {time_2:.8f} seconds")

    print(f"Speedup: {time_1 / time_2:.2f}x")

if __name__ == "__main__":
    benchmark_indexing()
