
import time
import numpy as np
from datasets import Dataset
import os

def benchmark_indexing():
    # Create a large dummy dataset (1 million rows)
    print("Creating 1M row dataset...")
    data = {"text": ["Row " + str(i) for i in range(1000000)]}
    dataset = Dataset.from_dict(data)

    # Save to disk and reload to ensure it's memory-mapped
    dataset.save_to_disk("test_dataset_indexing")
    from datasets import load_from_disk
    dataset = load_from_disk("test_dataset_indexing")

    N = 100

    # Warmup
    _ = dataset[0]

    print(f"Benchmarking dataset[COL][:N] vs dataset[:N][COL] for N={N}")

    # Method 1: dataset[COL][:N]
    start = time.time()
    for _ in range(100):
        _ = dataset["text"][:N]
    end = time.time()
    time_1 = (end - start) / 100
    print(f"dataset['text'][:N]: {time_1:.6f} seconds")

    # Method 2: dataset[:N][COL]
    start = time.time()
    for _ in range(100):
        _ = dataset[:N]["text"]
    end = time.time()
    time_2 = (end - start) / 100
    print(f"dataset[:N]['text']: {time_2:.6f} seconds")

    print(f"Speedup: {time_1 / time_2:.2f}x")

if __name__ == "__main__":
    benchmark_indexing()
