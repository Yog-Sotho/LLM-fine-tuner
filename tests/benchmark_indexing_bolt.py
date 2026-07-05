
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
