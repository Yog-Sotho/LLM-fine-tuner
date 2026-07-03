import time
import numpy as np
from datasets import Dataset

def benchmark():
    # Create a large dataset: 1 million rows, one column with some text
    num_rows = 1_000_000
    data = {"text": ["Hello world! This is a test sentence for benchmarking." for _ in range(num_rows)]}
    ds = Dataset.from_dict(data)

    N = 10

    # Pattern 1: dataset[COL][:N]
    start_time = time.time()
    for _ in range(100):
        _ = ds["text"][:N]
    end_time = time.time()
    pattern1_time = end_time - start_time
    print(f"Pattern 1 (ds[COL][:N]): {pattern1_time:.6f} seconds")

    # Pattern 2: dataset[:N][COL]
    start_time = time.time()
    for _ in range(100):
        _ = ds[:N]["text"]
    end_time = time.time()
    pattern2_time = end_time - start_time
    print(f"Pattern 2 (ds[:N][COL]): {pattern2_time:.6f} seconds")

    improvement = pattern1_time / pattern2_time
    print(f"Improvement: {improvement:.2f}x")

if __name__ == "__main__":
    benchmark()
