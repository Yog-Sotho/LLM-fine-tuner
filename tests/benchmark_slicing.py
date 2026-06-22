import time
import pandas as pd
from datasets import Dataset

def benchmark_slicing():
    # Create a large dataset (100k rows)
    data = {"text": ["This is a sample sentence."] * 1000000}
    ds = Dataset.from_dict(data)

    N = 10
    COL = "text"

    # Warmup
    _ = ds[COL][:N]
    _ = ds[:N][COL]

    # Method 1: dataset[COL][:N]
    start = time.perf_counter()
    for _ in range(100):
        _ = ds[COL][:N]
    end = time.perf_counter()
    time1 = (end - start) / 100
    print(f"dataset[COL][:N]: {time1:.8f}s")

    # Method 2: dataset[:N][COL]
    start = time.perf_counter()
    for _ in range(100):
        _ = ds[:N][COL]
    end = time.perf_counter()
    time2 = (end - start) / 100
    print(f"dataset[:N][COL]: {time2:.8f}s")

    print(f"Speedup: {time1 / time2:.2f}x")

if __name__ == "__main__":
    benchmark_slicing()
