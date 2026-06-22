import time
from datasets import Dataset

def benchmark_column_access():
    # Create a large dataset (100k rows)
    data = {"text": ["This is a sample sentence."] * 100000}
    ds = Dataset.from_dict(data)

    COL = "text"

    # Method 1: [x[COL] for x in ds]
    start = time.perf_counter()
    res1 = [x[COL] for x in ds]
    end = time.perf_counter()
    time1 = end - start
    print(f"[x[COL] for x in ds]: {time1:.4f}s")

    # Method 2: ds[COL]
    start = time.perf_counter()
    res2 = ds[COL]
    end = time.perf_counter()
    time2 = end - start
    print(f"ds[COL]: {time2:.4f}s")

    print(f"Speedup: {time1 / time2:.2f}x")

if __name__ == "__main__":
    benchmark_column_access()
