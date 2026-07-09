
import time
from datasets import Dataset

def benchmark():
    print("Creating a large dataset (1,000,000 rows)...")
    data = {
        "text": ["This is a text " + str(i) for i in range(1000000)],
    }
    ds = Dataset.from_dict(data)

    COL = "text"

    print(f"Benchmarking [x[COL] for x in ds] (Slow)...")
    start = time.time()
    # Only doing first 10k because 1M is too slow
    _ = [x[COL] for x in ds.select(range(10000))]
    end = time.time()
    slow_time_10k = end - start
    print(f"Slow time (10k): {slow_time_10k:.6f}s")

    print(f"Benchmarking ds[COL] (Fast)...")
    start = time.time()
    _ = ds[COL]
    end = time.time()
    fast_time_1M = end - start
    print(f"Fast time (1M): {fast_time_1M:.6f}s")

    print(f"Estimated speedup for 1M rows: {(slow_time_10k * 100) / fast_time_1M:.2f}x")

if __name__ == "__main__":
    benchmark()
