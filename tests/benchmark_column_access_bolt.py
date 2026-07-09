
import time
from datasets import Dataset

def benchmark():
    print("Creating a large dataset (100,000 rows)...")
    data = {
        "text": ["This is a text " + str(i) for i in range(100000)],
    }
    ds = Dataset.from_dict(data)

    COL = "text"

    print(f"Benchmarking [str(x[COL]) for x in ds] (Slow)...")
    start = time.time()
    _ = [str(x[COL]) for x in ds]
    end = time.time()
    slow_time = end - start
    print(f"Slow time: {slow_time:.6f}s")

    print(f"Benchmarking list(ds[COL]) (Fast)...")
    start = time.time()
    _ = list(ds[COL])
    end = time.time()
    fast_time = end - start
    print(f"Fast time: {fast_time:.6f}s")

    print(f"Speedup: {slow_time / fast_time:.2f}x")

if __name__ == "__main__":
    benchmark()
