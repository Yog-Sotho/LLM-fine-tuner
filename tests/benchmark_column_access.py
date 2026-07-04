import time
from datasets import Dataset

def benchmark():
    N = 100000
    data = {"text": ["row " + str(i) for i in range(N)]}
    ds = Dataset.from_dict(data)

    print(f"Dataset size: {len(ds)} rows")

    # Method 1: [x[COL] for x in ds]
    start = time.time()
    _ = [x["text"] for x in ds]
    end = time.time()
    old_time = end - start
    print(f"Method 1 ([x[COL] for x in ds]): {old_time:.6f}s")

    # Method 2: ds[COL]
    start = time.time()
    _ = ds["text"]
    end = time.time()
    new_time = end - start
    print(f"Method 2 (ds[COL]): {new_time:.6f}s")

    print(f"Speedup: {old_time / new_time:.2f}x")

if __name__ == "__main__":
    benchmark()
