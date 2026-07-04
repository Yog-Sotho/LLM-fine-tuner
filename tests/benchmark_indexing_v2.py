import time
from datasets import Dataset

def benchmark():
    # Create a large-ish dataset
    data = {"text": ["row " + str(i) for i in range(1000000)]}
    ds = Dataset.from_dict(data)

    print(f"Dataset size: {len(ds)} rows")

    # Method 1: ds[COL][:N]
    start = time.time()
    for _ in range(100):
        _ = ds["text"][:10]
    end = time.time()
    old_time = (end - start) / 100
    print(f"Method 1 (ds[COL][:10]): {old_time:.6f}s per call")

    # Method 2: ds[:N][COL]
    start = time.time()
    for _ in range(100):
        _ = ds[:10]["text"]
    end = time.time()
    new_time = (end - start) / 100
    print(f"Method 2 (ds[:10][COL]): {new_time:.6f}s per call")

    print(f"Speedup: {old_time / new_time:.2f}x")

if __name__ == "__main__":
    benchmark()
