import time
from datasets import Dataset

def benchmark():
    num_rows = 100_000
    data = {"text": ["Hello world! " * 10 for _ in range(num_rows)]}
    ds = Dataset.from_dict(data)

    # Row-wise
    start = time.time()
    _ = [str(x["text"]) for x in ds]
    row_time = time.time() - start
    print(f"Row-wise access: {row_time:.6f}s")

    # Columnar
    start = time.time()
    _ = ds["text"]
    col_time = time.time() - start
    print(f"Columnar access: {col_time:.6f}s")

    print(f"Speedup: {row_time/col_time:.2f}x")

if __name__ == "__main__":
    benchmark()
