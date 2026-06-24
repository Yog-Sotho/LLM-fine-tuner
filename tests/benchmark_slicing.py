
import time
from datasets import Dataset
import numpy as np

def benchmark():
    # Create a large dummy dataset
    N = 1000000
    data = {"text": ["Sentence " + str(i) for i in range(N)]}
    dataset = Dataset.from_dict(data)

    # Pre-warm
    _ = dataset[:5]["text"]

    # Method 1: Column first, then slice
    start_time = time.time()
    for _ in range(100):
        res1 = dataset["text"][:10]
    end_time = time.time()
    print(f"dataset[COL][:10]: {end_time - start_time:.6f} seconds")

    # Method 2: Slice first, then column
    start_time = time.time()
    for _ in range(100):
        res2 = dataset[:10]["text"]
    end_time = time.time()
    print(f"dataset[:10][COL]: {end_time - start_time:.6f} seconds")

    # Verify results are same
    assert dataset["text"][:10] == dataset[:10]["text"]

if __name__ == "__main__":
    benchmark()
