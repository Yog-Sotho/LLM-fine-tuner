
import time
from datasets import Dataset
import numpy as np

def benchmark():
    # Create a dummy dataset with 100,000 rows
    data = {"text": ["This is a test sentence number " + str(i) for i in range(100000)]}
    dataset = Dataset.from_dict(data)

    # Method 1: Row-wise iteration (List comprehension)
    start_time = time.time()
    texts_1 = [str(x["text"]) for x in dataset]
    end_time = time.time()
    print(f"Row-wise iteration: {end_time - start_time:.4f} seconds")

    # Method 2: Column access
    start_time = time.time()
    texts_2 = dataset["text"]
    end_time = time.time()
    print(f"Column access: {end_time - start_time:.4f} seconds")

    # Method 3: to_dict() access
    start_time = time.time()
    texts_3 = dataset.to_dict()["text"]
    end_time = time.time()
    print(f"to_dict() access: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
