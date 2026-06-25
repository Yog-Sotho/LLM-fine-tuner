
import time
import os
import sys
from datasets import Dataset

# Add current directory to sys.path
sys.path.append(os.getcwd())

from data.augmentation import augment_dataset_v27

def benchmark():
    N = 10000
    print(f"Benchmarking augmentation on {N} rows...")

    data = {
        "instruction": ["Tell me a story about a " + str(i) + " little cat." for i in range(N)],
        "output": ["Once upon a time, there was a " + str(i) + " little cat." for i in range(N)]
    }
    dataset = Dataset.from_dict(data)

    start = time.time()
    # Using 'random_word' as it's usually faster than 'synonym' and doesn't depend as much on large dictionaries
    aug_ds, msg = augment_dataset_v27(dataset, augmentation_factor=2, aug_type="random_word")
    end = time.time()

    print(msg)
    print(f"Time taken: {end - start:.4f}s")
    print(f"Rows per second: {N / (end - start):.2f}")

if __name__ == "__main__":
    benchmark()
