import time
import os
from datasets import Dataset
from data.augmentation import augment_dataset_v27

def benchmark_augmentation():
    # Create a dummy dataset (10k rows)
    num_rows = 10000
    data = {
        "instruction": ["Translate the following sentence to French: 'Hello, how are you?'"] * num_rows,
        "output": ["Bonjour, comment allez-vous ?"] * num_rows,
        "other_col": ["some other data"] * num_rows
    }
    ds = Dataset.from_dict(data)

    print(f"Benchmarking augment_dataset_v27 with {num_rows} rows...")

    # We'll use 'random_word' because it doesn't need external downloads (usually)
    # Actually, SynonymAug is already set up, let's use it.

    start_time = time.perf_counter()
    aug_ds, msg = augment_dataset_v27(ds, augmentation_factor=2, aug_type="synonym")
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Augmentation took {duration:.4f} seconds")
    print(f"Resulting dataset size: {len(aug_ds)}")
    print(f"Message: {msg}")

if __name__ == "__main__":
    benchmark_augmentation()
