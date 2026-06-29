import time
import pandas as pd
from datasets import Dataset
from data.augmentation import augment_dataset_v27
from data.preprocessing import preview_dataset

def benchmark_augmentation_reconstruction():
    print("\n--- Benchmarking Augmentation Reconstruction ---")
    num_rows = 10000
    data = {
        "text": ["This is a sample sentence for augmentation."] * num_rows,
        "other": ["metadata"] * num_rows
    }
    ds = Dataset.from_dict(data)

    # Mock nlpaug being missing or using a simple repeat if it fails
    # To actually measure the RECONSTRUCTION (from_dict vs from_list loop)
    # we'll time the function call with a factor of 2.

    start = time.time()
    _, _ = augment_dataset_v27(ds, augmentation_factor=2, aug_type="synonym")
    end = time.time()
    print(f"Augmentation (10k rows, factor=2) took: {end - start:.4f}s")

def benchmark_preview_slicing():
    print("\n--- Benchmarking Preview Slicing ---")
    num_rows = 1_000_000
    data = {"text": ["Large dataset row"] * num_rows}
    ds = Dataset.from_dict(data)

    start = time.time()
    for _ in range(10):
        _ = preview_dataset(ds)
    end = time.time()
    print(f"Preview (1M rows, 10 iterations) took: {end - start:.6f}s total")
    print(f"Average preview time: {(end - start)/10:.6f}s")

if __name__ == "__main__":
    benchmark_augmentation_reconstruction()
    benchmark_preview_slicing()
