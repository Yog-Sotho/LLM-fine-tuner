
import time
import pandas as pd
import numpy as np
from datasets import Dataset
import os
import sys

def original_reconstruction(dataset, augmentation_factor, target_col, all_aug_versions):
    augmented_rows = []
    # This imitates the old loop in augment_dataset_v27
    for idx, example in enumerate(dataset):
        # 1. Add original example
        augmented_rows.append(dict(example))
        # 2. Add each augmented version
        for version_list in all_aug_versions:
            new_example = dict(example)
            aug_text = version_list[idx] if idx < len(version_list) else dataset[idx][target_col]
            new_example[target_col] = aug_text
            augmented_rows.append(new_example)
    return Dataset.from_list(augmented_rows)

def optimized_reconstruction(dataset, augmentation_factor, target_col, all_aug_versions):
    data_dict = dataset.to_dict()
    num_rows = len(dataset)
    new_num_rows = num_rows * augmentation_factor
    new_data = {}

    for col_name, values in data_dict.items():
        interleaved = [None] * new_num_rows
        interleaved[0::augmentation_factor] = values
        if col_name == target_col:
            for i, aug_version in enumerate(all_aug_versions):
                interleaved[i + 1::augmentation_factor] = aug_version[:num_rows]
        else:
            for i in range(1, augmentation_factor):
                interleaved[i::augmentation_factor] = values
        new_data[col_name] = interleaved

    return Dataset.from_dict(new_data)

def main():
    num_samples = 10000
    augmentation_factor = 2
    target_col = "text"

    print(f"Benchmarking pure reconstruction with {num_samples} samples and factor {augmentation_factor}...")

    # Create dummy dataset
    df = pd.DataFrame({
        "text": ["This is sample text " + str(i) for i in range(num_samples)],
        "other": ["other " + str(i) for i in range(num_samples)]
    })
    dataset = Dataset.from_pandas(df)

    # Mock augmented versions
    all_aug_versions = [[t + " (aug)" for t in dataset[target_col]] for _ in range(augmentation_factor - 1)]

    # Measure Original
    start = time.time()
    _ = original_reconstruction(dataset, augmentation_factor, target_col, all_aug_versions)
    original_time = time.time() - start
    print(f"Original reconstruction time: {original_time:.4f}s")

    # Measure Optimized
    start = time.time()
    _ = optimized_reconstruction(dataset, augmentation_factor, target_col, all_aug_versions)
    optimized_time = time.time() - start
    print(f"Optimized reconstruction time: {optimized_time:.4f}s")

    print(f"Pure speedup: {original_time / optimized_time:.2f}x")

if __name__ == "__main__":
    main()
