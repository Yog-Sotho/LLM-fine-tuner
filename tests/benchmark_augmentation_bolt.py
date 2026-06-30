
import time
import os
import sys
import numpy as np
import pandas as pd
from datasets import Dataset

# Add current directory to path
sys.path.append(os.getcwd())

from config.constants import COL_TEXT

def benchmark_augmentation_reconstruction():
    print("Creating 100k row dataset...")
    data = {COL_TEXT: ["This is example " + str(i) for i in range(100000)]}
    dataset = Dataset.from_dict(data)

    augmentation_factor = 2
    texts_to_aug = dataset[COL_TEXT]
    all_aug_versions = [[t + " (aug)" for t in texts_to_aug]]

    print("\nBenchmarking current reconstruction logic (100k rows)...")
    start_time = time.time()

    augmented_rows = []
    for idx, example in enumerate(dataset):
        augmented_rows.append(dict(example))
        for version_list in all_aug_versions:
            new_example = dict(example)
            aug_text = version_list[idx] if idx < len(version_list) else texts_to_aug[idx]
            new_example[COL_TEXT] = aug_text
            augmented_rows.append(new_example)

    aug_ds = Dataset.from_list(augmented_rows)
    end_time = time.time()
    current_time = end_time - start_time
    print(f"Current reconstruction time: {current_time:.4f} seconds")

    print("\nBenchmarking optimized reconstruction logic (Pandas) (100k rows)...")
    start_time = time.time()

    df = dataset.to_pandas()
    dfs = [df]
    for version_list in all_aug_versions:
        aug_df = df.copy()
        aug_df[COL_TEXT] = version_list
        dfs.append(aug_df)

    # Interleave by concatenating and then sorting or just using a specific order
    # The requirement is: original, aug1, aug2, original, aug1, aug2...
    # We can use pd.concat and then reshape/reindex

    combined_df = pd.concat(dfs, axis=0).sort_index(kind='stable')
    aug_ds_opt = Dataset.from_pandas(combined_df, preserve_index=False)

    end_time = time.time()
    opt_time = end_time - start_time
    print(f"Optimized reconstruction time: {opt_time:.4f} seconds")
    print(f"Speedup: {current_time / opt_time:.2f}x")

if __name__ == "__main__":
    benchmark_augmentation_reconstruction()
