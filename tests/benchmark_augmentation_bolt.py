
import time
import pandas as pd
from datasets import Dataset
import numpy as np

def original_augment_reconstruction(dataset, all_aug_versions, target_col, texts_to_aug):
    augmented_rows = []
    for idx, example in enumerate(dataset):
        # 1. Add original example
        augmented_rows.append(dict(example))
        # 2. Add each augmented version
        for version_list in all_aug_versions:
            new_example = dict(example)
            # Safeguard index in case nlpaug returns fewer items than requested
            aug_text = version_list[idx] if idx < len(version_list) else texts_to_aug[idx]
            new_example[target_col] = aug_text
            augmented_rows.append(new_example)
    return Dataset.from_list(augmented_rows)

def optimized_augment_reconstruction(dataset, all_aug_versions, target_col, texts_to_aug):
    df_orig = dataset.to_pandas()
    dfs = [df_orig]
    for aug_results in all_aug_versions:
        df_aug = df_orig.copy()
        if len(aug_results) < len(df_orig):
            # Pad if nlpaug returned fewer results
            aug_results = list(aug_results) + texts_to_aug[len(aug_results):]
        elif len(aug_results) > len(df_orig):
            # Truncate if nlpaug returned more (unlikely but safe)
            aug_results = aug_results[:len(df_orig)]

        df_aug[target_col] = aug_results
        dfs.append(df_aug)

    combined = pd.concat(dfs).sort_index(kind='stable')
    return Dataset.from_pandas(combined, preserve_index=False)

# Setup
N = 10000
augmentation_factor = 3
target_col = "text"
data = {target_col: [f"Text example {i}" for i in range(N)], "other": [i for i in range(N)]}
dataset = Dataset.from_dict(data)
texts_to_aug = data[target_col]

all_aug_versions = [
    [f"Augmented {j} version {i}" for i in range(N)]
    for j in range(augmentation_factor - 1)
]

print(f"Benchmarking augmentation reconstruction for {N} rows, factor {augmentation_factor}...")

start = time.time()
ds1 = original_augment_reconstruction(dataset, all_aug_versions, target_col, texts_to_aug)
end = time.time()
t_orig = end - start
print(f"Original reconstruction: {t_orig:.4f}s")

start = time.time()
ds2 = optimized_augment_reconstruction(dataset, all_aug_versions, target_col, texts_to_aug)
end = time.time()
t_opt = end - start
print(f"Optimized reconstruction: {t_opt:.4f}s")

print(f"Speedup: {t_orig / t_opt:.2f}x")

# Verification
assert len(ds1) == len(ds2)
# Check first few rows
print("Checking first 6 rows for parity...")
p1 = ds1.to_pandas().head(6)
p2 = ds2.to_pandas().head(6)
pd.testing.assert_frame_equal(p1, p2)
print("Parity check passed!")
