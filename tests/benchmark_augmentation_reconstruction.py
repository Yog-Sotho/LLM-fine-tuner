
import time
import pandas as pd
from datasets import Dataset

def benchmark_reconstruction():
    # 10k rows, with a few columns
    n = 10000
    data = {
        "instruction": ["instruction " + str(i) for i in range(n)],
        "output": ["output " + str(i) for i in range(n)],
        "other_col": ["meta " + str(i) for i in range(n)],
    }
    dataset = Dataset.from_dict(data)
    target_col = "instruction"
    augmentation_factor = 2

    # Mock augmented versions (one version list)
    all_aug_versions = [["aug " + str(i) for i in range(n)]]
    texts_to_aug = data[target_col]

    print(f"Benchmarking reconstruction with {n} rows and factor {augmentation_factor}...")

    # --- Method 1: Row-wise reconstruction (Current) ---
    t0 = time.time()
    augmented_rows = []
    for idx, example in enumerate(dataset):
        # 1. Add original example
        augmented_rows.append(dict(example))
        # 2. Add each augmented version
        for version_list in all_aug_versions:
            new_example = dict(example)
            aug_text = version_list[idx] if idx < len(version_list) else texts_to_aug[idx]
            new_example[target_col] = aug_text
            augmented_rows.append(new_example)
    aug_ds_1 = Dataset.from_list(augmented_rows)
    t1 = time.time()
    time_rowwise = t1 - t0
    print(f"Row-wise reconstruction took: {time_rowwise:.4f}s")

    # --- Method 2: Columnar reconstruction (Optimized) ---
    t0 = time.time()
    num_versions = len(all_aug_versions)
    factor = num_versions + 1

    interleaved_data = dataset.to_dict()
    for col in interleaved_data:
        original_values = interleaved_data[col]
        expanded = [None] * (len(original_values) * factor)
        expanded[0::factor] = original_values
        if col != target_col:
            for i in range(1, factor):
                expanded[i::factor] = original_values
        else:
            for i, version_list in enumerate(all_aug_versions):
                expanded[i+1::factor] = version_list
        interleaved_data[col] = expanded

    aug_ds_2 = Dataset.from_dict(interleaved_data)
    t1 = time.time()
    time_columnar = t1 - t0
    print(f"Columnar reconstruction took: {time_columnar:.4f}s")

    print(f"Speedup: {time_rowwise/time_columnar:.2f}x")

    # Verify equality
    assert len(aug_ds_1) == len(aug_ds_2)
    # Check a few rows
    assert aug_ds_1[0] == aug_ds_2[0]
    assert aug_ds_1[1] == aug_ds_2[1]
    assert aug_ds_1[-1] == aug_ds_2[-1]
    print("Verification successful: Results are identical.")

if __name__ == "__main__":
    benchmark_reconstruction()
