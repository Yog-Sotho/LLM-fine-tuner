import time
import pandas as pd
from datasets import Dataset

# Mock constants
COL_TEXT = "text"

def augment_reconstruct_old(dataset, all_aug_versions, target_col, augmentation_factor):
    augmented_rows = []
    texts_to_aug = list(dataset[target_col])
    for idx, example in enumerate(dataset):
        # 1. Add original example
        augmented_rows.append(dict(example))
        # 2. Add each augmented version
        for version_list in all_aug_versions:
            new_example = dict(example)
            aug_text = version_list[idx] if idx < len(version_list) else texts_to_aug[idx]
            new_example[target_col] = aug_text
            augmented_rows.append(new_example)
    return Dataset.from_list(augmented_rows)

def augment_reconstruct_new(dataset, all_aug_versions, target_col, augmentation_factor):
    df_orig = dataset.to_pandas()
    all_dfs = [df_orig]
    texts_to_aug = list(dataset[target_col])

    for version_list in all_aug_versions:
        df_aug = df_orig.copy()
        if len(version_list) < len(df_orig):
            version_list = list(version_list) + texts_to_aug[len(version_list):]
        df_aug[target_col] = version_list[:len(df_orig)]
        all_dfs.append(df_aug)

    combined = pd.concat(all_dfs).sort_index(kind='stable')
    return Dataset.from_pandas(combined, preserve_index=False)

def benchmark():
    num_rows = 50_000
    augmentation_factor = 2
    print(f"Benchmarking reconstruction with {num_rows} rows, factor {augmentation_factor}...")

    data = {COL_TEXT: ["Hello world! " * 5 for _ in range(num_rows)], "other": [i for i in range(num_rows)]}
    ds = Dataset.from_dict(data)

    all_aug_versions = [["Augmented " * 5 for _ in range(num_rows)]]

    # Old way
    start = time.time()
    ds_old = augment_reconstruct_old(ds, all_aug_versions, COL_TEXT, augmentation_factor)
    old_time = time.time() - start
    print(f"Old reconstruction: {old_time:.4f}s")

    # New way
    start = time.time()
    ds_new = augment_reconstruct_new(ds, all_aug_versions, COL_TEXT, augmentation_factor)
    new_time = time.time() - start
    print(f"New reconstruction: {new_time:.4f}s")

    print(f"Speedup: {old_time/new_time:.2f}x")

    # Verify parity
    assert len(ds_old) == len(ds_new)
    # Check some samples
    assert ds_old[0][COL_TEXT] == ds_new[0][COL_TEXT]
    assert ds_old[1][COL_TEXT] == ds_new[1][COL_TEXT]
    assert ds_old[1]["other"] == ds_new[1]["other"]

if __name__ == "__main__":
    benchmark()
