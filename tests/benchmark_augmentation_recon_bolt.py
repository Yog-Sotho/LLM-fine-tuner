import time
from datasets import Dataset

def benchmark_reconstruction(num_rows=10000, factor=2):
    # Setup
    data = {
        "text": ["This is a sample sentence for augmentation."] * num_rows,
        "other_col": ["some metadata"] * num_rows,
        "id": list(range(num_rows))
    }
    ds = Dataset.from_dict(data)
    target_col = "text"

    # Mock augmented versions
    all_aug_versions = [["Augmented: " + t for t in ds[target_col]]] * (factor - 1)

    # Original implementation (simulating the current one)
    start = time.time()
    augmented_rows = []
    # Simulating [str(x[target_col]) for x in ds]
    texts_to_aug = [str(x[target_col]) for x in ds]

    for idx, example in enumerate(ds):
        augmented_rows.append(dict(example))
        for version_list in all_aug_versions:
            new_example = dict(example)
            aug_text = version_list[idx] if idx < len(version_list) else texts_to_aug[idx]
            new_example[target_col] = aug_text
            augmented_rows.append(new_example)
    old_ds = Dataset.from_list(augmented_rows)
    old_time = time.time() - start

    # Optimized implementation
    start = time.time()
    num_rows_val = len(ds)
    new_data = {}
    original_dict = ds.to_dict()

    for col in ds.column_names:
        values = original_dict[col]
        interleaved = [None] * (num_rows_val * factor)
        # Fill original values
        interleaved[0::factor] = values

        if col == target_col:
            # Fill augmented versions
            for i, version_list in enumerate(all_aug_versions):
                fill_len = min(num_rows_val, len(version_list))
                interleaved[i+1::factor] = version_list[:fill_len] + values[fill_len:] if fill_len < num_rows_val else version_list
        else:
            # Fill with original values for other columns
            for i in range(1, factor):
                interleaved[i::factor] = values
        new_data[col] = interleaved

    new_ds = Dataset.from_dict(new_data)
    new_time = time.time() - start

    print(f"Old time: {old_time:.4f}s")
    print(f"New time: {new_time:.4f}s")
    print(f"Speedup: {old_time / new_time:.2f}x")

    # Verify correctness
    assert len(old_ds) == len(new_ds)
    # Compare dicts for equality
    d1 = old_ds.to_dict()
    d2 = new_ds.to_dict()
    for k in d1:
        if d1[k] != d2[k]:
            print(f"Mismatch in column {k}")
            raise AssertionError(f"Mismatch in column {k}")
    print("Verification passed!")

if __name__ == "__main__":
    benchmark_reconstruction(num_rows=10000, factor=2)
