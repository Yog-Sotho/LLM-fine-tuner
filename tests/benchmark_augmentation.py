
import time
from datasets import Dataset

def benchmark():
    N = 100000
    factor = 3
    data = {
        "instruction": ["Instruction " + str(i) for i in range(N)],
        "output": ["Output " + str(i) for i in range(N)]
    }
    dataset = Dataset.from_dict(data)
    target_col = "instruction"
    all_aug_versions = [
        ["Aug " + str(i) + " version " + str(v) for i in range(N)]
        for v in range(factor - 1)
    ]

    # Method 1: Row-wise reconstruction (current)
    start_time = time.time()
    augmented_rows = []
    for idx, example in enumerate(dataset):
        augmented_rows.append(dict(example))
        for version_list in all_aug_versions:
            new_example = dict(example)
            aug_text = version_list[idx]
            new_example[target_col] = aug_text
            augmented_rows.append(new_example)
    ds1 = Dataset.from_list(augmented_rows)
    end_time = time.time()
    print(f"Row-wise reconstruction (100k): {end_time - start_time:.4f} seconds")

    # Method 2: Columnar reconstruction (optimized)
    start_time = time.time()
    new_data = {}
    ds_dict = dataset.to_dict()
    for col in dataset.column_names:
        orig_vals = ds_dict[col]
        new_vals = [None] * (len(orig_vals) * factor)
        new_vals[0::factor] = orig_vals
        if col == target_col:
            for i, aug_version in enumerate(all_aug_versions):
                new_vals[i+1::factor] = aug_version
        else:
            for i in range(1, factor):
                new_vals[i::factor] = orig_vals
        new_data[col] = new_vals
    ds2 = Dataset.from_dict(new_data)
    end_time = time.time()
    print(f"Columnar reconstruction (100k): {end_time - start_time:.4f} seconds")

    # Verify
    assert len(ds1) == len(ds2)
    assert ds1[0] == ds2[0]
    assert ds1[1] == ds2[1]
    assert ds1[2] == ds2[2]
    assert ds1[3] == ds2[3]

if __name__ == "__main__":
    benchmark()
