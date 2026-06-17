import time
from datasets import Dataset

def original_reconstruction(dataset, all_aug_versions, target_col):
    texts_to_aug = [str(x[target_col]) for x in dataset]
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

def optimized_reconstruction(dataset, all_aug_versions, target_col):
    factor = len(all_aug_versions) + 1
    full_dict = dataset.to_dict()
    new_data = {}

    total_rows = len(dataset) * factor

    for col in dataset.column_names:
        original_values = full_dict[col]
        interleaved = [None] * total_rows

        # Place original values
        interleaved[0::factor] = original_values

        if col == target_col:
            # Place augmented versions
            for i, version_list in enumerate(all_aug_versions):
                interleaved[i+1::factor] = version_list
        else:
            # Repeat original values for other columns
            for i in range(1, factor):
                interleaved[i::factor] = original_values

        new_data[col] = interleaved

    return Dataset.from_dict(new_data)

# Benchmark
N = 10000
ds = Dataset.from_dict({
    "instruction": ["Instruction " + str(i) for i in range(N)],
    "output": ["Output " + str(i) for i in range(N)],
    "other": ["Other " + str(i) for i in range(N)]
})
target_col = "instruction"
all_aug_versions = [["Aug " + str(i) for i in range(N)]] # factor = 2

print(f"Benchmarking with {N} rows and factor 2...")

start = time.time()
original_reconstruction(ds, all_aug_versions, target_col)
print(f"Original took: {time.time() - start:.4f}s")

start = time.time()
optimized_reconstruction(ds, all_aug_versions, target_col)
print(f"Optimized took: {time.time() - start:.4f}s")
