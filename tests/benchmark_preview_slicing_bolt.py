import time
import pandas as pd
from datasets import Dataset
from data.preprocessing import preview_dataset
from config.constants import COL_INSTRUCTION, COL_OUTPUT, COL_TEXT, COL_PROMPT, COL_CHOSEN, COL_REJECTED

def benchmark_preview():
    # Create large datasets
    N = 1_000_000
    print(f"Creating large datasets with {N:,} rows...")

    sft_data = {
        COL_INSTRUCTION: ["Tell me a joke. " + str(i) for i in range(N)],
        COL_OUTPUT: ["Why did the chicken... " + str(i) for i in range(N)]
    }
    sft_ds = Dataset.from_dict(sft_data)

    dpo_data = {
        COL_PROMPT: ["Explain gravity. " + str(i) for i in range(N)],
        COL_CHOSEN: ["Gravity is... " + str(i) for i in range(N)],
        COL_REJECTED: ["Gravity is a mystery... " + str(i) for i in range(N)]
    }
    dpo_ds = Dataset.from_dict(dpo_data)

    print("\n--- Benchmarking SFT Preview ---")

    # Measure new optimized preview_dataset (which only slices once and direct accesses)
    t0 = time.perf_counter()
    for _ in range(100):
        _ = preview_dataset(sft_ds, is_dpo=False)
    t1 = time.perf_counter()
    optimized_sft_time = (t1 - t0) / 100
    print(f"Optimized SFT Preview: {optimized_sft_time:.6f}s per call")

    # Reconstruct the old redundant pattern for comparison
    def preview_dataset_old_pattern(dataset, is_dpo=False):
        if len(dataset) == 0:
            return pd.DataFrame({"Status": ["Dataset is empty."]})
        if is_dpo:
            subset = dataset[:5]
            batch = dataset[:5]
            return pd.DataFrame({
                COL_PROMPT:   subset.get(COL_PROMPT, []),
                COL_CHOSEN:   subset.get(COL_CHOSEN, []),
                COL_REJECTED: subset.get(COL_REJECTED, []),
            })
        else:
            subset = dataset[:5]
            batch = dataset[:5]
            inst_data = batch[COL_INSTRUCTION] if COL_INSTRUCTION in dataset.column_names else []
            out_data  = batch[COL_OUTPUT]      if COL_OUTPUT      in dataset.column_names else []
            return pd.DataFrame({
                COL_INSTRUCTION: subset.get(COL_INSTRUCTION, []),
                COL_OUTPUT:      subset.get(COL_OUTPUT, []),
            })

    t0 = time.perf_counter()
    for _ in range(100):
        _ = preview_dataset_old_pattern(sft_ds, is_dpo=False)
    t1 = time.perf_counter()
    old_sft_time = (t1 - t0) / 100
    print(f"Old Redundant SFT Preview: {old_sft_time:.6f}s per call")
    print(f"SFT Speedup: {old_sft_time / optimized_sft_time:.2f}x")

    print("\n--- Benchmarking DPO Preview ---")
    t0 = time.perf_counter()
    for _ in range(100):
        _ = preview_dataset(dpo_ds, is_dpo=True)
    t1 = time.perf_counter()
    optimized_dpo_time = (t1 - t0) / 100
    print(f"Optimized DPO Preview: {optimized_dpo_time:.6f}s per call")

    t0 = time.perf_counter()
    for _ in range(100):
        _ = preview_dataset_old_pattern(dpo_ds, is_dpo=True)
    t1 = time.perf_counter()
    old_dpo_time = (t1 - t0) / 100
    print(f"Old Redundant DPO Preview: {old_dpo_time:.6f}s per call")
    print(f"DPO Speedup: {old_dpo_time / optimized_dpo_time:.2f}x")

if __name__ == "__main__":
    benchmark_preview()
