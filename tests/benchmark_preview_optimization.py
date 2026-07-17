import time
import pandas as pd
from datasets import Dataset
import numpy as np

def main():
    # Create a large dummy dataset (1,000,000 rows)
    print("Creating dummy dataset (1,000,000 rows)...")
    data = {
        "text": ["Sample sentence " + str(i) for i in range(1000000)],
        "instruction": ["Instruction " + str(i) for i in range(1000000)],
        "output": ["Output " + str(i) for i in range(1000000)]
    }
    ds = Dataset.from_dict(data)

    COL = "text"
    N = 10

    print(f"Benchmarking old pattern: ds['{COL}'][:{N}] (loads full column)...")
    # Warmup
    _ = ds[COL][:N]

    start = time.perf_counter()
    for _ in range(100):
        _ = ds[COL][:N]
    end = time.perf_counter()
    old_time = (end - start) / 100
    print(f"Average time (Old): {old_time:.6f} s")

    print(f"Benchmarking new pattern: ds[:{N}]['{COL}'] (slices first)...")
    # Warmup
    _ = ds[:N][COL]

    start = time.perf_counter()
    for _ in range(100):
        _ = ds[:N][COL]
    end = time.perf_counter()
    new_time = (end - start) / 100
    print(f"Average time (New): {new_time:.6f} s")

    speedup = old_time / new_time
    print(f"--- Results for 1M rows ---")
    print(f"Old: {old_time:.6f} s")
    print(f"New: {new_time:.6f} s")
    print(f"Speedup: {speedup:.2f}x")

    # Also benchmark DPO-like subset access
    print("\nBenchmarking DPO-like subset access...")

    start = time.perf_counter()
    for _ in range(100):
        _ = {
            "prompt": ds["instruction"][:5],
            "chosen": ds["output"][:5],
            "rejected": ds["text"][:5]
        }
    end = time.perf_counter()
    old_dpo_time = (end - start) / 100

    start = time.perf_counter()
    for _ in range(100):
        subset = ds[:5]
        _ = {
            "prompt": subset.get("instruction", []),
            "chosen": subset.get("output", []),
            "rejected": subset.get("text", [])
        }
    end = time.perf_counter()
    new_dpo_time = (end - start) / 100

    print(f"Old DPO Time: {old_dpo_time:.6f} s")
    print(f"New DPO Time: {new_dpo_time:.6f} s")
    print(f"DPO Speedup: {old_dpo_time / new_dpo_time:.2f}x")

if __name__ == "__main__":
    main()
