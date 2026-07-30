import time
import pandas as pd
from datasets import Dataset

COL_PROMPT = "prompt"
COL_CHOSEN = "chosen"
COL_REJECTED = "rejected"

def current_logic(df):
    original_len = len(df)
    # Vectorized strip and empty check for DPO
    p_stripped = df[COL_PROMPT].astype(str).str.strip()
    c_stripped = df[COL_CHOSEN].astype(str).str.strip()
    r_stripped = df[COL_REJECTED].astype(str).str.strip()
    mask = (p_stripped != "") & (c_stripped != "") & (r_stripped != "")

    # Calculate lengths on ALL rows before deduplication
    lengths = p_stripped.str.len() + c_stripped.str.len() + r_stripped.str.len()

    df = df[mask].reset_index(drop=True)
    lengths = lengths[mask].reset_index(drop=True)

    pre_dup_len = len(df)
    df = df.drop_duplicates(subset=[COL_PROMPT, COL_CHOSEN, COL_REJECTED], keep='first')

    # Redundant index lookup/alignment
    lengths = lengths.loc[df.index].reset_index(drop=True)
    df = df.reset_index(drop=True)
    return df, lengths

def optimized_logic(df):
    original_len = len(df)
    # Vectorized strip and empty check for DPO
    p_stripped = df[COL_PROMPT].astype(str).str.strip()
    c_stripped = df[COL_CHOSEN].astype(str).str.strip()
    r_stripped = df[COL_REJECTED].astype(str).str.strip()
    mask = (p_stripped != "") & (c_stripped != "") & (r_stripped != "")

    df = df[mask].reset_index(drop=True)

    pre_dup_len = len(df)
    df = df.drop_duplicates(subset=[COL_PROMPT, COL_CHOSEN, COL_REJECTED], keep='first').reset_index(drop=True)

    # Calculate lengths ONLY on final deduplicated rows
    if len(df) > 0:
        lengths = df[COL_PROMPT].astype(str).str.strip().str.len() + \
                  df[COL_CHOSEN].astype(str).str.strip().str.len() + \
                  df[COL_REJECTED].astype(str).str.strip().str.len()
    else:
        lengths = pd.Series(dtype=int)
    return df, lengths

def benchmark():
    N = 100000
    print(f"Generating {N} rows of dummy DPO data with duplicates...")
    data = {
        COL_PROMPT: ["Prompt " + str(i) for i in range(N)],
        COL_CHOSEN: ["Chosen " + str(i) for i in range(N)],
        COL_REJECTED: ["Rejected " + str(i) for i in range(N)]
    }
    # Create 50% duplicates
    for i in range(N // 2):
        data[COL_PROMPT][i + N // 2] = data[COL_PROMPT][i]
        data[COL_CHOSEN][i + N // 2] = data[COL_CHOSEN][i]
        data[COL_REJECTED][i + N // 2] = data[COL_REJECTED][i]

    df = pd.DataFrame(data)

    print("Benchmarking current logic...")
    t0 = time.time()
    for _ in range(10):
        df_cur, len_cur = current_logic(df.copy())
    cur_time = time.time() - t0
    print(f"Current logic took: {cur_time:.4f}s")

    print("Benchmarking optimized logic...")
    t0 = time.time()
    for _ in range(10):
        df_opt, len_opt = optimized_logic(df.copy())
    opt_time = time.time() - t0
    print(f"Optimized logic took: {opt_time:.4f}s")

    print(f"Speedup: {cur_time / opt_time:.2f}x")

    # Assert correctness
    assert len(df_cur) == len(df_opt)
    assert len(len_cur) == len(len_opt)
    pd.testing.assert_series_equal(len_cur, len_opt, check_names=False)
    print("Verification passed! Both methods yield exact parity.")

if __name__ == "__main__":
    benchmark()
