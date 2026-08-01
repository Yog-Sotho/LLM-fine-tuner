import time
import pandas as pd
from datasets import Dataset

COL_PROMPT = "prompt"
COL_CHOSEN = "chosen"
COL_REJECTED = "rejected"

def current_logic(df, is_dpo=True):
    # ── Single-pass validation and filtering ──────────────────────────────
    if is_dpo:
        # Vectorized strip and empty check for DPO
        p_stripped = df[COL_PROMPT].astype(str).str.strip()
        c_stripped = df[COL_CHOSEN].astype(str).str.strip()
        r_stripped = df[COL_REJECTED].astype(str).str.strip()
        mask = (p_stripped != "") & (c_stripped != "") & (r_stripped != "")

    # Filter rows
    df = df[mask].reset_index(drop=True)

    # ── Duplicate detection AND removal ──────────────────────────
    df = df.drop_duplicates(subset=[COL_PROMPT, COL_CHOSEN, COL_REJECTED], keep='first').reset_index(drop=True)

    # Calculate lengths (redundant .astype(str).str.strip())
    if len(df) > 0:
        lengths = df[COL_PROMPT].astype(str).str.strip().str.len() + \
                  df[COL_CHOSEN].astype(str).str.strip().str.len() + \
                  df[COL_REJECTED].astype(str).str.strip().str.len()
    else:
        lengths = pd.Series(dtype=int)
    return df, lengths

def optimized_logic(df, is_dpo=True):
    # ── Single-pass validation and filtering ──────────────────────────────
    if is_dpo:
        # Vectorized strip and empty check for DPO - in-place!
        df[COL_PROMPT] = df[COL_PROMPT].astype(str).str.strip()
        df[COL_CHOSEN] = df[COL_CHOSEN].astype(str).str.strip()
        df[COL_REJECTED] = df[COL_REJECTED].astype(str).str.strip()
        mask = (df[COL_PROMPT] != "") & (df[COL_CHOSEN] != "") & (df[COL_REJECTED] != "")

    # Filter rows
    df = df[mask].reset_index(drop=True)

    # ── Duplicate detection AND removal ──────────────────────────
    df = df.drop_duplicates(subset=[COL_PROMPT, COL_CHOSEN, COL_REJECTED], keep='first').reset_index(drop=True)

    # Calculate lengths on already stripped/string columns!
    if len(df) > 0:
        lengths = df[COL_PROMPT].str.len() + \
                  df[COL_CHOSEN].str.len() + \
                  df[COL_REJECTED].str.len()
    else:
        lengths = pd.Series(dtype=int)
    return df, lengths

def benchmark():
    N = 100000
    print(f"Generating {N} rows of dummy DPO data...")
    data = {
        COL_PROMPT: ["  Prompt  " + str(i) + "   " for i in range(N)],
        COL_CHOSEN: ["   Chosen " + str(i) + "  " for i in range(N)],
        COL_REJECTED: [" Rejected " + str(i) + " " for i in range(N)]
    }
    df = pd.DataFrame(data)

    print("Running current logic 10 times...")
    t0 = time.time()
    for _ in range(10):
        df_cur, len_cur = current_logic(df.copy())
    cur_time = time.time() - t0
    print(f"Current logic took: {cur_time:.4f}s")

    print("Running optimized logic 10 times...")
    t0 = time.time()
    for _ in range(10):
        df_opt, len_opt = optimized_logic(df.copy())
    opt_time = time.time() - t0
    print(f"Optimized logic took: {opt_time:.4f}s")

    print(f"Speedup: {cur_time / opt_time:.2f}x")

    # Assert correctness (stripping df_cur columns for comparison)
    assert len(df_cur) == len(df_opt)
    assert len(len_cur) == len(len_opt)
    df_cur_stripped = df_cur.copy()
    for col in [COL_PROMPT, COL_CHOSEN, COL_REJECTED]:
        df_cur_stripped[col] = df_cur_stripped[col].astype(str).str.strip()
    pd.testing.assert_frame_equal(df_cur_stripped, df_opt)
    pd.testing.assert_series_equal(len_cur, len_opt, check_names=False)
    print("Verification passed! Exact parity achieved.")

if __name__ == "__main__":
    benchmark()
