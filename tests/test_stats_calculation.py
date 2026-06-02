import pytest
from unittest.mock import MagicMock
import pandas as pd
from datasets import Dataset

# Mock constants since we can't easily import them without full env
COL_INSTRUCTION = "instruction"
COL_OUTPUT = "output"
COL_TEXT = "text"
COL_PROMPT = "prompt"
COL_CHOSEN = "chosen"
COL_REJECTED = "rejected"

def calculate_stats_optimized(ds):
    # This is the logic I implemented in handlers.py
    try:
        df_stats = ds.to_pandas()
        if COL_PROMPT in df_stats.columns and COL_CHOSEN in df_stats.columns:
            _lengths = (
                df_stats[COL_PROMPT].astype(str).str.len() +
                df_stats[COL_CHOSEN].astype(str).str.len() +
                df_stats[COL_REJECTED].astype(str).str.len()
            ).tolist()
        elif COL_TEXT in df_stats.columns:
            _lengths = df_stats[COL_TEXT].astype(str).str.len().tolist()
        elif COL_INSTRUCTION in df_stats.columns and COL_OUTPUT in df_stats.columns:
            _lengths = (
                df_stats[COL_INSTRUCTION].astype(str).str.len() +
                df_stats[COL_OUTPUT].astype(str).str.len()
            ).tolist()
        else:
            first_col = df_stats.columns[0] if not df_stats.empty else None
            _lengths = df_stats[first_col].astype(str).str.len().tolist() if first_col else []
    except Exception:
        _lengths = [100] * len(ds)
    return _lengths

def test_stats_sft_instruction_output():
    data = {
        COL_INSTRUCTION: ["inst1", "inst22"],
        COL_OUTPUT: ["out111", "out2"]
    }
    ds = Dataset.from_dict(data)
    lengths = calculate_stats_optimized(ds)
    assert lengths == [5+6, 6+4]

def test_stats_sft_text():
    data = {
        COL_TEXT: ["text1", "text222"]
    }
    ds = Dataset.from_dict(data)
    lengths = calculate_stats_optimized(ds)
    assert lengths == [5, 7]

def test_stats_dpo():
    data = {
        COL_PROMPT: ["p1"],
        COL_CHOSEN: ["c11"],
        COL_REJECTED: ["r111"]
    }
    ds = Dataset.from_dict(data)
    lengths = calculate_stats_optimized(ds)
    assert lengths == [2+3+4]

def test_stats_fallback():
    data = {
        "unknown": ["val11", "val2"]
    }
    ds = Dataset.from_dict(data)
    lengths = calculate_stats_optimized(ds)
    assert lengths == [5, 4]

def test_stats_empty():
    ds = Dataset.from_dict({})
    # Dataset.from_dict({}) might not have columns, handle it
    lengths = calculate_stats_optimized(ds)
    assert lengths == []
