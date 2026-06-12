import pytest
import numpy as np
from datasets import Dataset
from data.preprocessing import get_dataset_stats
from config.constants import (
    COL_PROMPT, COL_CHOSEN, COL_REJECTED,
    COL_TEXT, COL_INSTRUCTION, COL_OUTPUT
)

def test_get_dataset_stats_sft_text():
    data = {COL_TEXT: ["hello", "world", "vectorized"]}
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)

    # lengths: 5, 5, 10 -> mean = 6.666...
    assert stats["num_examples"] == 3
    assert pytest.approx(stats["avg_length"]) == 20 / 3

def test_get_dataset_stats_sft_instruction():
    data = {
        COL_INSTRUCTION: ["tell me a joke", "what is 2+2"],
        COL_OUTPUT: ["no", "4"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)

    # lengths: (14+2), (11+1) -> 16, 12 -> mean = 14.0
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 14.0

def test_get_dataset_stats_dpo():
    data = {
        COL_PROMPT: ["p1", "p2"],
        COL_CHOSEN: ["c1", "c2"],
        COL_REJECTED: ["r1", "r2"]
    }
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds, is_dpo=True)

    # lengths: (2+2+2), (2+2+2) -> 6, 6 -> mean = 6.0
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 6.0

def test_get_dataset_stats_empty():
    ds = Dataset.from_dict({COL_TEXT: []})
    stats = get_dataset_stats(ds)
    assert stats["num_examples"] == 0
    assert stats["avg_length"] == 0.0

def test_get_dataset_stats_fallback():
    # Unknown column
    data = {"random": ["abc", "de"]}
    ds = Dataset.from_dict(data)
    stats = get_dataset_stats(ds)

    # lengths: 3, 2 -> mean = 2.5
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == 2.5
