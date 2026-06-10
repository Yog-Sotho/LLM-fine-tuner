import pytest
from datasets import Dataset
from data.preprocessing import get_dataset_stats
from config.constants import COL_PROMPT, COL_CHOSEN, COL_REJECTED, COL_TEXT, COL_INSTRUCTION, COL_OUTPUT

def test_get_dataset_stats_sft_text():
    ds = Dataset.from_dict({
        COL_TEXT: ["hello", "world", "test"]
    })
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 3
    assert stats["avg_length"] == (5 + 5 + 4) / 3

def test_get_dataset_stats_sft_instruction():
    ds = Dataset.from_dict({
        COL_INSTRUCTION: ["inst1", "inst2"],
        COL_OUTPUT: ["out1", "out2"]
    })
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 2
    # len("inst1") + len("out1") = 5 + 4 = 9
    # len("inst2") + len("out2") = 5 + 4 = 9
    assert stats["avg_length"] == 9.0

def test_get_dataset_stats_dpo():
    ds = Dataset.from_dict({
        COL_PROMPT: ["p1", "p2"],
        COL_CHOSEN: ["c1", "c2"],
        COL_REJECTED: ["r1", "r2"]
    })
    stats = get_dataset_stats(ds, is_dpo=True)
    assert stats["num_examples"] == 2
    # len("p1") + len("c1") + len("r1") = 2 + 2 + 2 = 6
    assert stats["avg_length"] == 6.0

def test_get_dataset_stats_empty():
    ds = Dataset.from_dict({COL_TEXT: []})
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 0
    assert stats["avg_length"] == 0.0

def test_get_dataset_stats_unknown_column():
    ds = Dataset.from_dict({"unknown": ["value1", "value22"]})
    stats = get_dataset_stats(ds, is_dpo=False)
    assert stats["num_examples"] == 2
    assert stats["avg_length"] == (6 + 7) / 2
