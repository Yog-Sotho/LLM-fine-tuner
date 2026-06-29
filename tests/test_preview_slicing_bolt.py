import pytest
from datasets import Dataset
from data.preprocessing import preview_dataset
from config.constants import COL_TEXT, COL_PROMPT, COL_CHOSEN, COL_REJECTED

def test_preview_slicing_sft():
    ds = Dataset.from_dict({COL_TEXT: ["Row " + str(i) for i in range(20)]})
    preview = preview_dataset(ds)
    assert len(preview) == 10
    assert preview[COL_TEXT].tolist() == ["Row " + str(i) for i in range(10)]

def test_preview_slicing_dpo():
    ds = Dataset.from_dict({
        COL_PROMPT: ["P" + str(i) for i in range(10)],
        COL_CHOSEN: ["C" + str(i) for i in range(10)],
        COL_REJECTED: ["R" + str(i) for i in range(10)]
    })
    preview = preview_dataset(ds, is_dpo=True)
    assert len(preview) == 5
    assert preview[COL_PROMPT].tolist() == ["P" + str(i) for i in range(5)]

def test_preview_empty():
    ds = Dataset.from_dict({"text": []})
    preview = preview_dataset(ds)
    assert len(preview) == 1
    assert "empty" in preview["Status"].iloc[0].lower()
