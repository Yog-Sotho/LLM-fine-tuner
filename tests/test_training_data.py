"""Unit tests (no model downloads) for trainer data preparation and helpers."""

import pytest
from datasets import Dataset

from core.hardware import compute_dtype, select_precision
from data.preprocessing import to_sft_dataset
from training.grpo import reference_match_reward
from training.kto import to_kto_dataset


def test_sft_instruction_rows_become_prompt_completion():
    ds = Dataset.from_dict({"instruction": ["Say hi"], "output": ["Hi"]})
    out = to_sft_dataset(ds, use_chat_template=False, system_prompt="ignored")
    assert out.column_names == ["prompt", "completion"]
    assert out[0]["prompt"] == "### Instruction:\nSay hi\n\n### Response:\n"
    assert out[0]["completion"] == "Hi"


def test_sft_chat_template_rows_are_conversational():
    ds = Dataset.from_dict({"instruction": ["Say hi"], "output": ["Hi"]})
    out = to_sft_dataset(ds, use_chat_template=True, system_prompt="Be nice.")
    assert out[0]["prompt"] == [
        {"role": "system", "content": "Be nice."},
        {"role": "user", "content": "Say hi"},
    ]
    assert out[0]["completion"] == [{"role": "assistant", "content": "Hi"}]


def test_sft_chat_template_without_system_prompt_omits_system_turn():
    ds = Dataset.from_dict({"instruction": ["Say hi"], "output": ["Hi"]})
    out = to_sft_dataset(ds, use_chat_template=True, system_prompt="")
    assert out[0]["prompt"] == [{"role": "user", "content": "Say hi"}]


def test_sft_text_rows_stay_language_modelling():
    ds = Dataset.from_dict({"text": ["hello world"], "extra": [1]})
    assert to_sft_dataset(ds, use_chat_template=True, system_prompt="").column_names == ["text"]


def test_sft_rejects_unknown_columns():
    with pytest.raises(ValueError, match="SFT needs"):
        to_sft_dataset(Dataset.from_dict({"foo": ["x"]}), use_chat_template=False, system_prompt="")


def test_kto_paired_rows_are_split_into_desirable_and_undesirable():
    ds = Dataset.from_dict({"prompt": ["p"], "chosen": ["good"], "rejected": ["bad"]})
    out = to_kto_dataset(ds)
    assert out["completion"] == ["good", "bad"]
    assert out["label"] == [True, False]
    assert out["prompt"] == ["p", "p"]


@pytest.mark.parametrize(
    ("raw", "expected"), [("true", True), ("0", False), ("Yes", True), ("bad", False)]
)
def test_kto_labels_are_parsed(raw, expected):
    ds = Dataset.from_dict({"prompt": ["p"], "completion": ["c"], "label": [raw]})
    assert to_kto_dataset(ds)["label"] == [expected]


def test_kto_rejects_unknown_label():
    ds = Dataset.from_dict({"prompt": ["p"], "completion": ["c"], "label": ["maybe"]})
    with pytest.raises(ValueError, match="Unrecognised KTO label"):
        to_kto_dataset(ds)


def test_kto_drops_empty_rows():
    ds = Dataset.from_dict({"prompt": ["p", " "], "completion": ["c", "d"], "label": [1, 0]})
    assert len(to_kto_dataset(ds)) == 1


def test_reference_match_reward_handles_text_and_messages():
    completions = ["The answer is 4.", [{"role": "assistant", "content": "It is SIX"}], "no idea"]
    assert reference_match_reward(completions, reference=["4", "six", "7"]) == [1.0, 1.0, 0.0]


def test_reference_match_reward_ignores_blank_reference():
    assert reference_match_reward(["anything"], reference=[" "]) == [0.0]


def test_precision_on_cpu_is_full_fp32(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_precision("cpu") == {"bf16": False, "fp16": False}
    assert compute_dtype("cpu") == torch.float32


@pytest.mark.parametrize(
    ("bf16_ok", "expected", "dtype_name"),
    [
        (True, {"bf16": True, "fp16": False}, "bfloat16"),
        (False, {"bf16": False, "fp16": True}, "float16"),
    ],
)
def test_precision_on_gpu(monkeypatch, bf16_ok, expected, dtype_name):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: bf16_ok)
    assert select_precision("cuda") == expected
    assert compute_dtype("cuda") == getattr(torch, dtype_name)
