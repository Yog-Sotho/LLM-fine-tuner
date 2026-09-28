"""How train_model loads the model on a (simulated) CUDA GPU — no download, no GPU needed."""

from unittest.mock import MagicMock

import pytest
import torch
from datasets import Dataset

import core.hardware as hardware
import training.sft as sft


class _Loaded(Exception):
    """Raised by the fake from_pretrained to stop train_model after loading."""


@pytest.fixture
def cuda_load(monkeypatch):
    """Pretend a CUDA GPU exists and record from_pretrained kwargs instead of loading."""
    calls: list[dict] = []

    def fake_from_pretrained(name, **kwargs):
        calls.append(kwargs)
        raise _Loaded

    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sft.AutoTokenizer, "from_pretrained", lambda *a, **k: MagicMock())
    monkeypatch.setattr(sft.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    return calls


def _load(peft_method: str, use_lora: bool = True, flash_attn: bool = False) -> None:
    ds = Dataset.from_dict({"text": ["a", "b"]})
    hyper = {"learning_rate": 1e-3, "epochs": 1, "batch_size": 1, "grad_accum": 1,
             "max_length": 32, "warmup_steps": 0}  # fmt: skip
    with pytest.raises(RuntimeError, match="Training failed"):
        sft.train_model(
            "some/model", ds, "out", hyper, "cuda", peft_method, use_lora, 4, 8, 10, 64, 1,
            10, 16, False, 0, "linear", False, False, False, "", use_flash_attn=flash_attn,
        )  # fmt: skip


@pytest.mark.parametrize(("peft_method", "use_lora"), [("Full Fine-tuning", True), ("Auto", False)])
@pytest.mark.parametrize(("bf16", "dtype"), [(True, torch.bfloat16), (False, torch.float32)])
def test_full_finetuning_loads_unquantised(cuda_load, monkeypatch, peft_method, use_lora,
                                            bf16, dtype):  # fmt: skip
    # Transformers refuses to train a quantised model without adapters; fp16 mixed
    # precision needs fp32 master weights, so the weights are bf16 or fp32, never fp16.
    monkeypatch.setattr(hardware.torch.cuda, "is_bf16_supported", lambda *a, **k: bf16)
    _load(peft_method, use_lora, flash_attn=True)
    (kwargs,) = cuda_load
    assert "quantization_config" not in kwargs
    assert kwargs["torch_dtype"] == dtype
    # Flash Attention 2 only runs in fp16/bf16.
    assert (kwargs.get("attn_implementation") == "flash_attention_2") == bf16


def test_lora_still_loads_in_4bit(cuda_load, monkeypatch):
    monkeypatch.setattr(hardware.torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    _load("LoRA")
    (kwargs,) = cuda_load
    assert kwargs["quantization_config"].load_in_4bit


# ── LoRA targets and variants ──────────────────────────────────────────────


def test_lora_targets_every_linear_layer():
    assert hardware.get_lora_targets() == "all-linear"


@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [("LoRA", {}), ("rsLoRA", {"use_rslora": True}), ("DoRA", {"use_dora": True})],
)
def test_lora_variant_kwargs(variant, kwargs):
    assert hardware.lora_variant_kwargs(variant) == kwargs


def test_unknown_lora_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown LoRA variant"):
        hardware.lora_variant_kwargs("PiSSA")
