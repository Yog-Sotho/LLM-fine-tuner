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
            10, False, 0, "linear", False, False, False, "", use_flash_attn=flash_attn,
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


# ── Multi-process training and the GPU queue ───────────────────────────────


def test_quantized_models_load_on_the_process_gpu_under_multi_process_runs(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert hardware.quantized_device_map() == "auto" and hardware.is_main_process()
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("RANK", "7")
    assert hardware.quantized_device_map() == {"": 3}
    assert hardware.world_size() == 4 and not hardware.is_main_process()


def test_training_device_args():
    assert hardware.training_device_args("cpu") == {"bf16": False, "fp16": False, "use_cpu": True}
    assert hardware.training_device_args("cuda")["use_cpu"] is False


def test_ui_refuses_to_start_once_per_process(monkeypatch, capsys):
    import main as entry

    monkeypatch.setattr(entry.sys, "argv", ["main.py"])
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit) as exc:
        entry.main()
    assert exc.value.code == 2 and "Multi-GPU runs use the CLI" in capsys.readouterr().err


def test_heavy_gpu_jobs_share_one_queue():
    from ui.app import build_demo

    demo = build_demo()
    gpu = {fn.name: fn.concurrency_limit for fn in demo.fns.values()
           if getattr(fn, "concurrency_id", None) == "gpu"}  # fmt: skip
    assert set(gpu) >= {
        "_on_train_and_chart", "train_grpo", "train_kto", "train_orpo_v27",
        "train_reward_model_v27", "on_evaluate_click", "on_benchmark_click", "on_export_gguf",
        "train_distill", "on_merge_adapters_click",
    }  # fmt: skip
    assert set(gpu.values()) == {1}
    assert "on_stop" not in gpu  # Stop must run while a job holds the queue


@pytest.mark.parametrize(("value", "expected"), [("", 1), ("3", 3), ("0", 1), ("99", 8), ("x", 1)])
def test_gpu_job_concurrency_setting(monkeypatch, value, expected):
    import importlib

    import config.constants as constants

    monkeypatch.setenv("LFT_GPU_JOBS", value)
    try:
        assert importlib.reload(constants).GPU_JOB_CONCURRENCY == expected
    finally:
        monkeypatch.delenv("LFT_GPU_JOBS")
        importlib.reload(constants)
