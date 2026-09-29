"""GPU-only paths: 4-bit QLoRA, full fine-tuning in half precision, Flash Attention 2 +
packing, vLLM generation for GRPO, and NCCL data-parallel training.

Skipped without CUDA (the CPU CI jobs). Run them on a GPU machine with
``pytest -m gpu`` — the GPU workflow (.github/workflows/gpu.yml) does this on a
GitHub GPU runner (Tesla T4: fp16 only, no Flash Attention 2) or a self-hosted one.
Each test also checks the capability it needs (bf16, compute capability 8.0+,
vLLM, two GPUs) and skips with the reason when it is missing.
"""

import importlib.util
import json
import math
import os
import subprocess
import sys

import pytest
import torch
from datasets import Dataset

from tests.test_smoke_training import (
    _DDP_PROBE,
    TINY_CHAT_MODEL,
    TINY_MODEL,
    _assert_run_config,
    _assert_safetensors_adapter,
    _cached_model,
    _hyperparams,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

SFT_ROWS = Dataset.from_dict({"instruction": [f"Say {i}" for i in range(12)],
                              "output": [f"Word number {i}" for i in range(12)]})  # fmt: skip


@pytest.fixture(scope="module")
def tiny_model() -> str:
    return _cached_model(TINY_MODEL)


def _train(model, out, peft_method="LoRA", mode="sft", data=SFT_ROWS, **kwargs):
    from training.sft import train_model

    return train_model(
        model, data, str(out), {**_hyperparams(), **kwargs.pop("hp", {})}, "cuda", peft_method,
        True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False, False, False, "",
        training_mode=mode, progress=None, **kwargs,
    )  # fmt: skip


def _final_loss(records) -> float:
    from core.callbacks import final_train_loss

    loss = final_train_loss(records)
    assert loss != "N/A", records
    return float(loss)


def test_qlora_trains_a_4bit_model(tiny_model, tmp_path, monkeypatch):
    import training.sft as sft

    seen = {}
    original = sft.AutoModelForCausalLM.from_pretrained

    def spy(name, **kwargs):
        seen.update(kwargs)
        return original(name, **kwargs)

    monkeypatch.setattr(sft.AutoModelForCausalLM, "from_pretrained", spy)
    summary, records = _train(tiny_model, tmp_path)
    assert seen["quantization_config"].load_in_4bit
    assert math.isfinite(_final_loss(records)), summary
    _assert_safetensors_adapter(tmp_path)
    _assert_run_config(tmp_path, "sft", tiny_model)


def test_full_finetuning_in_half_precision(tiny_model, tmp_path):
    summary, records = _train(tiny_model, tmp_path, peft_method="Full Fine-tuning")
    assert math.isfinite(_final_loss(records)), summary
    assert (tmp_path / "model.safetensors").is_file()
    assert not (tmp_path / "adapter_config.json").exists()


def test_dpo_on_gpu(tiny_model, tmp_path):
    prefs = Dataset.from_dict({"prompt": [f"Question {i}?" for i in range(8)],
                               "chosen": [f"Good answer {i}" for i in range(8)],
                               "rejected": [f"Bad {i}" for i in range(8)]})  # fmt: skip
    summary, records = _train(tiny_model, tmp_path, mode="dpo", data=prefs)
    assert math.isfinite(_final_loss(records)), summary
    _assert_safetensors_adapter(tmp_path)


def test_inference_with_a_gpu_trained_adapter(tiny_model, tmp_path):
    from inference.generate import generate_text

    _train(tiny_model, tmp_path)
    reply = generate_text(tiny_model, str(tmp_path), "Say 3", 8, 0.7, 0.9)
    assert isinstance(reply, str) and not reply.startswith("❌"), reply


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="Flash Attention 2 needs compute capability 8.0+ (Ampere or newer)",
)
@pytest.mark.skipif(
    importlib.util.find_spec("flash_attn") is None, reason="flash-attn not installed"
)
def test_flash_attention_with_packing(tiny_model, tmp_path):
    summary, records = _train(tiny_model, tmp_path, use_flash_attn=True, hp={"packing": True})
    assert not any("Packing skipped" in r.get("note", "") for r in records), records
    assert math.isfinite(_final_loss(records)), summary


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason="vLLM not installed")
def test_grpo_generates_with_vllm(tmp_path):
    from training.grpo import train_grpo

    model = _cached_model(TINY_CHAT_MODEL)
    data = tmp_path / "prompts.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"What is {i}+{i}?", "reference": str(2 * i)})
                            + "\n" for i in range(4)))  # fmt: skip

    class File:
        name = str(data)

    status = train_grpo(model, "", File(), str(tmp_path / "out"), num_generations=2,
                        max_completion_length=16, use_vllm=True, progress=None)  # fmt: skip
    assert status.startswith("✅"), status
    _assert_safetensors_adapter(tmp_path / "out")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2, reason="needs two GPUs"
)
def test_two_gpu_training_is_data_parallel(tiny_model, tmp_path):
    report, out = tmp_path / "report", tmp_path / "out"
    report.mkdir()
    script = tmp_path / "probe.py"
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    probe = _DDP_PROBE.format(repo=repo, report=str(report), out=str(out), model=tiny_model)
    script.write_text(probe.replace('hp, "cpu", "LoRA"', 'hp, "cuda", "LoRA"'))
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(script)],
        capture_output=True, text=True, timeout=900, env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-3000:]
    ranks = [json.loads((report / f"rank{r}.json").read_text()) for r in (0, 1)]
    assert [r["world_size"] for r in ranks] == [2, 2]
    assert ranks[0]["weights"] == ranks[1]["weights"]
    _assert_safetensors_adapter(out)
