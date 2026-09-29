"""Scale features that CPU tests can check: sharding detection and the settings the app
passes for FSDP / DeepSpeed, long-context options, accelerate presets, MoE helpers.

Real sharded / long-context training runs on GPUs in tests/test_gpu.py.
"""

import glob
from unittest.mock import MagicMock

import pytest
import torch
import yaml
from datasets import Dataset

import core.hardware as hardware
import training.sft as sft

# ── Sharding detection and refusals ────────────────────────────────────────


@pytest.fixture
def no_sharding(monkeypatch):
    monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
    # accelerate only shards on GPUs; pretend one exists for the sharding tests.
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)


def test_no_sharding_without_a_gpu(monkeypatch):
    """accelerate launch sets ACCELERATE_USE_FSDP on CPU too, but only shards on GPUs."""
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: False)
    assert hardware.sharding_backend() is None


@pytest.mark.parametrize(
    ("env", "backend"),
    [({}, None), ({"ACCELERATE_USE_FSDP": "true"}, "fsdp"),
     ({"ACCELERATE_USE_DEEPSPEED": "TRUE"}, "deepspeed"), ({"ACCELERATE_USE_FSDP": "false"}, None)],
)  # fmt: skip
def test_sharding_backend_follows_accelerate_launch(no_sharding, monkeypatch, env, backend):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert hardware.sharding_backend() == backend
    assert (hardware.sharding_unsupported("KTO") is None) == (backend is None)


class _File:
    name = "data.jsonl"


@pytest.mark.parametrize(
    "call",
    [
        lambda: __import__("training.kto", fromlist=["x"]).train_kto("m", _File(), "o", progress=None),
        lambda: __import__("training.grpo", fromlist=["x"]).train_grpo("m", "", _File(), "o",
                                                                       progress=None),
        lambda: __import__("training.orpo", fromlist=["x"]).train_orpo_v27("m", _File(), "o",
                                                                           progress=None),
        lambda: __import__("training.reward", fromlist=["x"]).train_reward_model_v27(
            "m", _File(), "o", progress=None),
        lambda: __import__("training.distill", fromlist=["x"]).train_distill("s", "t", _File(), "o",
                                                                             progress=None),
    ],
)  # fmt: skip
def test_other_trainers_refuse_sharding(no_sharding, monkeypatch, call):
    monkeypatch.setenv("ACCELERATE_USE_DEEPSPEED", "true")
    status = call()
    assert status.startswith("❌") and "not with DEEPSPEED sharding" in status
    assert "configs/accelerate/multi_gpu.yaml" in status


@pytest.mark.parametrize(("bf16", "dtype"), [(True, torch.bfloat16), (False, torch.float32)])
def test_sharded_quant_storage_matches_the_training_precision(monkeypatch, bf16, dtype):
    monkeypatch.setattr(
        hardware, "select_precision", lambda device: {"bf16": bf16, "fp16": not bf16}
    )
    assert hardware.sharded_quant_storage("cuda") == dtype


# ── What train_model loads under sharding (simulated GPU) ──────────────────


class _Loaded(Exception):
    pass


@pytest.fixture
def cuda_load(monkeypatch, no_sharding):
    calls: list[dict] = []

    def fake_from_pretrained(name, **kwargs):
        calls.append(kwargs)
        raise _Loaded

    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(hardware.torch.cuda, "is_bf16_supported", lambda *a, **k: True)
    monkeypatch.setattr(sft.AutoTokenizer, "from_pretrained", lambda *a, **k: MagicMock())
    monkeypatch.setattr(sft.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    return calls


def _load(peft_method="LoRA", flash_attn=False, **kwargs):
    ds = Dataset.from_dict({"text": ["a", "b"]})
    hyper = {"learning_rate": 1e-3, "epochs": 1, "batch_size": 1, "grad_accum": 1,
             "max_length": 32, "warmup_steps": 0}  # fmt: skip
    with pytest.raises(RuntimeError, match="Training failed"):
        sft.train_model("some/model", ds, "out", hyper, "cuda", peft_method, True, 4, 8, 10, 64, 1,
                        10, False, 0, "linear", False, False, False, "", use_flash_attn=flash_attn,
                        **kwargs)  # fmt: skip


@pytest.mark.parametrize("peft_method", ["LoRA", "QLoRA Enhanced"])
def test_sharded_4bit_loads_without_device_map_and_matching_storage(cuda_load, monkeypatch,
                                                                     peft_method):  # fmt: skip
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    _load(peft_method)
    (kwargs,) = cuda_load
    assert "device_map" not in kwargs  # FSDP places the shards
    storage = kwargs["quantization_config"].bnb_4bit_quant_storage
    assert storage == kwargs["torch_dtype"] == torch.bfloat16


def test_unsharded_4bit_keeps_the_per_process_device_map(cuda_load):
    _load("LoRA")
    (kwargs,) = cuda_load
    assert kwargs["device_map"] == "auto" and kwargs["torch_dtype"] == torch.bfloat16


def test_sharded_vision_data_is_refused(no_sharding, monkeypatch):
    monkeypatch.setenv("ACCELERATE_USE_FSDP", "true")
    ds = Dataset.from_dict({"messages": ["x"], "images": [["y"]]})
    with pytest.raises(RuntimeError, match="Vision fine-tuning runs data-parallel only"):
        sft.train_model("m", ds, "out", {"max_length": 32}, "cpu", "LoRA", True, 4, 8, 10, 64, 1,
                        10, False, 0, "linear", False, False, False, "")  # fmt: skip


# ── Long-context options ───────────────────────────────────────────────────


class _Log:
    def __init__(self):
        self.records = []


@pytest.mark.parametrize(
    ("hp", "device", "fa2", "args", "notes"),
    [
        ({}, "cuda", True, {}, []),
        ({"activation_offloading": True, "padding_free": True}, "cuda", True,
         {"activation_offloading": True, "padding_free": True}, []),
        ({"activation_offloading": True, "padding_free": True}, "cuda", False,
         {"activation_offloading": True}, ["Padding-free skipped"]),
        ({"activation_offloading": True, "padding_free": True}, "cpu", False, {},
         ["Activation offloading skipped", "Padding-free skipped"]),
    ],
)  # fmt: skip
def test_long_context_options_apply_only_where_they_work(hp, device, fa2, args, notes):
    log = _Log()
    assert sft._long_context_args(hp, device, fa2, log) == args
    assert [r["note"].split(":")[0].lstrip("⚠️ ") for r in log.records] == notes


def test_long_context_options_reach_the_sft_config(monkeypatch, tmp_path):
    """Whatever _long_context_args decides is passed to SFTConfig unchanged."""
    import trl

    seen = {}
    original = trl.SFTConfig

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(trl, "SFTConfig", spy)
    monkeypatch.setattr(sft, "_long_context_args", lambda hp, d, fa2, log: {"padding_free": True})
    monkeypatch.setattr(trl, "SFTTrainer", MagicMock(side_effect=_Loaded))
    monkeypatch.setattr(sft.AutoTokenizer, "from_pretrained",
                        lambda *a, **k: MagicMock(chat_template=None, eos_token="</s>"))  # fmt: skip
    model = MagicMock()
    model.config._attn_implementation = "sdpa"
    monkeypatch.setattr(sft.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    monkeypatch.setattr(sft, "get_peft_model", lambda m, c: m)
    monkeypatch.setattr(sft, "setup_moe", lambda m, **kwargs: None)
    monkeypatch.setattr(sft, "token_length_report", lambda *a: {})
    monkeypatch.setattr(sft, "drop_prompts_over_limit", lambda ds, *a: (ds, 0))
    with pytest.raises(RuntimeError):
        _load_cpu(tmp_path)
    assert seen.get("padding_free") is True


def _load_cpu(tmp_path):
    ds = Dataset.from_dict({"text": ["alpha beta", "gamma delta"]})
    hyper = {"learning_rate": 1e-3, "epochs": 1, "batch_size": 1, "grad_accum": 1,
             "max_length": 32, "warmup_steps": 0, "eval_split": 0.0}  # fmt: skip
    sft.train_model("m", ds, str(tmp_path), hyper, "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10,
                    False, 0, "linear", False, False, False, "", progress=None)  # fmt: skip


# ── Accelerate presets ─────────────────────────────────────────────────────


def test_accelerate_presets():
    from accelerate.commands.config.config_args import load_config_from_file

    presets = {p.rsplit("/", 1)[-1]: p for p in glob.glob("configs/accelerate/*.yaml")}
    assert set(presets) == {"multi_gpu.yaml", "fsdp2.yaml", "fsdp_qlora.yaml",
                            "deepspeed_zero2.yaml", "deepspeed_zero3.yaml"}  # fmt: skip
    for path in presets.values():
        load_config_from_file(path)  # accelerate itself accepts it
        config = yaml.safe_load(open(path))
        # The app picks bf16 / fp16 / fp32; an accelerate default would override fp32 runs.
        assert config["mixed_precision"] == "no" and config["use_cpu"] is False
    fsdp2 = yaml.safe_load(open(presets["fsdp2.yaml"]))["fsdp_config"]
    assert fsdp2["fsdp_version"] == 2 and fsdp2["fsdp_cpu_ram_efficient_loading"] is True
    qlora = yaml.safe_load(open(presets["fsdp_qlora.yaml"]))["fsdp_config"]
    assert qlora["fsdp_use_orig_params"] is False and qlora["fsdp_offload_params"] is True
    zero3 = yaml.safe_load(open(presets["deepspeed_zero3.yaml"]))["deepspeed_config"]
    assert zero3["zero_stage"] == 3 and zero3["zero3_save_16bit_model"] is True


# ── MoE helpers ────────────────────────────────────────────────────────────


def _config(**fields):
    config = MagicMock(spec=["get_text_config", *fields])
    for key, value in fields.items():
        setattr(config, key, value)
    config.get_text_config.return_value = config
    return MagicMock(config=config)


@pytest.mark.parametrize(
    ("fields", "experts"),
    [({"num_experts": 8}, 8), ({"num_local_experts": 4}, 4), ({"n_routed_experts": 64}, 64),
     ({"num_experts": 1}, 0), ({}, 0)],
)  # fmt: skip
def test_expert_count_and_dropout(fields, experts):
    model = _config(**fields)
    assert hardware.moe_expert_count(model) == experts
    assert hardware.lora_dropout(model) == (0.0 if experts else 0.05)
