"""Characterization: the exact from_pretrained arguments of every GPU loading path.

GPU paths can't run on CPU CI, so each one is run on a simulated CUDA device and its
load arguments (dtype, device map, Flash Attention, 4-bit config) are compared with
tests/fixtures/loading_kwargs.json — recorded before refactoring the loaders. Any change
to how a model is loaded on a GPU fails here. Regenerate deliberately with
``LFT_UPDATE_SNAPSHOT=1 pytest tests/test_loading_snapshot.py``.
"""

import json
import os
import pathlib
from unittest.mock import MagicMock

import transformers
from datasets import Dataset

import core.hardware as hardware
import training.orpo as orpo
import training.sft as sft
import training.vision as vision

SNAPSHOT = pathlib.Path(__file__).parent / "fixtures" / "loading_kwargs.json"
QUANT_FIELDS = ("load_in_4bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype",
                "bnb_4bit_use_double_quant", "bnb_4bit_quant_storage")  # fmt: skip
HYPER = {"learning_rate": 1e-3, "epochs": 1, "batch_size": 1, "grad_accum": 1,
         "max_length": 32, "warmup_steps": 0}  # fmt: skip


class _Loaded(Exception):
    pass


def _normalise(kwargs: dict) -> dict:
    out = {}
    for key, value in sorted(kwargs.items()):
        if key == "quantization_config":
            config = value.to_dict()
            out[key] = {f: str(config.get(f)) for f in QUANT_FIELDS}
        else:
            out[key] = str(value)
    return out


def _sft(peft, flash):
    return lambda: sft.train_model(
        "some/model", Dataset.from_dict({"text": ["a", "b"]}), "out", HYPER, "cuda", peft, True,
        4, 8, 10, 64, 1, 10, False, 0, "linear", False, False, False, "", use_flash_attn=flash,
    )  # fmt: skip


def _cases(tmp_path):
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("prompt,chosen,rejected\n" + "".join(f"q{i},good,bad\n" for i in range(4)))
    images = Dataset.from_dict({"images": [[{"bytes": b"x", "path": None}]] * 4,
                                "messages": ["[]"] * 4})  # fmt: skip

    class File:
        name = str(pairs)

    for bf16 in (True, False):
        for peft in ("LoRA", "QLoRA Enhanced", "Full Fine-tuning"):
            for flash in (False, True):
                yield f"sft|{peft}|bf16={bf16}|fa2={flash}", bf16, None, _sft(peft, flash)
                yield f"sft|{peft}|bf16={bf16}|fa2={flash}|fsdp", bf16, "true", _sft(peft, flash)
        yield (
            f"orpo|bf16={bf16}",
            bf16,
            None,
            (lambda: orpo.train_orpo_v27("some/model", File(), "out", progress=None)),
        )
        for peft in ("LoRA", "Full Fine-tuning"):
            yield f"vision|{peft}|bf16={bf16}", bf16, None, (
                lambda peft=peft: vision.train_vision_sft(
                    "some/model", images, "out", HYPER, "cuda", peft, True, 4, 8, "LoRA",
                    False, "linear", 0, False, 42, "none", None, MagicMock(), None,
                )
            )  # fmt: skip


def test_gpu_load_arguments_match_the_snapshot(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_from_pretrained(name, **kwargs):
        calls.append(kwargs)
        raise _Loaded

    monkeypatch.setattr(hardware.torch.cuda, "is_available", lambda: True)
    for module in (sft, orpo):
        monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *a, **k: MagicMock())
        monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained",
                        fake_from_pretrained)  # fmt: skip
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *a, **k: MagicMock())

    seen = {}
    for name, bf16, fsdp, run in _cases(tmp_path):
        monkeypatch.setattr(hardware.torch.cuda, "is_bf16_supported", lambda *a, b=bf16, **k: b)
        monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising=False)
        if fsdp:
            monkeypatch.setenv("ACCELERATE_USE_FSDP", fsdp)
        else:
            monkeypatch.delenv("ACCELERATE_USE_FSDP", raising=False)
        calls.clear()
        try:  # the fake load ends every run (some trainers report it as a ❌ status)
            run()
        except Exception:
            pass
        assert calls, f"{name}: the model was never loaded"
        seen[name] = _normalise(calls[0])

    if os.environ.get("LFT_UPDATE_SNAPSHOT") == "1":
        SNAPSHOT.write_text(json.dumps(seen, indent=1, sort_keys=True) + "\n")
    expected = json.loads(SNAPSHOT.read_text())
    assert sorted(seen) == sorted(expected)
    for name in expected:
        assert seen[name] == expected[name], name
