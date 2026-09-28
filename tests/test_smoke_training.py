"""End-to-end smoke tests: real training steps on a ~1M-parameter model.

These catch library API drift (TRL / Transformers / PEFT / Gradio) that unit tests
with mocks cannot. The model is fetched from the Hugging Face Hub (or the local
cache). Without network access the tests are skipped — unless
REQUIRE_SMOKE_MODELS=1 (set in CI), in which case a missing model is a failure.
"""

import os
import pathlib

import pandas as pd
import pytest
from datasets import Dataset

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


@pytest.fixture(scope="module")
def tiny_model() -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(TINY_MODEL)
        AutoModelForCausalLM.from_pretrained(TINY_MODEL)
    except OSError as exc:
        if os.environ.get("REQUIRE_SMOKE_MODELS") == "1":
            pytest.fail(f"Smoke-test model unavailable: {exc}")
        pytest.skip(f"Smoke-test model unavailable (offline?): {exc}")
    return TINY_MODEL


def _hyperparams() -> dict:
    return {
        "learning_rate": 1e-3,
        "epochs": 1,
        "batch_size": 2,
        "grad_accum": 1,
        "max_length": 64,
        "warmup_steps": 0,
    }


def _train(model: str, dataset: Dataset, output_dir: pathlib.Path, mode: str) -> tuple:
    from training.sft import train_model

    return train_model(
        model,
        dataset,
        str(output_dir),
        _hyperparams(),
        "cpu",
        "LoRA",
        True,
        4,
        8,  # use_lora, lora_rank, lora_alpha
        10,
        64,
        1,
        10,
        16,  # prefix/prompt-tuning/adapter settings (unused for LoRA)
        False,
        0,
        "linear",
        False,  # resume, early_stop, scheduler, grad checkpointing
        False,
        False,
        "You are a helpful assistant.",  # unsloth, chat template, system prompt
        training_mode=mode,
        progress=None,
    )


def _assert_run_config(directory: pathlib.Path, mode: str, model: str) -> dict:
    from core.run_config import load_run_config

    cfg = load_run_config(str(directory / "run_config.yaml"))
    assert cfg["mode"] == mode and cfg["model"] == model
    assert cfg["seed"] == 42 and len(cfg["dataset"]["sha256"]) == 64
    return cfg


def _assert_safetensors_adapter(directory: pathlib.Path) -> None:
    assert (directory / "adapter_config.json").is_file()
    assert (directory / "adapter_model.safetensors").is_file()


def test_ui_builds_and_main_imports():
    import gradio as gr

    import main  # noqa: F401 — module import must not fail
    from ui.app import build_demo, build_theme

    assert isinstance(build_demo(), gr.Blocks)
    assert isinstance(build_theme(), gr.themes.Base)


def test_sft_trains_and_saves_safetensors_adapter(tiny_model, tmp_path):
    ds = Dataset.from_dict(
        {
            "instruction": ["Say hi", "Say bye", "Count to two", "Name a colour"],
            "output": ["Hi", "Bye", "One two", "Blue"],
        }
    )
    summary, records = _train(tiny_model, ds, tmp_path, "sft")
    assert summary.startswith("✅ Training complete")
    _assert_safetensors_adapter(tmp_path)


def test_dpo_trains_and_saves_safetensors_adapter(tiny_model, tmp_path):
    ds = Dataset.from_dict(
        {
            "prompt": ["Greet me", "Say goodbye", "Pick a number", "Name a fruit"],
            "chosen": ["Hello!", "Goodbye!", "Seven", "Apple"],
            "rejected": ["Go away", "Whatever", "Banana", "Seven"],
        }
    )
    summary, _ = _train(tiny_model, ds, tmp_path, "dpo")
    assert summary.startswith("✅ Training complete")
    _assert_safetensors_adapter(tmp_path)


def test_orpo_trains_and_saves_adapter(tiny_model, tmp_path):
    from config.constants import HAS_ORPO
    from training.orpo import train_orpo_v27

    if not HAS_ORPO:
        pytest.skip("ORPO not available in the installed TRL")
    data = tmp_path / "prefs.csv"
    pd.DataFrame(
        {
            "prompt": ["Greet me", "Say goodbye", "Pick a number", "Name a fruit"],
            "chosen": ["Hello!", "Goodbye!", "Seven", "Apple"],
            "rejected": ["Go away", "Whatever", "Banana", "Seven"],
        }
    ).to_csv(data, index=False)

    class _Upload:
        name = str(data)

    out = tmp_path / "orpo"
    result = train_orpo_v27(
        tiny_model,
        _Upload(),
        str(out),
        orpo_epochs=1,
        orpo_batch_size=2,
        progress=None,
    )
    assert result.startswith("✅ ORPO training complete"), result
    _assert_safetensors_adapter(out)
    _assert_run_config(out, "orpo", tiny_model)


def test_inference_with_trained_adapter(tiny_model, tmp_path):
    from inference.generate import generate_text

    ds = Dataset.from_dict({"text": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    _train(tiny_model, ds, tmp_path, "sft")

    reply = generate_text(tiny_model, str(tmp_path), "alpha", max_new_tokens=4)
    assert isinstance(reply, str)
    assert not reply.startswith("❌"), reply


class _Upload:
    def __init__(self, path) -> None:
        self.name = str(path)


PREFS = {
    "prompt": ["Greet me", "Say goodbye", "Pick a number", "Name a fruit"],
    "chosen": ["Hello!", "Goodbye!", "Seven", "Apple"],
    "rejected": ["Go away", "Whatever", "Banana", "Seven"],
}


def test_sft_masks_prompt_and_trains_eos(tiny_model):
    """Completion-only loss: prompt tokens are -100 and the EOS token is a training label."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from data.preprocessing import to_sft_dataset

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    tokenizer.pad_token = tokenizer.eos_token  # same setup as train_model
    ds = to_sft_dataset(
        Dataset.from_dict({"instruction": ["Say hi"], "output": ["Hi there"]}),
        use_chat_template=False,
        system_prompt="",
    )
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(tiny_model),
        args=SFTConfig(output_dir="/tmp/sft_label_check", report_to="none", bf16=False),
        train_dataset=ds,
        processing_class=tokenizer,
    )
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()
    prompt_len = len(tokenizer("### Instruction:\nSay hi\n\n### Response:\n")["input_ids"])
    assert all(label == -100 for label in labels[: prompt_len - 1]), "prompt must not be trained"
    trained = [t for t in labels if t != -100]
    assert trained, "completion must be trained"
    assert trained[-1] == tokenizer.eos_token_id, "EOS must be a training target"
    assert input_ids[-1] == tokenizer.eos_token_id


def test_sft_with_chat_template_trains(tiny_model, tmp_path):
    from training.sft import train_model

    ds = Dataset.from_dict({"instruction": ["Say hi", "Say bye"], "output": ["Hi", "Bye"]})
    summary, _ = train_model(
        tiny_model, ds, str(tmp_path), _hyperparams(), "cpu", "LoRA",
        True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, True, "You are a helpful assistant.",
        training_mode="sft", progress=None,
    )  # fmt: skip
    assert summary.startswith("✅ Training complete")
    _assert_safetensors_adapter(tmp_path)


def test_packing_is_skipped_without_flash_attention(tiny_model, tmp_path):
    """Packing needs Flash Attention 2 on CUDA; otherwise it is skipped with a visible note."""
    from training.sft import train_model

    ds = Dataset.from_dict({"text": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    summary, records = train_model(
        tiny_model, ds, str(tmp_path), {**_hyperparams(), "packing": True}, "cpu", "LoRA",
        True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, False, "", training_mode="sft", progress=None,
    )  # fmt: skip
    assert summary.startswith("✅ Training complete")
    assert any("Packing skipped" in r.get("note", "") for r in records)


@pytest.fixture(scope="module")
def reward_model_dir(tiny_model, tmp_path_factory):
    from training.reward import train_reward_model_v27

    root = tmp_path_factory.mktemp("reward")
    data = root / "prefs.csv"
    pd.DataFrame(PREFS).to_csv(data, index=False)
    out = root / "rm"
    result = train_reward_model_v27(
        tiny_model, _Upload(data), str(out),
        rm_epochs=1, rm_batch_size=2, rm_eval_steps=10, rm_max_length=64, progress=None,
    )  # fmt: skip
    assert result.startswith("✅ Reward model training complete"), result
    _assert_run_config(out, "reward", tiny_model)
    return out


def test_reward_model_is_a_merged_sequence_classifier(reward_model_dir):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    assert (reward_model_dir / "config.json").is_file()
    assert not (reward_model_dir / "adapter_config.json").exists(), "LoRA must be merged"
    model = AutoModelForSequenceClassification.from_pretrained(reward_model_dir, num_labels=1)
    tok = AutoTokenizer.from_pretrained(reward_model_dir)
    with torch.no_grad():
        score = model(**tok("Greet me Hello!", return_tensors="pt")).logits
    assert score.shape == (1, 1)


def test_grpo_with_reference_reward(tiny_model, tmp_path):
    from training.grpo import train_grpo

    data = tmp_path / "prompts.csv"
    pd.DataFrame({"prompt": ["2+2=", "3+3="], "reference": ["4", "6"]}).to_csv(data, index=False)
    out = tmp_path / "grpo"
    result = train_grpo(
        tiny_model, "", _Upload(data), str(out),
        num_generations=2, max_completion_length=8, progress=None,
    )  # fmt: skip
    assert result.startswith("✅ GRPO training complete"), result
    assert "reference match" in result
    _assert_safetensors_adapter(out)
    cfg = _assert_run_config(out, "grpo", tiny_model)
    assert cfg["reward"] == "reference match" and cfg["hyperparams"]["num_generations"] == 2


def test_grpo_with_trained_reward_model(tiny_model, reward_model_dir, tmp_path):
    from training.grpo import train_grpo

    data = tmp_path / "prompts.csv"
    pd.DataFrame({"prompt": ["Greet me", "Name a fruit"]}).to_csv(data, index=False)
    out = tmp_path / "grpo_rm"
    result = train_grpo(
        tiny_model, str(reward_model_dir), _Upload(data), str(out),
        num_generations=2, max_completion_length=8, progress=None,
    )  # fmt: skip
    assert result.startswith("✅ GRPO training complete"), result
    assert "reward model" in result
    _assert_safetensors_adapter(out)


@pytest.mark.parametrize("fmt", ["paired", "unpaired"])
def test_kto_trains(tiny_model, tmp_path, fmt):
    from training.kto import train_kto

    data = tmp_path / "kto.csv"
    if fmt == "paired":
        pd.DataFrame(PREFS).to_csv(data, index=False)
    else:
        pd.DataFrame(
            {
                "prompt": ["Greet me", "Greet me", "Name a fruit", "Name a fruit"],
                "completion": ["Hello!", "Go away", "Apple", "Seven"],
                "label": ["true", "false", "true", "false"],
            }
        ).to_csv(data, index=False)
    out = tmp_path / "kto"
    result = train_kto(
        tiny_model, _Upload(data), str(out), batch_size=2, max_length=64, progress=None
    )
    assert result.startswith("✅ KTO training complete"), result
    _assert_safetensors_adapter(out)
    rows = _assert_run_config(out, "kto", tiny_model)["dataset"]["rows"]
    assert rows == (8 if fmt == "paired" else 4)  # pairs become two rows each


def test_cli_train_runs_end_to_end(tiny_model, tmp_path):
    from typer.testing import CliRunner

    from cli.commands import app

    data = tmp_path / "train.csv"
    pd.DataFrame({"text": ["one two", "three four", "five six", "seven eight"]}).to_csv(
        data, index=False
    )
    out = tmp_path / "cli_out"
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--model",
            tiny_model,
            "--data",
            str(data),
            "--output",
            str(out),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--max-length",
            "32",
            "--lora-rank",
            "4",
        ],
    )
    assert result.exit_code == 0, result.output
    _assert_safetensors_adapter(out)


# ── Reproducibility ────────────────────────────────────────────────────────


def _adapter_tensors(directory: pathlib.Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(directory / "adapter_model.safetensors"))


def _train_seeded(model: str, out: pathlib.Path, seed: int) -> None:
    from training.sft import train_model

    ds = Dataset.from_dict(
        {"instruction": ["Say hi", "Say bye", "Count"], "output": ["Hi", "Bye", "1 2"]}
    )
    train_model(
        model, ds, str(out), _hyperparams(), "cpu", "LoRA",
        True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, False, "", training_mode="sft", progress=None, seed=seed,
    )  # fmt: skip


def test_same_seed_gives_identical_weights_and_different_seed_does_not(tiny_model, tmp_path):
    import torch

    _train_seeded(tiny_model, tmp_path / "a", seed=123)
    _train_seeded(tiny_model, tmp_path / "b", seed=123)
    _train_seeded(tiny_model, tmp_path / "c", seed=7)
    a, b, c = (_adapter_tensors(tmp_path / n) for n in "abc")
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a), "same seed must reproduce the weights"
    assert any(not torch.equal(a[k], c[k]) for k in a), "a different seed must change them"
    from core.run_config import load_run_config

    assert load_run_config(str(tmp_path / "a" / "run_config.yaml"))["seed"] == 123


def test_cli_replay_reproduces_run_and_warns_on_different_data(tiny_model, tmp_path):
    from typer.testing import CliRunner

    from cli.commands import app
    from core.run_config import load_run_config

    data = tmp_path / "train.csv"
    pd.DataFrame({"text": ["one two", "three four", "five six", "seven eight"]}).to_csv(
        data, index=False
    )
    first = tmp_path / "first"
    base = ["--epochs", "1", "--batch-size", "2", "--max-length", "32", "--lora-rank", "4"]
    result = CliRunner().invoke(
        app,
        [
            "train",
            "--model",
            tiny_model,
            "--data",
            str(data),
            "--output",
            str(first),
            "--seed",
            "5",
            *base,
        ],
    )
    assert result.exit_code == 0, result.output

    replay = tmp_path / "replay"
    cfg_path = first / "run_config.yaml"
    result = CliRunner().invoke(
        app, ["train", "--config", str(cfg_path), "--data", str(data), "--output", str(replay)]
    )
    assert result.exit_code == 0, result.output
    assert "differs" not in result.output
    original, replayed = (
        load_run_config(str(cfg_path)),
        load_run_config(str(replay / "run_config.yaml")),
    )
    for key in ("mode", "model", "seed", "hyperparams", "peft", "dataset", "early_stop"):
        assert replayed[key] == original[key], key
    assert _adapter_tensors(first).keys() == _adapter_tensors(replay).keys()

    other = tmp_path / "other.csv"
    pd.DataFrame({"text": ["different", "rows", "entirely", "here"]}).to_csv(other, index=False)
    result = CliRunner().invoke(
        app,
        ["train", "--config", str(cfg_path), "--data", str(other), "--output", str(tmp_path / "x")],
    )
    assert result.exit_code == 0, result.output
    assert "Dataset differs" in result.output


def test_cli_replay_rejects_non_sft_config(tmp_path):
    from typer.testing import CliRunner

    from cli.commands import app

    cfg = tmp_path / "run_config.yaml"
    cfg.write_text("mode: kto\nmodel: m\n")
    data = tmp_path / "d.csv"
    data.write_text("text\nhi\n")
    result = CliRunner().invoke(app, ["train", "--config", str(cfg), "--data", str(data)])
    assert result.exit_code == 1
    assert "replays sft/dpo runs" in result.output


def _ui_train(model: str, data: pathlib.Path, run_name: str, resume: bool) -> str:
    from ui.handlers import on_train_click

    class _File:
        name = str(data)

    msg, _zip, _dir, _records = on_train_click(
        _File(), "gpt2", model, "Advanced", "LoRA", True, 4, 8, 30, 512, 2, 20, 16,
        1e-3, 1, 2, 1, 64, 0, 0, "linear", False, resume, None, None, None,
        False, False, "", "SFT (Supervised Fine-Tuning)", 0.1, False,
        run_name=run_name, progress=None,
    )  # fmt: skip
    return msg


def test_ui_runs_persist_and_resume(tiny_model, tmp_path, monkeypatch):
    import core.run_config as rc
    import ui.handlers as handlers

    runs = tmp_path / "runs"
    monkeypatch.setattr(rc, "RUNS_DIR", str(runs))
    monkeypatch.setattr(handlers, "RUNS_DIR", str(runs))
    data = tmp_path / "sft.csv"
    pd.DataFrame({"instruction": ["Say hi", "Say bye"], "output": ["Hi", "Bye"]}).to_csv(
        data, index=False
    )

    assert _ui_train(tiny_model, data, "exp1", resume=False).startswith("✅")
    assert (runs / "exp1" / "run_config.yaml").is_file()
    assert "already exists" in _ui_train(tiny_model, data, "exp1", resume=False)
    assert (runs / "exp1" / "adapter_model.safetensors").is_file(), "existing run must be kept"
    assert _ui_train(tiny_model, data, "exp1", resume=True).startswith("✅")
    assert "Nothing to resume" in _ui_train(tiny_model, data, "missing", resume=True)
    assert "Run name" in _ui_train(tiny_model, data, "../escape", resume=False)
    assert not (tmp_path / "escape").exists()
