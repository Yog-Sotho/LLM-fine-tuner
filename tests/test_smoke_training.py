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


# Real tokenizer and chat template with tool calling and <think> reasoning.
TINY_CHAT_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"
TINY_VLM = "trl-internal-testing/tiny-Qwen2_5_VLForConditionalGeneration"


def _cached_model(name: str) -> str:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(name)
        AutoModelForCausalLM.from_pretrained(name)
    except OSError as exc:
        if os.environ.get("REQUIRE_SMOKE_MODELS") == "1":
            pytest.fail(f"Smoke-test model unavailable: {exc}")
        pytest.skip(f"Smoke-test model unavailable (offline?): {exc}")
    return name


@pytest.fixture(scope="module")
def tiny_model() -> str:
    return _cached_model(TINY_MODEL)


@pytest.fixture(scope="module")
def tiny_chat_model() -> str:
    return _cached_model(TINY_CHAT_MODEL)


@pytest.fixture(scope="module")
def tiny_vlm() -> str:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    try:
        AutoProcessor.from_pretrained(TINY_VLM)
        AutoModelForImageTextToText.from_pretrained(TINY_VLM)
    except (OSError, ImportError) as exc:  # ImportError: torchvision missing
        if os.environ.get("REQUIRE_SMOKE_MODELS") == "1":
            pytest.fail(f"Vision smoke-test model unavailable: {exc}")
        pytest.skip(f"Vision smoke-test model unavailable: {exc}")
    return TINY_VLM


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
    # Every trainer writes our model card last (after TRL/PEFT wrote theirs).
    from huggingface_hub import ModelCard

    card = ModelCard.load(str(directory / "README.md")).data
    assert card.base_model == model and mode in card.tags and "llm-fine-tuner" in card.tags
    adapter = (directory / "adapter_config.json").is_file()
    assert card.library_name == ("peft" if adapter else "transformers")
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
    _assert_run_config(tmp_path, "sft", tiny_model)


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
    _assert_run_config(tmp_path, "dpo", tiny_model)


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


# ── Data formats ───────────────────────────────────────────────────────────

CHAT = [
    [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "Hi"},
        {"role": "user", "content": "Again"},
        {"role": "assistant", "content": "Hi again"},
    ],
    [{"role": "user", "content": "Say bye"}, {"role": "assistant", "content": "Bye"}],
    [{"role": "user", "content": "Count"}, {"role": "assistant", "content": "1 2"}],
]


def test_chat_sft_trains_only_the_final_answer(tiny_model):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    from data.preprocessing import to_sft_dataset

    tokenizer = AutoTokenizer.from_pretrained(tiny_model)
    tokenizer.pad_token = tokenizer.eos_token
    ds = to_sft_dataset(Dataset.from_dict({"messages": CHAT[:1]}), False, "")
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(tiny_model),
        args=SFTConfig(output_dir="/tmp/chat_label_check", report_to="none", bf16=False),
        train_dataset=ds,
        processing_class=tokenizer,
    )
    batch = next(iter(trainer.get_train_dataloader()))
    trained_ids = [t for t in batch["labels"][0].tolist() if t != -100]
    trained_text = tokenizer.decode(trained_ids)
    assert "Hi again" in trained_text
    assert "Say hi" not in trained_text and "Again" not in trained_text


def test_chat_sft_trains_end_to_end_with_token_report(tiny_model, tmp_path):
    from core.run_config import load_run_config
    from training.sft import train_model

    summary, _ = train_model(
        tiny_model, Dataset.from_dict({"messages": CHAT}), str(tmp_path), _hyperparams(),
        "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, True, "", training_mode="sft", progress=None,
    )  # fmt: skip
    assert summary.startswith("✅ Training complete") and "📏 Tokens per example" in summary
    stats = load_run_config(str(tmp_path / "run_config.yaml"))["token_stats"]
    assert stats["sampled"] == 3 and stats["max_length"] == 64


def test_truncation_is_reported(tiny_model, tmp_path):
    from training.sft import train_model

    ds = Dataset.from_dict({"text": ["word " * 60, "short text", "another short"]})
    summary, _ = train_model(
        tiny_model, ds, str(tmp_path), {**_hyperparams(), "max_length": 16}, "cpu", "LoRA",
        True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, False, "", training_mode="sft", progress=None,
    )  # fmt: skip
    assert "⚠️ 1 (33.3%) exceed Max Sequence Length 16" in summary


def test_eval_split_zero_disables_evaluation(tiny_model, tmp_path, monkeypatch):
    import trl

    from training.sft import train_model

    seen = {}
    original = trl.SFTTrainer

    class Spy(original):
        def __init__(self, *args, **kwargs):
            seen["eval"] = kwargs.get("eval_dataset")
            seen["train_rows"] = len(kwargs["train_dataset"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(trl, "SFTTrainer", Spy)
    ds = Dataset.from_dict({"text": [f"row {i}" for i in range(10)]})
    train_model(
        tiny_model, ds, str(tmp_path), {**_hyperparams(), "eval_split": 0.0}, "cpu", "LoRA",
        True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, False, "", training_mode="sft", progress=None,
    )  # fmt: skip
    assert seen == {"eval": None, "train_rows": 10}


def test_dpo_uses_the_max_sequence_length(tiny_model, tmp_path, monkeypatch):
    import trl

    from training.sft import train_model

    seen = {}
    original = trl.DPOTrainer

    class Spy(original):
        def __init__(self, *args, **kwargs):
            seen["max_length"] = kwargs["args"].max_length
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(trl, "DPOTrainer", Spy)
    train_model(
        tiny_model, Dataset.from_dict(PREFS), str(tmp_path), {**_hyperparams(), "max_length": 48},
        "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, False, "", training_mode="dpo", progress=None,
    )  # fmt: skip
    assert seen["max_length"] == 48


def test_chat_data_without_chat_template_is_a_clear_error(tiny_model, tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    from training.sft import train_model

    original = AutoTokenizer.from_pretrained

    def no_template(*args, **kwargs):
        tok = original(*args, **kwargs)
        tok.chat_template = None
        return tok

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", no_template)
    with pytest.raises(RuntimeError, match="no chat template"):
        train_model(
            tiny_model, Dataset.from_dict({"messages": CHAT}), str(tmp_path), _hyperparams(),
            "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
            False, True, "", training_mode="sft", progress=None,
        )  # fmt: skip


def test_cli_trains_from_hub_dataset(tiny_model, tmp_path, monkeypatch):
    import datasets
    from typer.testing import CliRunner

    from cli.commands import app

    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda repo, config, split, streaming: Dataset.from_dict(
            {"messages": CHAT, "extra": [1, 2, 3]}
        ).to_iterable_dataset(),
    )
    out = tmp_path / "hub_run"
    result = CliRunner().invoke(app, [
        "train", "--model", tiny_model, "--hf-dataset", "owner/chat", "--output", str(out),
        "--epochs", "1", "--batch-size", "2", "--max-length", "64", "--lora-rank", "4",
        "--eval-split", "0",
    ])  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "hf:owner/chat" in result.output
    _assert_safetensors_adapter(out)

    both = CliRunner().invoke(
        app, ["train", "--model", tiny_model, "--data", "x.csv", "--hf-dataset", "o/n"]
    )
    assert both.exit_code == 1 and "exactly one" in both.output


def test_long_prompts_are_skipped_and_reported(tiny_model, tmp_path):
    from core.run_config import load_run_config
    from training.sft import train_model

    long_chat = [{"role": "user", "content": "word " * 200}, {"role": "assistant", "content": "ok"}]
    ds = Dataset.from_dict({"messages": [long_chat, *CHAT]})
    summary, _ = train_model(
        tiny_model, ds, str(tmp_path), {**_hyperparams(), "max_length": 64}, "cpu", "LoRA",
        True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False,
        False, True, "", training_mode="sft", progress=None,
    )  # fmt: skip
    assert "⚠️ 1 examples skipped" in summary
    assert load_run_config(str(tmp_path / "run_config.yaml"))["dropped_long_prompts"] == 1


def test_all_prompts_too_long_is_a_clear_error(tiny_model, tmp_path):
    from training.sft import train_model

    long_chat = [{"role": "user", "content": "word " * 200}, {"role": "assistant", "content": "ok"}]
    with pytest.raises(RuntimeError, match="Raise Max Sequence Length"):
        train_model(
            tiny_model, Dataset.from_dict({"messages": [long_chat] * 3}), str(tmp_path),
            {**_hyperparams(), "max_length": 64}, "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10, 16,
            False, 0, "linear", False, False, True, "", training_mode="sft", progress=None,
        )  # fmt: skip


# ── Evaluation: fine-tuned vs base ─────────────────────────────────────────


@pytest.fixture
def random_lora(tiny_model, tmp_path):
    """A LoRA adapter with non-zero weights, so it visibly changes the output."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    config = LoraConfig(r=8, lora_alpha=64, target_modules=["q_proj", "v_proj"],
                        init_lora_weights=False)  # fmt: skip
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(tiny_model), config)
    model.save_pretrained(tmp_path / "adapter")
    return str(tmp_path / "adapter")


def test_base_predictions_match_the_plain_base_model(tiny_model, random_lora):
    from transformers import AutoModelForCausalLM

    from inference.evaluation import generate_predictions
    from inference.generate import _load_for_inference

    prompts = ["alpha beta", "a longer prompt with more words", "x"]
    model, tokenizer = _load_for_inference(tiny_model, random_lora)
    tuned = generate_predictions(model, tokenizer, prompts, 6)
    base = generate_predictions(model, tokenizer, prompts, 6, base_model=True)
    plain = AutoModelForCausalLM.from_pretrained(tiny_model).eval()
    assert base == generate_predictions(plain, tokenizer, prompts, 6)
    assert tuned != base
    # The adapter is only switched off for that call: the shared model is unchanged.
    assert generate_predictions(model, tokenizer, prompts, 6) == tuned
    with pytest.raises(ValueError, match="LoRA"):
        generate_predictions(plain, tokenizer, prompts, 6, base_model=True)


def test_ui_evaluation_compares_with_base(tiny_model, random_lora, tmp_path):
    from inference.evaluation import on_evaluate_click

    data = tmp_path / "test.csv"
    pd.DataFrame({"prompt": ["alpha", "beta", "gamma"], "reference": ["a", "b", "c"]}).to_csv(
        data, index=False
    )
    metrics, table, _ = on_evaluate_click(
        "gpt2", tiny_model, random_lora, _Upload(data), False, True, tiny_model, "helpfulness",
        eval_max_new_tokens=4, compare_base=True, progress=lambda *a, **k: None,
    )  # fmt: skip
    assert "| Metric | Fine-tuned | Base | Δ |" in metrics, metrics
    assert "| Judge score (1-10) |" in metrics  # a random judge gives "n/a" or a number
    assert list(table.columns) == [
        "prompt", "prediction", "base_prediction", "reference",
        "judge_score", "judgment", "base_judge_score", "base_judgment",
    ]  # fmt: skip
    assert len(table) == 3


# ── GGUF export: llama.cpp fallback ────────────────────────────────────────


def test_gguf_fallback_converts_the_merged_model(tiny_model, random_lora, tmp_path, monkeypatch):
    """llama.cpp converts full models only: an adapter is merged into its base first."""
    import subprocess
    import sys

    import export.gguf as gguf

    seen = {}

    def fake_run(cmd, **kwargs):
        model_dir = cmd[2]
        seen.update(cmd=cmd, files=sorted(os.listdir(model_dir)), model_dir=model_dir)
        pathlib.Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(gguf, "HAS_UNSLOTH", False)
    monkeypatch.setattr(gguf.shutil, "which", lambda name: "/x/convert_hf_to_gguf.py"
                        if name == "convert_hf_to_gguf.py" else None)  # fmt: skip
    monkeypatch.setattr(gguf.subprocess, "run", fake_run)

    status = gguf.export_to_gguf(random_lora, str(tmp_path / "out"), "q4_k_m")
    assert status.startswith("✅ GGUF exported (FP16 only)"), status
    assert seen["cmd"][0] == sys.executable  # not whatever "python" is on PATH
    assert "config.json" in seen["files"] and "adapter_config.json" not in seen["files"]
    assert any(f.endswith(".safetensors") for f in seen["files"])
    assert not os.path.exists(seen["model_dir"])  # temporary merge removed


def test_gguf_fallback_rejects_non_lora_adapters(tmp_path, monkeypatch):
    import json

    import export.gguf as gguf

    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "PROMPT_TUNING"}))
    monkeypatch.setattr(gguf, "HAS_UNSLOTH", False)
    monkeypatch.setattr(gguf.shutil, "which", lambda name: "/x/convert_hf_to_gguf.py")
    status = gguf.export_to_gguf(str(tmp_path), str(tmp_path / "out"), "q4_k_m")
    assert "PROMPT_TUNING adapters cannot be merged" in status


# ── GRPO / KTO checkpoints and resume ──────────────────────────────────────


@pytest.mark.parametrize("trainer_name", ["grpo", "kto"])
def test_grpo_and_kto_save_checkpoints_and_resume(tiny_model, tmp_path, monkeypatch,
                                                   trainer_name):  # fmt: skip
    import transformers

    import training.grpo as grpo
    import training.kto as kto
    from core.run_config import latest_checkpoint

    resumed_from = []
    original_train = transformers.Trainer.train

    def spy_train(self, resume_from_checkpoint=None, **kwargs):
        resumed_from.append(resume_from_checkpoint)
        return original_train(self, resume_from_checkpoint=resume_from_checkpoint, **kwargs)

    monkeypatch.setattr(transformers.Trainer, "train", spy_train)
    monkeypatch.setattr(grpo, "CHECKPOINT_SAVE_STEPS", 1)
    monkeypatch.setattr(kto, "CHECKPOINT_SAVE_STEPS", 1)
    out = tmp_path / trainer_name

    def run(epochs: int, resume: bool) -> str:
        if trainer_name == "grpo":
            data = tmp_path / "prompts.csv"
            pd.DataFrame({"prompt": ["2+2=", "3+3="], "reference": ["4", "6"]}).to_csv(
                data, index=False
            )
            return grpo.train_grpo(
                tiny_model, "", _Upload(data), str(out), epochs=epochs,
                num_generations=2, max_completion_length=8, resume=resume, progress=None,
            )  # fmt: skip
        data = tmp_path / "kto.csv"
        pd.DataFrame(PREFS).to_csv(data, index=False)
        return kto.train_kto(
            tiny_model, _Upload(data), str(out), epochs=epochs, batch_size=2,
            max_length=64, resume=resume, progress=None,
        )  # fmt: skip

    assert "✅" in run(epochs=1, resume=False)
    first = latest_checkpoint(str(out))
    assert first is not None and resumed_from == [None]
    assert "✅" in run(epochs=2, resume=True)
    assert resumed_from[-1] == first  # continued from the newest checkpoint
    assert int(latest_checkpoint(str(out)).rsplit("-", 1)[-1]) > int(first.rsplit("-", 1)[-1])
    assert len(list(out.glob("checkpoint-*"))) <= 2  # older checkpoints are pruned


@pytest.mark.parametrize(("exit_code", "expected"), [(0, "🔓 Heretic Mode applied!"),
                                                     (1, "⚠️ Heretic failed (exit code 1)")])  # fmt: skip
def test_heretic_result_follows_its_exit_code(tiny_model, tmp_path, monkeypatch, exit_code,
                                              expected):  # fmt: skip
    import training.sft as sft

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "heretic"
    fake.write_text(f"#!/bin/sh\necho 'ValueError: model too small' >&2\nexit {exit_code}\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(sft, "HAS_HERETIC", True)

    ds = Dataset.from_dict({"text": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    summary, _ = sft.train_model(
        tiny_model, ds, str(tmp_path / "out"), _hyperparams(), "cpu", "LoRA", True, 4, 8,
        10, 64, 1, 10, 16, False, 0, "linear", False, False, False, "",
        heretic_mode=True, progress=None,
    )  # fmt: skip
    assert expected in summary
    assert ("model too small" in summary) == (exit_code != 0)


# ── LoRA variants (every linear layer) ─────────────────────────────────────


@pytest.mark.parametrize(
    ("variant", "flags"),
    [("LoRA", (False, False)), ("rsLoRA", (True, False)), ("DoRA", (False, True))],
)
def test_lora_variants_train_on_every_linear_layer(tiny_model, tmp_path, variant, flags):
    import json

    from training.sft import train_model

    ds = Dataset.from_dict({"text": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    summary, _ = train_model(
        tiny_model, ds, str(tmp_path), _hyperparams(), "cpu", "LoRA", True, 4, 8, 10, 64, 1,
        10, 16, False, 0, "linear", False, False, False, "", progress=None,
        lora_variant=variant,
    )  # fmt: skip
    assert summary.startswith("✅ Training complete"), summary
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert (config["use_rslora"], config["use_dora"]) == flags
    # "all-linear" resolves to every attention and MLP projection, never the output head.
    assert {name.rsplit(".", 1)[-1] for name in config["target_modules"]} == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }  # fmt: skip
    assert _assert_run_config(tmp_path, "sft", tiny_model)["peft"]["lora_variant"] == variant


def test_dora_adapters_cannot_be_compared_with_base(tiny_model, tmp_path):
    """PEFT cannot switch DoRA off per request (adapter_names), so the comparison refuses."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from inference.evaluation import generate_predictions, is_lora_model

    base = AutoModelForCausalLM.from_pretrained(tiny_model)
    dora = get_peft_model(base, LoraConfig(r=4, target_modules="all-linear", use_dora=True))
    assert not is_lora_model(dora)
    with pytest.raises(ValueError, match="DoRA"):
        generate_predictions(dora, AutoTokenizer.from_pretrained(tiny_model), ["hi"], 2,
                             base_model=True)  # fmt: skip


def test_cli_rejects_unknown_lora_variant(tmp_path):
    from typer.testing import CliRunner

    from cli.commands import app

    result = CliRunner().invoke(
        app, ["train", "--model", "gpt2", "--data", "x.csv", "--lora-variant", "PiSSA"]
    )
    assert result.exit_code == 1 and "--lora-variant must be one of" in result.output


def test_grpo_with_builtin_rewards_loss_type_and_lora_options(tiny_model, tmp_path):
    import json

    from training.grpo import train_grpo

    data = tmp_path / "prompts.csv"
    pd.DataFrame({"prompt": ["2+2=", "3+3="], "reference": ["4", "6"]}).to_csv(data, index=False)
    out = tmp_path / "grpo_opts"
    result = train_grpo(
        tiny_model, "", _Upload(data), str(out), num_generations=2, max_completion_length=8,
        loss_type="dr_grpo", rewards=["reference", "math", "think_format", "json", "regex"],
        regex_pattern=r"\d+", lora_rank=8, lora_alpha=16, lora_variant="rsLoRA",
        progress=None,
    )  # fmt: skip
    assert result.startswith("✅ GRPO training complete"), result
    assert "reference match + math + think format + json + regex" in result
    adapter = json.loads((out / "adapter_config.json").read_text())
    assert (adapter["r"], adapter["lora_alpha"], adapter["use_rslora"]) == (8, 16, True)
    cfg = _assert_run_config(out, "grpo", tiny_model)
    assert cfg["loss_type"] == "dr_grpo" and cfg["regex_pattern"] == r"\d+"
    assert cfg["rewards"] == ["reference", "math", "think_format", "json", "regex"]
    assert cfg["peft"]["lora_variant"] == "rsLoRA" and cfg["use_vllm"] is False


def test_cli_grpo_passes_new_options(tiny_model, tmp_path):
    from typer.testing import CliRunner

    from cli.commands import app

    data = tmp_path / "prompts.csv"
    pd.DataFrame({"prompt": ["Say JSON", "Say JSON"]}).to_csv(data, index=False)
    out = tmp_path / "cli_grpo"
    result = CliRunner().invoke(
        app,
        ["grpo", "--policy-model", tiny_model, "--data", str(data), "--output", str(out),
         "--num-generations", "2", "--max-completion-length", "8", "--reward", "json",
         "--loss-type", "bnpo", "--lora-variant", "DoRA", "--lora-rank", "4"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    cfg = _assert_run_config(out, "grpo", tiny_model)
    assert cfg["rewards"] == ["json"] and cfg["loss_type"] == "bnpo"
    assert cfg["peft"] == {"method": "LoRA", "lora_rank": 4, "lora_alpha": 32,
                           "lora_variant": "DoRA"}  # fmt: skip


# ── Tool calling and reasoning data ────────────────────────────────────────

TOOL_ROWS = [
    {
        "messages": [
            {"role": "user", "content": "Lights on in the kitchen"},
            {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                "name": "control_light", "arguments": {"room": "kitchen", "state": "on"}}}]},
            {"role": "tool", "name": "control_light", "content": "ok"},
            {"role": "assistant", "content": "Done!"},
        ],
        "tools": [{"type": "function", "function": {"name": "control_light",
                   "parameters": {"type": "object", "properties": {"room": {"type": "string"},
                                  "state": {"type": "string"}}}}}],
    },
    {
        "messages": [
            {"role": "user", "content": "Weather in Rome?"},
            {"role": "assistant", "reasoning_content": "Need the weather tool.",
             "tool_calls": [{"type": "function", "function": {
                 "name": "get_weather", "arguments": '{"city": "Rome"}'}}]},
        ],
        "tools": [{"type": "function", "function": {"name": "get_weather",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}],
    },
]  # fmt: skip


def _tool_sft_dataset(tmp_path):
    import json

    from data.loader import load_dataset_from_file
    from data.preprocessing import to_sft_dataset, validate_and_clean_dataset

    data = tmp_path / "tools.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in TOOL_ROWS))
    cleaned, _ = validate_and_clean_dataset(load_dataset_from_file(_Upload(data), "jsonl"))
    return to_sft_dataset(cleaned, True, "")


def test_tool_calls_and_reasoning_are_what_the_model_learns(tiny_chat_model, tmp_path):
    """Every assistant turn is a target: the calls (exact arguments) and the answer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(tiny_chat_model)
    trainer = SFTTrainer(
        model=AutoModelForCausalLM.from_pretrained(tiny_chat_model),
        args=SFTConfig(output_dir=str(tmp_path / "t"), max_length=512, report_to="none",
                       bf16=False, fp16=False),
        train_dataset=_tool_sft_dataset(tmp_path),
        processing_class=tokenizer,
    )  # fmt: skip

    def trained_text(i: int) -> str:
        row = trainer.train_dataset[i]
        mask = row.get("completion_mask") or [label != -100 for label in row["labels"]]
        return tokenizer.decode([t for t, m in zip(row["input_ids"], mask, strict=True) if m])

    call = trained_text(0)
    assert '{"name": "control_light", "arguments": {"room": "kitchen", "state": "on"}}' in call
    assert "Done!" in trained_text(1) and "tool_call" not in trained_text(1)
    weather = trained_text(2)  # JSON-string arguments became an object; reasoning kept
    assert "Need the weather tool." in weather
    assert '"arguments": {"city": "Rome"}' in weather and "room" not in weather
    prompt = tokenizer.decode(trainer.train_dataset[0]["input_ids"])
    assert "control_light" in prompt.split("Lights on")[0]  # tool schemas in the system prompt


def test_tool_calling_sft_trains_end_to_end(tiny_chat_model, tmp_path):
    import json

    from data.loader import load_dataset_from_file
    from data.preprocessing import validate_and_clean_dataset
    from training.sft import train_model

    data = tmp_path / "tools.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in TOOL_ROWS))
    ds, _ = validate_and_clean_dataset(load_dataset_from_file(_Upload(data), "jsonl"))
    summary, _ = train_model(
        tiny_chat_model, ds, str(tmp_path / "out"), {**_hyperparams(), "max_length": 512},
        "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False, False,
        True, "", progress=None,
    )  # fmt: skip
    assert summary.startswith("✅ Training complete"), summary
    _assert_safetensors_adapter(tmp_path / "out")


# ── Vision-language fine-tuning ────────────────────────────────────────────

COLOURS = ["red", "blue", "green", "yellow"]


def _vision_file(tmp_path, with_parts: bool = True):
    """A local chat file whose images are PNGs next to it (relative paths)."""
    import json

    from PIL import Image

    (tmp_path / "img").mkdir()
    rows = []
    for i, colour in enumerate(COLOURS):
        Image.new("RGB", (32, 32), colour).save(tmp_path / "img" / f"{i}.png")
        question = "What colour is this?"
        user = [{"type": "image"}, {"type": "text", "text": question}] if with_parts else question
        rows.append({"messages": [{"role": "user", "content": user},
                                  {"role": "assistant", "content": f"It is {colour}."}],
                     "images": [f"img/{i}.png"]})  # fmt: skip
    data = tmp_path / "vision.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in rows))
    return data


@pytest.mark.parametrize("with_parts", [True, False], ids=["image-parts", "text-only-content"])
def test_vision_sft_learns_only_the_answer(tiny_vlm, tmp_path, with_parts):
    """Image parts, or plain text where TRL places the image — loss on the answer only."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from trl import SFTConfig, SFTTrainer

    from data.loader import load_dataset_from_file
    from data.preprocessing import to_sft_dataset, validate_and_clean_dataset

    ds = load_dataset_from_file(_Upload(_vision_file(tmp_path, with_parts)), "jsonl")
    cleaned, issues = validate_and_clean_dataset(ds)
    assert len(cleaned) == 4 and not issues
    processor = AutoProcessor.from_pretrained(tiny_vlm)
    trainer = SFTTrainer(
        model=AutoModelForImageTextToText.from_pretrained(tiny_vlm),
        args=SFTConfig(output_dir=str(tmp_path / "t"), max_length=None, report_to="none",
                       bf16=False, fp16=False, per_device_train_batch_size=1),
        train_dataset=to_sft_dataset(cleaned, True, ""),
        processing_class=processor,
    )  # fmt: skip
    batch = next(iter(trainer.get_train_dataloader()))
    labels, ids = batch["labels"][0], batch["input_ids"][0]
    trained = processor.tokenizer.decode(ids[labels != -100])
    assert trained.startswith("It is ") and "colour" not in trained
    assert batch["pixel_values"].numel() > 0  # the image reached the model


def test_vision_sft_trains_end_to_end_with_card(tiny_vlm, tmp_path):
    from huggingface_hub import ModelCard

    from data.loader import load_dataset_from_file
    from data.preprocessing import validate_and_clean_dataset
    from training.sft import train_model

    ds, _ = validate_and_clean_dataset(
        load_dataset_from_file(_Upload(_vision_file(tmp_path)), "jsonl")
    )
    out = tmp_path / "vlm"
    summary, _ = train_model(
        tiny_vlm, ds, str(out), _hyperparams(), "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10, 16,
        False, 0, "linear", False, False, True, "", progress=None, lora_variant="rsLoRA",
    )  # fmt: skip
    assert summary.startswith("✅ Training complete") and "Vision-language" in summary, summary
    _assert_safetensors_adapter(out)
    assert (out / "preprocessor_config.json").is_file() or (out / "processor_config.json").is_file()
    card = ModelCard.load(str(out / "README.md")).data
    assert card.pipeline_tag == "image-text-to-text" and "vision" in card.tags
    cfg = _assert_run_config(out, "sft", tiny_vlm)
    assert cfg["vision"] is True and cfg["peft"]["lora_variant"] == "rsLoRA"


def test_vision_rejects_dpo_and_prompt_tuning(tiny_vlm, tmp_path):
    from data.loader import load_dataset_from_file
    from data.preprocessing import validate_and_clean_dataset
    from training.sft import train_model

    ds, _ = validate_and_clean_dataset(
        load_dataset_from_file(_Upload(_vision_file(tmp_path)), "jsonl")
    )
    args = (tiny_vlm, ds, str(tmp_path / "x"), _hyperparams(), "cpu")
    rest = (True, 4, 8, 10, 64, 1, 10, 16, False, 0, "linear", False, False, True, "")
    with pytest.raises(RuntimeError, match="DPO on image"):
        train_model(*args, "LoRA", *rest, training_mode="dpo", progress=None)
    with pytest.raises(RuntimeError, match="not supported for vision"):
        train_model(*args, "Prompt Tuning", *rest, progress=None)


# ── Multi-process (data-parallel) training ─────────────────────────────────

_DDP_PROBE = """
import hashlib, json, os, sys
sys.path.insert(0, {repo!r})
import transformers
from datasets import Dataset
from training.sft import train_model

original = transformers.Trainer.train

def spy(self, *args, **kwargs):
    out = original(self, *args, **kwargs)
    digest = hashlib.sha256()
    for name, param in sorted(self.model.named_parameters()):
        if param.requires_grad:
            digest.update(name.encode() + param.detach().cpu().numpy().tobytes())
    with open(os.path.join({report!r}, f"rank{{os.environ['RANK']}}.json"), "w") as f:
        json.dump({{"world_size": self.args.world_size, "steps": self.state.global_step,
                   "weights": digest.hexdigest()}}, f)
    return out

transformers.Trainer.train = spy
ds = Dataset.from_dict({{"instruction": [f"Say {{i}}" for i in range(8)],
                        "output": [str(i) for i in range(8)]}})
hp = {{"learning_rate": 1e-2, "epochs": 1, "batch_size": 2, "grad_accum": 1,
      "max_length": 64, "warmup_steps": 0, "eval_split": 0.0}}
train_model({model!r}, ds, {out!r}, hp, "cpu", "LoRA", True, 4, 8, 10, 64, 1, 10, 16,
            False, 0, "linear", False, False, False, "", progress=None)
"""


def test_two_process_training_is_data_parallel(tiny_model, tmp_path):
    """torchrun with 2 CPU processes: gradients are synchronised, rank 0 alone saves."""
    import json
    import subprocess
    import sys

    report, out = tmp_path / "report", tmp_path / "out"
    report.mkdir()
    script = tmp_path / "probe.py"
    repo = str(pathlib.Path(__file__).resolve().parents[1])
    script.write_text(_DDP_PROBE.format(repo=repo, report=str(report), out=str(out),
                                        model=tiny_model))  # fmt: skip
    result = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2", str(script)],
        capture_output=True, text=True, timeout=600,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1"},
    )  # fmt: skip
    assert result.returncode == 0, result.stderr[-3000:]
    ranks = [json.loads((report / f"rank{r}.json").read_text()) for r in (0, 1)]
    assert [r["world_size"] for r in ranks] == [2, 2]
    # 8 rows, batch 2 per process, 2 processes → 2 optimiser steps (not 4).
    assert [r["steps"] for r in ranks] == [2, 2]
    assert ranks[0]["weights"] == ranks[1]["weights"]  # DDP kept the copies identical
    _assert_safetensors_adapter(out)
    _assert_run_config(out, "sft", tiny_model)
