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


def test_inference_with_trained_adapter(tiny_model, tmp_path):
    from inference.generate import generate_text

    ds = Dataset.from_dict({"text": ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]})
    _train(tiny_model, ds, tmp_path, "sft")

    reply = generate_text(tiny_model, str(tmp_path), "alpha", max_new_tokens=4)
    assert isinstance(reply, str)
    assert not reply.startswith("❌"), reply


def test_reward_and_ppo_report_unavailable_instead_of_crashing():
    from config.constants import HAS_PPO
    from training.ppo import run_ppo_v27
    from training.reward import train_reward_model_v27

    if HAS_PPO:
        pytest.skip("Legacy PPO API present; this test covers supported TRL versions")
    assert "being rebuilt" in train_reward_model_v27("m", object(), "out", progress=None)
    assert "being rebuilt" in run_ppo_v27("m", ".", object(), "out", progress=None)


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
