"""
tests/test_cli.py
==================
Unit tests for cli/commands.py.

Uses Typer's CliRunner so no subprocess is spawned.
Heavy functions (train_model, train_grpo, etc.) are patched so tests
run without GPU, models, or real datasets.

Covers:
  - v3.2 Fix #3: --help is handled by Typer, not Gradio
  - train command exits 1 when data file is missing
  - train command exits 1 when file extension is unsupported
  - --qlora-enhanced overrides --peft (Minor Fix 1)
  - reward exits 1 when HAS_REWARD_TRAINER is False
  - orpo exits 1 when HAS_ORPO is False
  - grpo exits 1 when the reward model path is not a saved model
  - evaluate exits 1 when data file is missing
  - DummyFile proxy carries .name attribute correctly
"""

import os
import re
import tempfile

import pandas as pd
from typer.testing import CliRunner

from cli.commands import DummyFile, app

runner = CliRunner()

# Typer forces Rich colour output when GITHUB_ACTIONS is set, and the ANSI codes
# split option names (e.g. "-\x1b[0m\x1b[1;36m-model"). Assert on plain text.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI.sub("", text)


# ── DummyFile ──────────────────────────────────────────────────────────────


def test_dummy_file_has_name():
    df = DummyFile("/tmp/foo.csv")
    assert df.name == "/tmp/foo.csv"


# ── --help (v3.2 Fix #3) ──────────────────────────────────────────────────


def test_help_flag_exits_zero():
    """--help must print usage and exit 0 — not launch Gradio."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in _plain(result.output) or "Commands" in _plain(result.output)


def test_train_help_exits_zero():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--model" in _plain(result.output)


# ── train — missing data ───────────────────────────────────────────────────


def test_train_missing_data_file_exits_one():
    result = runner.invoke(
        app,
        [
            "train",
            "--model",
            "gpt2",
            "--data",
            "/nonexistent/path/data.csv",
        ],
    )
    assert result.exit_code != 0


def test_train_unsupported_extension_exits_one():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        result = runner.invoke(
            app,
            [
                "train",
                "--model",
                "gpt2",
                "--data",
                path,
            ],
        )
        assert result.exit_code != 0
    finally:
        os.unlink(path)


# ── train — --qlora-enhanced overrides --peft (Minor Fix 1) ───────────────


def test_qlora_enhanced_override_message(monkeypatch):
    """Invoking --qlora-enhanced with a non-QLoRA --peft should print override warning."""
    import cli.commands as cmd_mod

    # Patch load + train so the command succeeds immediately after the guard
    monkeypatch.setattr(
        cmd_mod, "load_dataset_from_file", lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0))
    )

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    pd.DataFrame([{"text": "hello"}]).to_csv(path, index=False)
    try:
        result = runner.invoke(
            app,
            [
                "train",
                "--model",
                "gpt2",
                "--data",
                path,
                "--peft",
                "LoRA",
                "--qlora-enhanced",
            ],
        )
        # The override warning must appear before any error
        assert "overrides" in _plain(result.output) or "QLoRA Enhanced" in _plain(result.output)
    finally:
        os.unlink(path)


# ── reward — dependency missing ────────────────────────────────────────────


def test_reward_exits_when_no_reward_trainer(monkeypatch):
    import cli.commands as cmd_mod

    monkeypatch.setattr(cmd_mod, "HAS_REWARD_TRAINER", False)
    result = runner.invoke(
        app,
        [
            "reward",
            "--model",
            "gpt2",
            "--data",
            "fake.csv",
        ],
    )
    assert result.exit_code != 0
    assert "trl" in _plain(result.output).lower() or "install" in _plain(result.output).lower()


# ── orpo — dependency missing ──────────────────────────────────────────────


def test_orpo_exits_when_no_orpo(monkeypatch):
    import cli.commands as cmd_mod

    monkeypatch.setattr(cmd_mod, "HAS_ORPO", False)
    result = runner.invoke(
        app,
        [
            "orpo",
            "--model",
            "gpt2",
            "--data",
            "fake.csv",
        ],
    )
    assert result.exit_code != 0
    assert "trl" in _plain(result.output).lower() or "install" in _plain(result.output).lower()


# ── grpo — invalid reward model path ──────────────────────────────────────


def test_grpo_exits_on_invalid_reward_model_path(tmp_path):
    data = tmp_path / "prompts.csv"
    data.write_text("prompt\nhello\n")
    result = runner.invoke(
        app,
        [
            "grpo",
            "--policy-model",
            "gpt2",
            "--reward-model",
            str(tmp_path / "not_a_model"),
            "--data",
            str(data),
        ],
    )
    assert result.exit_code == 1
    assert "Reward model path must be a saved model directory" in _plain(result.output)


def test_grpo_requires_a_reward_source(tmp_path):
    data = tmp_path / "prompts.csv"
    data.write_text("prompt\nhello\n")
    result = runner.invoke(app, ["grpo", "--policy-model", "gpt2", "--data", str(data)])
    assert result.exit_code == 1
    assert "GRPO needs a reward" in _plain(result.output)


# ── evaluate — missing data file ──────────────────────────────────────────


def test_evaluate_exits_on_missing_data():
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--model",
            "gpt2",
            "--data",
            "/nonexistent/test.csv",
        ],
    )
    assert result.exit_code != 0
