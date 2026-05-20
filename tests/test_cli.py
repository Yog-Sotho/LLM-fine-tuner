"""
tests/test_cli.py
==================
Unit tests for cli/commands.py.

Uses Typer's CliRunner so no subprocess is spawned.
Heavy functions (train_model, run_ppo_v27, etc.) are patched so tests
run without GPU, models, or real datasets.

Covers:
  - v3.2 Fix #3: --help is handled by Typer, not Gradio
  - train command exits 1 when data file is missing
  - train command exits 1 when file extension is unsupported
  - --qlora-enhanced overrides --peft (Minor Fix 1)
  - reward exits 1 when HAS_REWARD_TRAINER is False
  - orpo exits 1 when HAS_ORPO is False
  - ppo exits 1 when reward model path does not exist
  - evaluate exits 1 when data file is missing
  - DummyFile proxy carries .name attribute correctly
"""

import os
import tempfile

import pandas as pd
import pytest
from typer.testing import CliRunner

from cli.commands import app, DummyFile


runner = CliRunner()


# ── DummyFile ──────────────────────────────────────────────────────────────

def test_dummy_file_has_name():
    df = DummyFile("/tmp/foo.csv")
    assert df.name == "/tmp/foo.csv"


# ── --help (v3.2 Fix #3) ──────────────────────────────────────────────────

def test_help_flag_exits_zero():
    """--help must print usage and exit 0 — not launch Gradio."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output or "Commands" in result.output


def test_train_help_exits_zero():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


# ── train — missing data ───────────────────────────────────────────────────

def test_train_missing_data_file_exits_one():
    result = runner.invoke(app, [
        "train",
        "--model", "gpt2",
        "--data", "/nonexistent/path/data.csv",
    ])
    assert result.exit_code != 0


def test_train_unsupported_extension_exits_one():
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        path = f.name
    try:
        result = runner.invoke(app, [
            "train",
            "--model", "gpt2",
            "--data", path,
        ])
        assert result.exit_code != 0
    finally:
        os.unlink(path)


# ── train — --qlora-enhanced overrides --peft (Minor Fix 1) ───────────────

def test_qlora_enhanced_override_message(monkeypatch):
    """Invoking --qlora-enhanced with a non-QLoRA --peft should print override warning."""
    import cli.commands as cmd_mod

    # Patch load + train so the command succeeds immediately after the guard
    monkeypatch.setattr(cmd_mod, "load_dataset_from_file",
                        lambda *a, **kw: (_ for _ in ()).throw(SystemExit(0)))

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    pd.DataFrame([{"text": "hello"}]).to_csv(path, index=False)
    try:
        result = runner.invoke(app, [
            "train",
            "--model", "gpt2",
            "--data", path,
            "--peft", "LoRA",
            "--qlora-enhanced",
        ])
        # The override warning must appear before any error
        assert "overrides" in result.output or "QLoRA Enhanced" in result.output
    finally:
        os.unlink(path)


# ── reward — dependency missing ────────────────────────────────────────────

def test_reward_exits_when_no_reward_trainer(monkeypatch):
    import cli.commands as cmd_mod
    monkeypatch.setattr(cmd_mod, "HAS_REWARD_TRAINER", False)
    result = runner.invoke(app, [
        "reward",
        "--model", "gpt2",
        "--data", "fake.csv",
    ])
    assert result.exit_code != 0
    assert "trl" in result.output.lower() or "install" in result.output.lower()


# ── orpo — dependency missing ──────────────────────────────────────────────

def test_orpo_exits_when_no_orpo(monkeypatch):
    import cli.commands as cmd_mod
    monkeypatch.setattr(cmd_mod, "HAS_ORPO", False)
    result = runner.invoke(app, [
        "orpo",
        "--model", "gpt2",
        "--data", "fake.csv",
    ])
    assert result.exit_code != 0
    assert "trl" in result.output.lower() or "install" in result.output.lower()


# ── ppo — invalid reward model path ───────────────────────────────────────

def test_ppo_exits_on_invalid_reward_model_path(monkeypatch):
    import cli.commands as cmd_mod
    monkeypatch.setattr(cmd_mod, "HAS_PPO", True)
    result = runner.invoke(app, [
        "ppo",
        "--policy-model", "gpt2",
        "--reward-model", "/nonexistent/reward",
        "--data", "fake.csv",
    ])
    assert result.exit_code != 0


# ── evaluate — missing data file ──────────────────────────────────────────

def test_evaluate_exits_on_missing_data():
    result = runner.invoke(app, [
        "evaluate",
        "--model", "gpt2",
        "--data", "/nonexistent/test.csv",
    ])
    assert result.exit_code != 0
