
import pytest
from typer.testing import CliRunner
from cli.commands import app

runner = CliRunner()

def test_cli_train_data_traversal():
    result = runner.invoke(app, ["train", "--model", "gpt2", "--data", "../unsafe.csv", "--output", "./ok"])
    # Currently this might fail with "Dataset not found" but not with the traversal error message
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_reward_data_traversal():
    result = runner.invoke(app, ["reward", "--model", "gpt2", "--data", "../unsafe.csv", "--output", "./ok"])
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_orpo_data_traversal():
    result = runner.invoke(app, ["orpo", "--model", "gpt2", "--data", "../unsafe.csv", "--output", "./ok"])
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_ppo_data_traversal():
    result = runner.invoke(app, ["ppo", "--policy-model", "gpt2", "--reward-model", "./ok", "--data", "../unsafe.csv", "--output", "./ok"])
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_evaluate_data_traversal():
    result = runner.invoke(app, ["evaluate", "--model", "gpt2", "--data", "../unsafe.csv"])
    assert "❌ Path traversal attempt detected." in result.stderr
