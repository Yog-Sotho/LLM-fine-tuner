
import pytest
from typer.testing import CliRunner
from cli.commands import app

runner = CliRunner()

def test_cli_train_path_traversal():
    result = runner.invoke(app, ["train", "--model", "../unsafe", "--data", "dummy.csv", "--output", "./ok"])
    assert result.exit_code == 1
    # Check stderr because typer.echo(..., err=True) writes to stderr
    assert "❌ Path traversal attempt detected." in result.stderr

    result = runner.invoke(app, ["train", "--model", "ok", "--data", "dummy.csv", "--output", "../unsafe"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_reward_path_traversal():
    result = runner.invoke(app, ["reward", "--model", "../unsafe", "--data", "dummy.csv", "--output", "./ok"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_orpo_path_traversal():
    result = runner.invoke(app, ["orpo", "--model", "../unsafe", "--data", "dummy.csv", "--output", "./ok"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_ppo_path_traversal():
    result = runner.invoke(app, ["ppo", "--policy-model", "../unsafe", "--reward-model", "./ok", "--data", "dummy.csv", "--output", "./ok"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_evaluate_path_traversal():
    result = runner.invoke(app, ["evaluate", "--model", "../unsafe", "--data", "dummy.csv"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr


def test_cli_data_path_traversal():
    """Sentinel: Verify that the --data parameter is also hardened against traversal."""
    # Test train
    result = runner.invoke(app, ["train", "--model", "model", "--data", "../secret.csv", "--output", "./out"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr

    # Test evaluate
    result = runner.invoke(app, ["evaluate", "--model", "model", "--data", "/etc/passwd\0.csv"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr
