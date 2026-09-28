from typer.testing import CliRunner

from cli.commands import app

runner = CliRunner()


def test_cli_train_path_traversal():
    result = runner.invoke(
        app, ["train", "--model", "../unsafe", "--data", "dummy.csv", "--output", "./ok"]
    )
    assert result.exit_code == 1
    # Check stderr because typer.echo(..., err=True) writes to stderr
    assert "❌ Path traversal attempt detected." in result.stderr

    result = runner.invoke(
        app, ["train", "--model", "ok", "--data", "dummy.csv", "--output", "../unsafe"]
    )
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr


def test_cli_reward_path_traversal():
    result = runner.invoke(
        app, ["reward", "--model", "../unsafe", "--data", "dummy.csv", "--output", "./ok"]
    )
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr


def test_cli_orpo_path_traversal():
    result = runner.invoke(
        app, ["orpo", "--model", "../unsafe", "--data", "dummy.csv", "--output", "./ok"]
    )
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr


def test_cli_grpo_path_traversal():
    result = runner.invoke(
        app,
        [
            "grpo",
            "--policy-model",
            "../unsafe",
            "--reward-model",
            "./ok",
            "--data",
            "dummy.csv",
            "--output",
            "./ok",
        ],
    )
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr


def test_cli_evaluate_path_traversal():
    result = runner.invoke(app, ["evaluate", "--model", "../unsafe", "--data", "dummy.csv"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr


def test_cli_kto_path_traversal():
    result = runner.invoke(
        app, ["kto", "--model", "../unsafe", "--data", "dummy.csv", "--output", "./ok"]
    )
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr


def test_cli_benchmark_path_traversal():
    result = runner.invoke(app, ["benchmark", "--model", "gpt2", "--output", "../scores.csv"])
    assert result.exit_code == 1
    assert "❌ Path traversal attempt detected." in result.stderr
