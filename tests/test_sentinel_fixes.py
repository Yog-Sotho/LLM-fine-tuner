
import pytest
from unittest.mock import MagicMock, patch
import os
from export.hub import push_to_hub
from export.registry import on_registry_upload, on_registry_list

def test_push_to_hub_token_traversal():
    with patch("os.path.isdir", return_value=True):
        status = push_to_hub("model_dir", "user/repo", "hf_token_with_traversal_../")
        assert "❌ Path traversal attempt detected." in status

def test_on_registry_upload_token_traversal():
    status = on_registry_upload("model_dir", "user/repo", "hf_../token", "1.0", "notes")
    assert "❌ Path traversal attempt detected." in status

def test_on_registry_list_token_traversal():
    status = on_registry_list("user/repo", "hf_\0_token")
    assert "❌ Path traversal attempt detected." in status

def test_cli_data_path_traversal():
    from cli.commands import app
    from typer.testing import CliRunner

    runner = CliRunner()

    # Test train command
    result = runner.invoke(app, ["train", "--model", "gpt2", "--data", "../secret.csv", "--output", "./out"])
    assert "❌ Path traversal attempt detected." in result.stderr

    # Test reward command
    result = runner.invoke(app, ["reward", "--model", "gpt2", "--data", "data/../secret.csv", "--output", "./out"])
    assert "❌ Path traversal attempt detected." in result.stderr

    # Test orpo command
    result = runner.invoke(app, ["orpo", "--model", "gpt2", "--data", r"..\..\Windows\System32\config\SAM", "--output", "./out"])
    assert "❌ Path traversal attempt detected." in result.stderr

    # Test ppo command
    result = runner.invoke(app, ["ppo", "--policy-model", "gpt2", "--reward-model", "./rm", "--data", "data/path/../../etc/passwd", "--output", "./out"])
    assert "❌ Path traversal attempt detected." in result.stderr

    # Test evaluate command
    result = runner.invoke(app, ["evaluate", "--model", "gpt2", "--data", "/absolute/../path.csv"])
    assert "❌ Path traversal attempt detected." in result.stderr
