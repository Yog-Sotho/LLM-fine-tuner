
import pytest
from typer.testing import CliRunner
from cli.commands import app
from export.hub import push_to_hub
from export.registry import on_registry_upload, on_registry_list
import os

runner = CliRunner()

def test_cli_data_path_traversal():
    commands = [
        ["train", "--model", "ok", "--data", "../unsafe.csv", "--output", "./ok"],
        ["reward", "--model", "ok", "--data", "../unsafe.csv", "--output", "./ok"],
        ["orpo", "--model", "ok", "--data", "../unsafe.csv", "--output", "./ok"],
        ["ppo", "--policy-model", "ok", "--reward-model", "./ok", "--data", "../unsafe.csv", "--output", "./ok"],
        ["evaluate", "--model", "ok", "--data", "../unsafe.csv"]
    ]
    for cmd in commands:
        result = runner.invoke(app, cmd)
        assert result.exit_code == 1, f"Command {cmd[0]} failed to block path traversal in --data"
        assert "❌ Path traversal attempt detected." in result.stderr

def test_cli_data_null_byte():
    commands = [
        ["train", "--model", "ok", "--data", "unsafe\0.csv", "--output", "./ok"],
        ["reward", "--model", "ok", "--data", "unsafe\0.csv", "--output", "./ok"],
        ["orpo", "--model", "ok", "--data", "unsafe\0.csv", "--output", "./ok"],
        ["ppo", "--policy-model", "ok", "--reward-model", "./ok", "--data", "unsafe\0.csv", "--output", "./ok"],
        ["evaluate", "--model", "ok", "--data", "unsafe\0.csv"]
    ]
    for cmd in commands:
        result = runner.invoke(app, cmd)
        assert result.exit_code == 1, f"Command {cmd[0]} failed to block null byte in --data"
        assert "❌ Path traversal attempt detected." in result.stderr

def test_hub_push_token_traversal():
    # Test path traversal pattern in token
    result = push_to_hub("./model", "user/repo", "../unsafe_token")
    assert "❌ Path traversal attempt detected." in result

    # Test null byte in token
    result = push_to_hub("./model", "user/repo", "hf_token\0secret")
    assert "❌ Path traversal attempt detected." in result

def test_registry_upload_token_traversal():
    # Test path traversal pattern in token
    result = on_registry_upload("./model", "user/repo", "../unsafe_token", "1.0", "notes")
    assert "❌ Path traversal attempt detected." in result

    # Test null byte in token
    result = on_registry_upload("./model", "user/repo", "hf_token\0secret", "1.0", "notes")
    assert "❌ Path traversal attempt detected." in result

def test_registry_list_token_traversal():
    # Test path traversal pattern in token
    result = on_registry_list("user/repo", "../unsafe_token")
    assert "❌ Path traversal attempt detected." in result

    # Test null byte in token
    result = on_registry_list("user/repo", "hf_token\0secret")
    assert "❌ Path traversal attempt detected." in result
