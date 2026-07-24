
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

def test_on_peft_zip_upload_traversal():
    from export.utils import on_peft_zip_upload
    mock_file = MagicMock()
    mock_file.name = "unsafe_../file.zip"
    _, status, _ = on_peft_zip_upload(mock_file)
    assert "❌ Path traversal attempt detected." in status

def test_on_batch_test_file_traversal():
    from ui.handlers import on_batch_test
    mock_file = MagicMock()
    mock_file.name = "unsafe_\\..\\file.csv"
    status = on_batch_test(mock_file, "gpt2", "", "")
    assert "❌ Path traversal attempt detected." in status

def test_on_evaluate_click_file_traversal():
    from inference.evaluation import on_evaluate_click
    mock_file = MagicMock()
    mock_file.name = "unsafe_\0_file.csv"
    status, _, _ = on_evaluate_click("gpt2", "", "", mock_file, False, False, "", "")
    assert "❌ Path traversal attempt detected." in status

def test_load_for_inference_security():
    from inference.generate import _load_for_inference

    with pytest.raises(ValueError, match="Path traversal attempt detected"):
        _load_for_inference("../unsafe_model", None)

    with pytest.raises(ValueError, match="Path traversal attempt detected"):
        _load_for_inference("safe_model", "unsafe_../path")

    with pytest.raises(ValueError, match="Path traversal attempt detected"):
        _load_for_inference("safe_model\0", None)
