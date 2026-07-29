
from unittest.mock import MagicMock, patch

import pytest

from export.hub import push_to_hub
from export.registry import on_registry_list, on_registry_upload


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
    from typer.testing import CliRunner

    from cli.commands import app

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
    # Test path traversal in model_name
    with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
        _load_for_inference("../unsafe_path", None)

    # Test path traversal in lora_path
    with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
        _load_for_inference("gpt2", "../../unsafe_lora")

    # Test null byte injection in model_name
    with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
        _load_for_inference("gpt2\0", None)

    # Test null byte injection in lora_path
    with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
        _load_for_inference("gpt2", "lora_path\0")

def test_batch_generate_security():
    from inference.generate import batch_generate
    mock_file = MagicMock()

    # Test path traversal pattern
    mock_file.name = "unsafe_../file.csv"
    result = batch_generate("gpt2", None, mock_file)
    assert "❌ Path traversal attempt detected." in result

    # Test backslash traversal pattern
    mock_file.name = "unsafe_\\..\\file.csv"
    result = batch_generate("gpt2", None, mock_file)
    assert "❌ Path traversal attempt detected." in result

    # Test null byte injection
    mock_file.name = "unsafe_\0_file.csv"
    result = batch_generate("gpt2", None, mock_file)
    assert "❌ Path traversal attempt detected." in result


def test_model_registry_class_security():
    from export.registry import ModelRegistry
    # Test path traversal in __init__ repo_id
    with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
        ModelRegistry("../unsafe_repo", "hf_valid_token_36_characters_minimum_length")

    # Test path traversal in __init__ token
    with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
        ModelRegistry("user/repo", "hf_unsafe_../token_value")

    # Test null byte injection in upload_model model_path
    reg = ModelRegistry("user/repo", "hf_valid_token_36_characters_minimum_length")
    res = reg.upload_model("unsafe_\0_path", "v1.0", {})
    assert "❌ Path traversal attempt detected." in res

    # Test path traversal / injection in upload_model version tag
    res = reg.upload_model("model_dir", "v1../0", {})
    assert "❌ Path traversal attempt detected." in res


def test_vllm_runner_security():
    from inference.vllm_runner import merge_adapter_for_inference, vllm_generate_v27

    # Test merge_adapter_for_inference with path traversal
    res = merge_adapter_for_inference("../unsafe_base", "adapter_dir", "output_dir")
    assert "❌ Path traversal attempt detected." in res

    res = merge_adapter_for_inference("gpt2", "../../unsafe_adapter", "output_dir")
    assert "❌ Path traversal attempt detected." in res

    res = merge_adapter_for_inference("gpt2", "adapter_dir", "output_\0_dir")
    assert "❌ Path traversal attempt detected." in res

    # Test vllm_generate_v27 with path traversal / null byte / identifier injection
    with patch("config.constants.HAS_VLLM", True):
        with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
            vllm_generate_v27("../unsafe_path", ["prompt"])

        with pytest.raises(ValueError, match="❌ Path traversal attempt detected."):
            vllm_generate_v27("gpt2", ["prompt"], vllm_quantization="awq/../")
