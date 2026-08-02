
import pytest
from unittest.mock import MagicMock, patch
import os

from export.hub import push_to_hub
from export.registry import on_registry_upload, on_registry_list

def test_push_to_hub_token_security():
    # Test with null byte in token
    with patch("os.path.isdir", return_value=True):
        result = push_to_hub("./model", "user/repo", "hf_valid_token\0")
        assert "❌ Path traversal attempt detected." in result

    # Test with traversal sequence in token
    with patch("os.path.isdir", return_value=True):
        result = push_to_hub("./model", "user/repo", "hf_valid_token..")
        assert "❌ Path traversal attempt detected." in result

    # Test with backslash in token
    with patch("os.path.isdir", return_value=True):
        result = push_to_hub("./model", "user/repo", "hf_valid\\token")
        assert "❌ Path traversal attempt detected." in result

def test_on_registry_upload_token_security():
    # Test with null byte in token
    with patch("os.path.isdir", return_value=True):
        result = on_registry_upload("./model", "user/repo", "hf_valid_token\0", "v1", "notes")
        assert "❌ Path traversal attempt detected." in result

    # Test with traversal sequence in token
    with patch("os.path.isdir", return_value=True):
        result = on_registry_upload("./model", "user/repo", "hf_valid_token..", "v1", "notes")
        assert "❌ Path traversal attempt detected." in result

def test_on_registry_list_token_security():
    # Test with null byte in token
    result = on_registry_list("user/repo", "hf_valid_token\0")
    assert "❌ Path traversal attempt detected." in result

    # Test with traversal sequence in token
    result = on_registry_list("user/repo", "hf_valid_token..")
    assert "❌ Path traversal attempt detected." in result

def test_push_to_hub_model_path_security_no_mock():
    # Verify that model_path path traversal is rejected immediately without os.path.isdir mock
    result = push_to_hub("../evil_path", "user/repo", "hf_valid_token")
    assert "❌ Path traversal attempt detected." in result

    result = push_to_hub("model_dir/\0", "user/repo", "hf_valid_token")
    assert "❌ Path traversal attempt detected." in result

def test_push_to_hub_token_redaction_in_exceptions():
    # Mock HfApi to raise an exception containing the token
    with patch("os.path.isdir", return_value=True):
        with patch("huggingface_hub.HfApi") as MockApi:
            mock_api_instance = MockApi.return_value
            # Make upload_folder raise an exception containing the sensitive token
            mock_api_instance.upload_folder.side_effect = Exception("Failed with token hf_valid_token_36_characters_minimum_len")

            result = push_to_hub("model_dir", "user/repo", "hf_valid_token_36_characters_minimum_len")
            assert "[REDACTED]" in result
            assert "hf_valid_token_36_characters_minimum_len" not in result

def test_on_registry_upload_token_redaction_in_exceptions():
    with patch("os.path.isdir", return_value=True):
        with patch("export.registry.ModelRegistry") as MockRegistry:
            mock_registry_instance = MockRegistry.return_value
            mock_registry_instance.upload_model.side_effect = Exception("Upload error for token hf_valid_token_36_characters_minimum_len")

            result = on_registry_upload("model_dir", "user/repo", "hf_valid_token_36_characters_minimum_len", "v1.0", "notes")
            assert "[REDACTED]" in result
            assert "hf_valid_token_36_characters_minimum_len" not in result

def test_on_registry_list_token_redaction_in_exceptions():
    with patch("export.registry.ModelRegistry") as MockRegistry:
        mock_registry_instance = MockRegistry.return_value
        mock_registry_instance.list_versions.side_effect = Exception("List error with hf_valid_token_36_characters_minimum_len")

        result = on_registry_list("user/repo", "hf_valid_token_36_characters_minimum_len")
        assert "[REDACTED]" in result
        assert "hf_valid_token_36_characters_minimum_len" not in result


def test_redact_sensitive_info_utility():
    from core.state import redact_sensitive_info

    # Test standard token redaction
    assert redact_sensitive_info("error hf_AbCdEfGhIjKlMnOpQrStUvWxYz12345678") == "error [REDACTED]"
    # Test shorter non-tokens are not redacted
    assert redact_sensitive_info("error hf_abc123") == "error hf_abc123"
    # Test none or empty string handling
    assert redact_sensitive_info("") == ""
    assert redact_sensitive_info(None) is None

    # Test environment variable HF_TOKEN redaction
    with patch.dict(os.environ, {"HF_TOKEN": "my_secret_token_123456"}):
        assert redact_sensitive_info("failed to authenticate with my_secret_token_123456") == "failed to authenticate with [REDACTED]"


def test_handlers_on_train_click_redacts_tokens():
    from ui.handlers import on_train_click

    # If loading file throws an exception containing a token, it should be redacted
    mock_file = MagicMock()
    mock_file.name = "valid_name.csv"

    # We patch detect_file_type and load_dataset_from_file to raise an exception containing a token
    with patch("ui.handlers.detect_file_type", return_value="csv"):
        with patch("ui.handlers.load_dataset_from_file", side_effect=Exception("Failed to access HF with hf_AbCdEfGhIjKlMnOpQrStUvWxYz12345678")):
            res, _, _, _ = on_train_click(
                mock_file, "gpt2", "", "Quick (1 epoch)", "LoRA",
                True, 8, 16, 30, 512, 2, 20, 16, 1.4e-5, 1, 1, 1, 256, 100,
                3, "cosine", True, False, "col_inst", "col_out", "col_text",
                False, False, "System prompt", "sft", 0.1, False
            )
            assert "[REDACTED]" in res
            assert "hf_AbCdEfGhIjKlMnOpQrStUvWxYz12345678" not in res
