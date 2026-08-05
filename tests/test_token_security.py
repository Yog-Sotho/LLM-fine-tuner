
import os
from unittest.mock import patch

from export.hub import push_to_hub
from export.registry import on_registry_list, on_registry_upload


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

    # 1. Test redacting valid-looking Hugging Face token in string
    sample_text = "Connection failed with token hf_abc123xyz789012345678901234567890abc"
    redacted = redact_sensitive_info(sample_text)
    assert "[REDACTED]" in redacted
    assert "hf_abc123xyz789012345678901234567890abc" not in redacted

    # 2. Test redacting environment HF_TOKEN
    with patch.dict(os.environ, {"HF_TOKEN": "my_secret_env_token_value_xyz"}):
        sample_text_2 = "Error uploading to repo using token my_secret_env_token_value_xyz"
        redacted_2 = redact_sensitive_info(sample_text_2)
        assert "[REDACTED]" in redacted_2
        assert "my_secret_env_token_value_xyz" not in redacted_2

    # 3. Test handling None or empty strings
    assert redact_sensitive_info(None) == ""
    assert redact_sensitive_info("") == ""
