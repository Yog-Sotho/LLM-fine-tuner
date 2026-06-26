
import pytest
from core.state import validate_path_traversal, validate_identifier

def test_validate_path_traversal_null_byte():
    assert validate_path_traversal("path\0with\0null") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("normal/path") is None
    assert validate_path_traversal("..") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("\\") == "❌ Path traversal attempt detected."

def test_validate_identifier_robust():
    assert validate_identifier("q4_k_m") is None
    assert validate_identifier("v1.0") is None
    assert validate_identifier("user/repo") == "❌ Path traversal attempt detected."
    assert validate_identifier("..") == "❌ Path traversal attempt detected."
    assert validate_identifier("\\") == "❌ Path traversal attempt detected."
    assert validate_identifier("id\0") == "❌ Path traversal attempt detected."

from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate
from export.registry import on_registry_upload
from unittest.mock import patch, MagicMock

def test_on_export_gguf_slash_in_quant():
    status, file_path = on_export_gguf("some_path", "q4/k")
    assert "❌ Path traversal attempt detected." in status

def test_on_vllm_generate_slash_in_quant():
    status = on_vllm_generate("some_path", "prompt", "awq/none", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

def test_on_registry_upload_traversal_in_version():
    with patch("os.path.isdir", return_value=True):
        status = on_registry_upload("path", "user/repo", "hf_token_is_at_least_36_chars_long_xxx", "v1/0", "notes")
        assert "❌ Path traversal attempt detected." in status
