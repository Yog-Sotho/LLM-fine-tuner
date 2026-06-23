
import pytest
from core.state import validate_path_traversal, validate_identifier
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate
from export.registry import on_registry_upload
from export.hub import push_to_hub
from unittest.mock import patch

def test_validate_path_traversal_null_byte():
    assert validate_path_traversal("normal/path") is None
    assert validate_path_traversal("path\0with/null") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("../traversal") == "❌ Path traversal attempt detected."
    assert validate_path_traversal("C:\\Windows") == "❌ Path traversal attempt detected."

def test_validate_identifier():
    assert validate_identifier("q4_k_m") is None
    assert validate_identifier("v1.0") is None
    assert validate_identifier("path/with/slash") == "❌ Path traversal attempt detected."
    assert validate_identifier("..") == "❌ Path traversal attempt detected."
    assert validate_identifier("back\\slash") == "❌ Path traversal attempt detected."
    assert validate_identifier("null\0byte") == "❌ Path traversal attempt detected."

def test_handlers_null_byte():
    # GGUF
    status, _ = on_export_gguf("model\0path", "q4_k_m")
    assert status == "❌ Path traversal attempt detected."
    status, _ = on_export_gguf("model", "q4\0k")
    assert status == "❌ Path traversal attempt detected."

    # vLLM
    status = on_vllm_generate("model\0path", "prompt", "none", 128, 0.7, 0.9)
    assert status == "❌ Path traversal attempt detected."
    status = on_vllm_generate("model", "prompt", "none\0", 128, 0.7, 0.9)
    assert status == "❌ Path traversal attempt detected."

    # Registry
    status = on_registry_upload("model", "repo", "hf_123", "v1\0", "notes")
    assert status == "❌ Path traversal attempt detected."
    status = on_registry_upload("model", "repo", "hf_123\0", "v1", "notes")
    assert status == "❌ Path traversal attempt detected."

    # Hub
    status = push_to_hub("model", "repo", "hf_123\0")
    assert status == "❌ Path traversal attempt detected."

def test_identifier_slash_blocking():
    # GGUF quantization
    status, _ = on_export_gguf("model", "q4/k")
    assert status == "❌ Path traversal attempt detected."

    # vLLM quantization
    status = on_vllm_generate("model", "prompt", "awq/unsafe", 128, 0.7, 0.9)
    assert status == "❌ Path traversal attempt detected."

    # Registry version
    status = on_registry_upload("model", "repo", "hf_token_valid_prefix_and_length_36", "v1/safe", "notes")
    assert status == "❌ Path traversal attempt detected."
