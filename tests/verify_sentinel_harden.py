
import os
import sys
from unittest.mock import MagicMock, patch

# Add current directory to path
sys.path.append(os.getcwd())

# Mock machine learning libraries that might not be installed
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['vllm'] = MagicMock()
sys.modules['unsloth'] = MagicMock()
sys.modules['gradio'] = MagicMock()
sys.modules['datasets'] = MagicMock()
sys.modules['pandas'] = MagicMock()
sys.modules['numpy'] = MagicMock()
sys.modules['PyPDF2'] = MagicMock()

from core.state import validate_path_traversal, validate_identifier
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate
from export.registry import on_registry_upload

def test_core_state_hardening():
    print("Testing core.state hardening...")

    # validate_path_traversal
    assert validate_path_traversal("ok/path") is None
    assert "❌" in validate_path_traversal("../unsafe")
    assert "❌" in validate_path_traversal("path\\unsafe")
    assert "❌" in validate_path_traversal("path\0unsafe")

    # validate_identifier
    assert validate_identifier("ok_id") is None
    assert "❌" in validate_identifier("sub/dir")
    assert "❌" in validate_identifier("../unsafe")
    assert "❌" in validate_identifier("path\\unsafe")
    assert "❌" in validate_identifier("id\0unsafe")

    print("✅ core.state tests passed!")

def test_on_export_gguf_harden():
    print("Testing on_export_gguf hardening (null byte)...")
    # Null byte in model_path
    status, _ = on_export_gguf("model\0path", "q6_k")
    assert "❌ Path traversal attempt detected." in status

    # Null byte in quantization
    status, _ = on_export_gguf("./ok", "q6\0k")
    assert "❌ Path traversal attempt detected." in status
    print("✅ on_export_gguf tests passed!")

def test_on_vllm_generate_harden():
    print("Testing on_vllm_generate hardening (null byte)...")
    # Null byte in model_path
    status = on_vllm_generate("model\0path", "prompt", "none", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

    # Null byte in vllm_quant
    status = on_vllm_generate("./ok", "prompt", "awq\0", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status
    print("✅ on_vllm_generate tests passed!")

def test_on_registry_upload_harden():
    print("Testing on_registry_upload hardening...")
    # Traversal in version
    status = on_registry_upload("./ok", "user/repo", "hf_123456789012345678901234567890123456", "../v1", "notes")
    assert "❌ Path traversal attempt detected." in status

    # Slash in version
    status = on_registry_upload("./ok", "user/repo", "hf_123456789012345678901234567890123456", "v1/v2", "notes")
    assert "❌ Path traversal attempt detected." in status

    # Null byte in version
    status = on_registry_upload("./ok", "user/repo", "hf_123456789012345678901234567890123456", "v1\0", "notes")
    assert "❌ Path traversal attempt detected." in status

    print("✅ on_registry_upload tests passed!")

if __name__ == "__main__":
    try:
        test_core_state_hardening()
        test_on_export_gguf_harden()
        test_on_vllm_generate_harden()
        test_on_registry_upload_harden()
        print("\n🎉 All sentinel hardening verification tests passed!")
    except AssertionError as e:
        print(f"\n❌ Verification failed!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
