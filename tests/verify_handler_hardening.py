
import sys
import os
from unittest.mock import MagicMock, patch

# Mock torch and other heavy dependencies before they are imported by the modules under test
sys.modules["torch"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()
sys.modules["gradio"] = MagicMock()
sys.modules["vllm"] = MagicMock()
sys.modules["pandas"] = MagicMock()
sys.modules["datasets"] = MagicMock()
sys.modules["unsloth"] = MagicMock()
sys.modules["PyPDF2"] = MagicMock()
sys.modules["openpyxl"] = MagicMock()
sys.modules["numpy"] = MagicMock()

# Now we can import the handlers
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate

def test_gguf_quant_hardening():
    print("Testing GGUF quantization hardening...")

    # Test path traversal in quantization
    status, file_path = on_export_gguf("valid_dir", "../unsafe")
    assert "❌ Path traversal attempt detected." in status
    print("✅ GGUF: Blocked '..' in quantization")

    # Test backslash in quantization
    status, file_path = on_export_gguf("valid_dir", "unsafe\\path")
    assert "❌ Path traversal attempt detected." in status
    print("✅ GGUF: Blocked '\\' in quantization")

    # Test forward slash in quantization
    status, file_path = on_export_gguf("valid_dir", "unsafe/path")
    assert "❌ Path traversal attempt detected." in status
    print("✅ GGUF: Blocked '/' in quantization")

    # Test whitespace stripping
    with patch("os.path.isdir", return_value=False):
        status, file_path = on_export_gguf(" valid_dir ", " q6_k ")
        # If it stripped correctly and passed traversal, it should hit the isdir check
        assert "❌ No trained model found." in status
    print("✅ GGUF: Stripped whitespace correctly")

def test_vllm_quant_hardening():
    print("\nTesting vLLM quantization hardening...")

    # Test path traversal in quantization
    status = on_vllm_generate("valid_dir", "prompt", "../unsafe", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status
    print("✅ vLLM: Blocked '..' in quantization")

    # Test backslash in quantization
    status = on_vllm_generate("valid_dir", "prompt", "unsafe\\path", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status
    print("✅ vLLM: Blocked '\\' in quantization")

    # Test forward slash in quantization
    status = on_vllm_generate("valid_dir", "prompt", "unsafe/path", 512, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status
    print("✅ vLLM: Blocked '/' in quantization")

    # Test whitespace stripping
    with patch("inference.vllm_runner.HAS_VLLM", True):
        with patch("os.path.isdir", return_value=False):
            status = on_vllm_generate(" valid_dir ", "prompt", " none ", 512, 0.7, 0.9)
            # If it stripped correctly and passed traversal, it should hit the isdir check
            assert "❌ No trained model path found." in status
    print("✅ vLLM: Stripped whitespace correctly")

if __name__ == "__main__":
    try:
        test_gguf_quant_hardening()
        test_vllm_quant_hardening()
        print("\n✨ All handler hardening tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        sys.exit(1)
