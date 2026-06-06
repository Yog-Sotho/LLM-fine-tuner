import sys
from unittest.mock import MagicMock

# Mock missing dependencies
mock_modules = [
    "torch", "transformers", "peft", "gradio", "huggingface_hub", "vllm", "datasets", "numpy", "pandas", "PyPDF2", "openpyxl"
]
for mod in mock_modules:
    sys.modules[mod] = MagicMock()

# Now we can import our modules
from export.utils import validate_hf_token
from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate
import os

def test_token_validation():
    print("Testing token validation...")
    assert validate_hf_token("hf_123456789012345678901234567890123456") is None
    assert "Invalid Hugging Face write token" in validate_hf_token("short")
    assert "Invalid Hugging Face write token" in validate_hf_token("no_prefix_123456789012345678901234567890123456")
    print("✅ Token validation tests passed.")

def test_gguf_quantization_hardening():
    print("Testing GGUF quantization hardening...")
    # Test path traversal in quantization
    status, file_path = on_export_gguf("valid_dir", "../unsafe")
    assert "Invalid quantization format" in status

    status, file_path = on_export_gguf("valid_dir", "q4_k_m/..")
    assert "Invalid quantization format" in status

    # Test forward slash in quantization
    status, file_path = on_export_gguf("valid_dir", "q4_k/m")
    assert "Invalid quantization format" in status

    print("✅ GGUF quantization hardening tests passed.")

def test_vllm_quantization_hardening():
    print("Testing vLLM quantization hardening...")
    # Test path traversal in quantization
    status = on_vllm_generate("valid_dir", "prompt", "../unsafe", 100, 0.7, 0.9)
    assert "Path traversal attempt detected" in status

    status = on_vllm_generate("valid_dir", "prompt", "quant\\..", 100, 0.7, 0.9)
    assert "Path traversal attempt detected" in status
    print("✅ vLLM quantization hardening tests passed.")

if __name__ == "__main__":
    try:
        test_token_validation()
        test_gguf_quantization_hardening()
        test_vllm_quantization_hardening()
        print("\nAll security verification tests passed! 🛡️")
    except AssertionError as e:
        print(f"\n❌ Test failed!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
