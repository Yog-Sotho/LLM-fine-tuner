
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

from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate

def test_on_export_gguf_hardening():
    print("Testing on_export_gguf hardening...")

    # 1. Test traversal in model_path
    status, _ = on_export_gguf("../unsafe", "q6_k")
    assert "❌ Path traversal attempt detected." in status

    # 2. Test traversal in quantization (..)
    status, _ = on_export_gguf("./ok", "q6_k/../../etc/passwd")
    assert "❌ Path traversal attempt detected." in status

    # 3. Test traversal in quantization (\)
    status, _ = on_export_gguf("./ok", "q6_k\\..\\..\\etc\\passwd")
    assert "❌ Path traversal attempt detected." in status

    # 4. Test slash in quantization
    status, _ = on_export_gguf("./ok", "sub/dir")
    assert "❌ Path traversal attempt detected." in status

    # 5. Test whitespace stripping
    with patch("os.path.isdir", return_value=False):
        status, _ = on_export_gguf("  /tmp/model  ", "  q4_k  ")
        # Should reach isdir check after stripping
        assert "❌ No trained model found." in status

    print("✅ on_export_gguf tests passed!")

def test_on_vllm_generate_hardening():
    print("Testing on_vllm_generate hardening...")

    # 1. Test traversal in model_path
    status = on_vllm_generate("../unsafe", "prompt", "none", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

    # 2. Test traversal in vllm_quant (..)
    status = on_vllm_generate("./ok", "prompt", "awq/../traversal", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

    # 3. Test traversal in vllm_quant (\)
    status = on_vllm_generate("./ok", "prompt", "awq\\..\\traversal", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

    # 4. Test slash in vllm_quant
    status = on_vllm_generate("./ok", "prompt", "illegal/slash", 128, 0.7, 0.9)
    assert "❌ Path traversal attempt detected." in status

    # 5. Test whitespace stripping
    with patch("inference.vllm_runner.HAS_VLLM", True):
        with patch("os.path.isdir", return_value=False):
            status = on_vllm_generate("  /tmp/model  ", "prompt", "  bnb  ", 128, 0.7, 0.9)
            # Should reach isdir check after stripping
            assert "❌ No trained model path found." in status

    print("✅ on_vllm_generate tests passed!")

if __name__ == "__main__":
    try:
        test_on_export_gguf_hardening()
        test_on_vllm_generate_hardening()
        print("\n🎉 All hardening verification tests passed!")
    except AssertionError as e:
        print(f"\n❌ Verification failed!")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
