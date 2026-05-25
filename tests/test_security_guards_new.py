
import os
import sys

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from export.gguf import on_export_gguf
from inference.vllm_runner import on_vllm_generate

def test_on_export_gguf_traversal():
    print("Testing on_export_gguf for path traversal...")
    # Test traversal
    status, gguf = on_export_gguf("../unsafe", "q6_k")
    assert status == "❌ Path traversal attempt detected."
    assert gguf is None

    # Test whitespace stripping + traversal
    status, gguf = on_export_gguf("  ../unsafe  ", "q6_k")
    assert status == "❌ Path traversal attempt detected."

    # Test valid-ish path (safe from traversal but doesn't exist)
    status, gguf = on_export_gguf("  /non/existent/path  ", "q6_k")
    assert "No trained model found" in status
    print("✅ on_export_gguf security check passed.")

def test_on_vllm_generate_traversal():
    print("Testing on_vllm_generate for path traversal...")
    # Test traversal
    status = on_vllm_generate("../unsafe", "prompt", "none", 128, 0.7, 0.9)
    assert status == "❌ Path traversal attempt detected."

    # Test whitespace stripping + traversal
    status = on_vllm_generate("  ../unsafe  ", "prompt", "none", 128, 0.7, 0.9)
    assert status == "❌ Path traversal attempt detected."

    # Test valid-ish path (safe from traversal but doesn't exist)
    status = on_vllm_generate("  /non/existent/path  ", "prompt", "none", 128, 0.7, 0.9)
    # Could be "vLLM not installed" or "No trained model path found"
    assert "Path traversal attempt detected" not in status
    print("✅ on_vllm_generate security check passed.")
